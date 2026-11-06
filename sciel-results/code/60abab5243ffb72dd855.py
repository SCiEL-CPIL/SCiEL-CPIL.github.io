import numpy as np
import time

def _supnorm(coeffs, n_pts=4096, refine_rounds=5, refine_points=10):
    """
    Estimate max |P(z)| for |z|=1 using dense sampling + golden-section refinement.
    coeffs: list/array of length n
    """
    n = len(coeffs)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    
    # Dense sampling
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    z = np.exp(1j * angles)
    k = np.arange(n)
    vals = np.sum(coeffs[None, :] * np.power(z[:, None], k[None, :]), axis=1)
    magnitudes = np.abs(vals)
    
    # Find top candidates for refinement
    if n_pts > refine_points:
        top_idx = np.argpartition(magnitudes, -refine_points)[-refine_points:]
    else:
        top_idx = np.arange(n_pts)
    
    best_mag = float(np.max(magnitudes))
    
    # Refine around the top peaks
    window = 2.0 * np.pi / n_pts * 2
    
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    inv_golden = 1.0 - 1.0 / golden_ratio
    
    for idx in top_idx:
        center = angles[idx]
        a = max(0.0, center - window)
        b = min(2.0 * np.pi, center + window)
        
        if b <= a:
            continue
            
        c = b - inv_golden * (b - a)
        d = a + inv_golden * (b - a)
        
        def _eval(angle):
            return np.abs(np.sum(coeffs * np.exp(1j * k * angle)))
        
        fc = _eval(c)
        fd = _eval(d)
        
        for _ in range(refine_rounds):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + inv_golden * (b - a)
                fd = _eval(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - inv_golden * (b - a)
                fc = _eval(c)
        
        fa = _eval(a)
        fb = _eval(b)
        
        local_best = max(fa, fb, fc, fd)
        if local_best > best_mag:
            best_mag = local_best
    
    return best_mag

def _score(coeffs):
    """Score = supnorm / sqrt(n)"""
    n = len(coeffs)
    return _supnorm(coeffs) / np.sqrt(n)

def _fast_supnorm(coeffs, z_powers, angles):
    """
    Fast sup-norm estimation using precomputed powers of z.
    coeffs: array of length N
    z_powers: (n_pts, N) array of z^k
    angles: (n_pts,) array of angles (not used for computation, but kept for signature consistency if needed)
    """
    # coeffs: (N,)
    # z_powers: (n_pts, N)
    # vals = coeffs @ z_powers.T -> (n_pts,)
    vals = coeffs @ z_powers.T
    mags = np.abs(vals)
    return np.max(mags)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    N = 24
    
    # Precompute grid for fast evaluation
    n_pts_search = 1024
    angles_search = np.linspace(0.0, 2.0 * np.pi, n_pts_search, endpoint=False)
    z_search = np.exp(1j * angles_search)
    k_arr = np.arange(N)
    z_powers = np.power(z_search[:, None], k_arr[None, :])
    
    best_coeffs = None
    best_score = np.inf
    
    # Local search strategy
    # We will run multiple restarts.
    # Each restart starts with a random configuration or a known good one.
    # We perform hill climbing: flip one bit at a time if it improves the score.
    
    # Known optimal pattern from Odlyzko (one of them)
    # The task says "Odlyzko's online tables list the optimal polynomials".
    # A common optimal for N=24 is related to Golay sequences or similar.
    # Let's try a few known good seeds and random seeds.
    
    seeds = [
        [1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1], # Rudin-Shapiro variant?
        np.random.choice([-1, 1], N)
    ]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
            
        current_coeffs = np.array(seed, dtype=np.float64)
        current_score = _fast_supnorm(current_coeffs, z_powers, angles_search)
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current_coeffs.copy()
        
        # Local search
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
                
            improved = False
            for i in range(N):
                # Try flipping bit i
                old_val = current_coeffs[i]
                current_coeffs[i] = -old_val
                
                new_score = _fast_supnorm(current_coeffs, z_powers, angles_search)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_coeffs = current_coeffs.copy()
                else:
                    # Revert
                    current_coeffs[i] = old_val
                    
    # Final high-precision check on the best candidate
    if best_coeffs is not None:
        final_score = _score(best_coeffs)
        # Note: The evaluator uses the high precision score.
        # We return the coefficients.
        return [int(c) for c in best_coeffs]
    else:
        # Fallback to random if something went wrong
        return [1] * N
