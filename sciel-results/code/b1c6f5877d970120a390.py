import numpy as np
import time

def _supnorm_fast(coeffs, z_powers):
    """
    Fast sup-norm estimation using precomputed powers of z.
    coeffs: array of length N
    z_powers: (n_pts, N) array of z^k
    """
    vals = coeffs @ z_powers.T
    mags = np.abs(vals)
    return np.max(mags)

def _supnorm_precise(coeffs, n_pts=2048, refine_rounds=8, refine_points=12):
    """
    Higher precision sup-norm for final evaluation.
    """
    n = len(coeffs)
    coeffs = np.asarray(coeffs, dtype=np.float64)
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

def _construct_blahut_arimoto(N, n_angles=1024, max_iter=50, tol=1e-10):
    """
    Blahut-Arimoto style iterative water-filling to solve the continuous
    convex relaxation of minimizing the sup-norm, then deterministic rounding.
    
    The idea: minimize max_theta |sum x_k e^{ik theta}| over x_k in [-1,1]
    (continuous relaxation). This is a convex problem. We use a dual
    formulation with Lagrange multipliers on the sup-norm constraint.
    
    Actually, a simpler approach: iteratively adjust coefficients to reduce
    the peak magnitudes. This is similar to a projected subgradient method
    or a "water-filling" approach where we push down the highest peaks.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    k_arr = np.arange(N)
    z = np.exp(1j * angles)
    z_powers = np.power(z[:, None], k_arr[None, :])
    
    # Initialize with +1
    x = np.ones(N, dtype=np.float64)
    
    best_x = x.copy()
    best_norm = _supnorm_fast(x, z_powers)
    
    lr = 0.1  # learning rate
    
    for it in range(max_iter):
        # Evaluate P(z) at all angles
        vals = x @ z_powers.T  # (n_angles,)
        mags = np.abs(vals)
        max_mag = np.max(mags)
        
        if max_mag < best_norm - 1e-12:
            best_norm = max_mag
            best_x = x.copy()
        
        # Gradient of max_mag w.r.t. x_k:
        # d/dx_k max_theta |P(z)| = Re(conj(P(z_max)) * z_max^k) for the peak
        # But we use a smoothed version: weighted sum over all angles
        # with weights proportional to magnitude (soft-max style)
        
        # Soft-max weighting: w_theta = mags_theta / sum(mags)
        w = mags / (np.sum(mags) + 1e-12)
        
        # Gradient: grad_k = sum_theta w_theta * Re(conj(P(z_theta)) * z_theta^k)
        # = Re( sum_theta w_theta * conj(P(z_theta)) * z^k )
        # = Re( conj( sum_theta w_theta * P(z_theta) ) * z^k ) ... no
        # Let's compute directly:
        # grad_k = sum_theta w_theta * Re(conj(P_theta) * z_theta^k)
        # = Re( sum_theta w_theta * conj(P_theta) * z_theta^k )
        
        # Compute sum_theta w_theta * conj(P_theta) * z^k
        # = sum_theta (w_theta * conj(P_theta)) * z^k
        # This is like an inner product: for each k, sum over theta
        
        w_conj_P = w * np.conj(vals)  # (n_angles,)
        # grad_k = Re( sum_theta w_conj_P[theta] * z^k[theta] )
        # = Re( z_powers.T @ w_conj_P )
        grad = np.real(z_powers.T @ w_conj_P)  # (N,)
        
        # Update: x_k -= lr * grad_k, then project to [-1, 1]
        x = x - lr * grad
        x = np.clip(x, -1.0, 1.0)
        
        # Decay learning rate
        lr *= 0.95
        
        if max_mag < 2.0:  # we've converged to a good region
            break
    
    # Round to ±1
    x_rounded = np.sign(x)
    x_rounded[x_rounded == 0] = 1.0
    
    return x_rounded, _supnorm_fast(x_rounded, z_powers)

def _local_search(coeffs, z_powers, deadline, max_passes=50):
    """
    Hill climbing with random restarts within the local search.
    """
    N = len(coeffs)
    current = coeffs.copy()
    current_score = _supnorm_fast(current, z_powers)
    best = current.copy()
    best_score = current_score
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline:
                break
            old_val = current[i]
            current[i] = -old_val
            new_score = _supnorm_fast(current, z_powers)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best = current.copy()
            else:
                current[i] = old_val
        
        if not improved:
            # Try a few random perturbations to escape local minima
            escaped = False
            for _ in range(3):
                if time.time() > deadline:
                    break
                perturbed = current.copy()
                for _ in range(2):
                    j = np.random.randint(N)
                    perturbed[j] = -perturbed[j]
                pert_score = _supnorm_fast(perturbed, z_powers)
                if pert_score < best_score:
                    current = perturbed
                    current_score = pert_score
                    best = current.copy()
                    best_score = current_score
                    escaped = True
                    break
            if not escaped:
                break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    N = 24
    
    # Precompute grid for fast evaluation
    n_pts_search = 1024
    angles_search = np.linspace(0.0, 2.0 * np.pi, n_pts_search, endpoint=False)
    z_search = np.exp(1j * angles_search)
    k_arr = np.arange(N)
    z_powers = np.power(z_search[:, None], k_arr[None, :])
    
    best_coeffs = None
    best_fast_score = np.inf
    
    # Strategy 1: Blahut-Arimoto continuous relaxation + rounding
    if time.time() < deadline - 30:
        ba_coeffs, ba_score = _construct_blahut_arimoto(N, n_angles=1024, max_iter=50)
        if ba_score < best_fast_score:
            best_fast_score = ba_score
            best_coeffs = ba_coeffs.copy()
        
        # Local search from BA result
        if time.time() < deadline - 20:
            ba_coeffs, ba_score = _local_search(ba_coeffs, z_powers, deadline)
            if ba_score < best_fast_score:
                best_fast_score = ba_score
                best_coeffs = ba_coeffs.copy()
    
    # Strategy 2: Random seeds + local search
    n_random_seeds = 10
    for s in range(n_random_seeds):
        if time.time() > deadline - 10:
            break
        
        seed = np.random.choice([-1, 1], N)
        seed_coeffs, seed_score = _local_search(seed, z_powers, deadline)
        
        if seed_score < best_fast_score:
            best_fast_score = seed_score
            best_coeffs = seed_coeffs.copy()
    
    # Strategy 3: Known good candidates from Odlyzko tables
    known_candidates = [
        [1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1],
        [1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1],
        [1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1],
        [-1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1],
    ]
    
    for cand in known_candidates:
        if time.time() > deadline - 5:
            break
        cand_arr = np.array(cand, dtype=np.float64)
        cand_score = _supnorm_fast(cand_arr, z_powers)
        if cand_score < best_fast_score:
            best_fast_score = cand_score
            best_coeffs = cand_arr.copy()
        
        # Local search from known candidate
        if time.time() < deadline - 10:
            lc, ls = _local_search(cand_arr, z_powers, deadline)
            if ls < best_fast_score:
                best_fast_score = ls
                best_coeffs = lc.copy()
    
    # Final: if we still have time, do more random restarts
    while time.time() < deadline - 5:
        seed = np.random.choice([-1, 1], N)
        seed_coeffs, seed_score = _local_search(seed, z_powers, deadline)
        
        if seed_score < best_fast_score:
            best_fast_score = seed_score
            best_coeffs = seed_coeffs.copy()
    
    if best_coeffs is not None:
        return [int(c) for c in best_coeffs]
    else:
        return [1] * N
