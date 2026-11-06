import numpy as np
import time
import random

def _supnorm(coeffs, n_pts=4096, refine_rounds=5, refine_points=10):
    """
    Estimate max |P(z)| for |z|=1 using dense sampling + golden-section refinement.
    coeffs: list/array of length n
    """
    n = len(coeffs)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    
    # Dense sampling
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    # Compute P(e^{i theta}) efficiently
    # Use np.exp and matrix multiplication or iterative
    # For n=24, direct is fine
    z = np.exp(1j * angles)
    # P(z) = sum c_k z^k
    # Build Vandermonde-like: shape (n_pts, n)
    # For speed, use broadcasting
    k = np.arange(n)
    # z^k for each angle
    # z[:, None] ** k[None, :]
    vals = np.sum(coeffs[None, :] * np.power(z[:, None], k[None, :]), axis=1)
    magnitudes = np.abs(vals)
    
    # Find top candidates for refinement
    # Get the top refine_points indices
    if n_pts > refine_points:
        # Use argpartition for efficiency
        top_idx = np.argpartition(magnitudes, -refine_points)[-refine_points:]
        top_mags = magnitudes[top_idx]
    else:
        top_idx = np.arange(n_pts)
        top_mags = magnitudes
    
    best_mag = max(magnitudes)
    
    # Refine around the top peaks
    # Golden section search in a small window around each peak
    # Window: one sample spacing on each side
    window = 2.0 * np.pi / n_pts * 2  # 2 sample spacings
    
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    inv_golden = 1.0 - 1.0 / golden_ratio  # ~0.382
    
    for idx in top_idx:
        center = angles[idx]
        # Make sure center is within [0, 2pi)
        a = max(0.0, center - window)
        b = min(2.0 * np.pi, center + window)
        
        if b <= a:
            continue
            
        # Golden section search for maximum of |P(z)| in [a, b]
        c = b - inv_golden * (b - a)
        d = a + inv_golden * (b - a)
        
        fc = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * c)))
        fd = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * d)))
        
        for _ in range(refine_rounds):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + inv_golden * (b - a)
                fd = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * d)))
            else:
                b = d
                d = c
                fd = fc
                c = b - inv_golden * (b - a)
                fc = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * c)))
        
        # Check endpoints too
        fa = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * a)))
        fb = np.abs(np.sum(coeffs * np.exp(1j * np.arange(n) * b)))
        
        local_best = max(fa, fb, fc, fd)
        if local_best > best_mag:
            best_mag = local_best
    
    return best_mag

def _score(coeffs):
    """Score = supnorm / sqrt(n)"""
    n = len(coeffs)
    return _supnorm(coeffs) / np.sqrt(n)

def _rs_sequence(length):
    """Generate Rudin-Shapiro sequence of given length."""
    p, q = [1], [1]
    while len(p) < length:
        p_new = p + q
        q_new = p + [-c for c in q]
        p, q = p_new, q_new
    return p[:length]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    N = 24
    target_score = 1.2583758
    
    # Start with Rudin-Shapiro
    coeffs = _rs_sequence(N)
    coeffs = np.array(coeffs, dtype=np.float64)
    
    best_coeffs = coeffs.copy()
    best_score = _score(best_coeffs)
    
    if best_score <= target_score * (1.0 + 1e-6):
        return [int(c) for c in best_coeffs]
    
    # Local search: flip individual coefficients, then pairs
    rng = np.random.default_rng(42)
    
    # Precompute for speed: we need to evaluate |P(z)| at many points
    # Strategy: use a smaller grid for local search, full grid for final check
    n_pts_fast = 512
    
    def fast_supnorm(c):
        """Fast supnorm estimate for local search."""
        angles = np.linspace(0.0, 2.0 * np.pi, n_pts_fast, endpoint=False)
        z = np.exp(1j * angles)
        k = np.arange(N)
        vals = np.sum(c[None, :] * np.power(z[:, None], k[None, :]), axis=1)
        return np.max(np.abs(vals))
    
    fast_best = fast_supnorm(best_coeffs) / np.sqrt(N)
    
    # Hill climbing: try flipping each coefficient
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            # Flip coefficient i
            old_val = coeffs[i]
            coeffs[i] = -old_val
            
            new_fast = fast_supnorm(coeffs) / np.sqrt(N)
            if new_fast < fast_best - 1e-10:
                fast_best = new_fast
                best_coeffs = coeffs.copy()
                improved = True
            else:
                coeffs[i] = old_val
        
        if not improved:
            # Try flipping pairs
            for i in range(N):
                for j in range(i + 1, N):
                    old_i, old_j = coeffs[i], coeffs[j]
                    coeffs[i] = -old_i
                    coeffs[j] = -old_j
                    
                    new_fast = fast_supnorm(coeffs) / np.sqrt(N)
                    if new_fast < fast_best - 1e-10:
                        fast_best = new_best
                        best_coeffs = coeffs.copy()
                        improved = True
                    else:
                        coeffs[i] = old_i
                        coeffs[j] = old_j
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
    
    # Verify with full precision
    best_score_full = _score(best_coeffs)
    
    if best_score_full <= target_score * (1.0 + 1e-6):
        return [int(c) for c in best_coeffs]
    
    # If we haven't found the optimum, try random restarts with local search
    # but we're likely out of time or stuck. Return best found.
    return [int(c) for c in best_coeffs]
