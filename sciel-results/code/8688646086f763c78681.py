import time
import numpy as np

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv(heights):
    """Compute autoconvolution using FFT."""
    n = len(heights)
    M = 1
    while M < 2 * n - 1:
        M *= 2
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    return conv

def _score_and_conv(heights):
    """Compute score and autoconvolution."""
    conv = _autoconv(heights)
    max_conv = np.max(conv)
    total = np.sum(heights)
    w = 0.5 / N
    if total <= 0:
        return np.inf, conv
    score = max_conv * N / (total * total * w)
    return score, conv

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0
    
    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Phase 1: Try structured candidates
    x = np.linspace(0, 1, N)
    
    structured = []
    # Cosine tapers
    for tf in [0.1, 0.2, 0.3, 0.4, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        structured.append(ht)
    
    # Gaussian bumps
    for width in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        hg /= np.max(hg)
        structured.append(hg)
    
    # Parabolic
    for a in [0.5, 1.0, 2.0, 3.0, 4.0]:
        hp = 1.0 - a * (2 * x - 1) ** 2
        hp = np.maximum(hp, 0.0)
        if np.max(hp) > 0:
            structured.append(hp)
    
    # Two-block structure (inspiration from other solutions)
    for left_frac in [0.2, 0.3, 0.4, 0.5]:
        for gap_frac in [0.0, 0.05, 0.1, 0.15, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                structured.append(hb)
    
    # Evaluate structured
    for hc in structured:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()
    
    # Phase 2: Simulated annealing with Gaussian perturbations
    # Key insight: perturb in a low-dimensional subspace to avoid flat landscape
    # Use random Fourier features as perturbation basis
    
    # Generate random basis functions (Fourier modes)
    num_basis = 20
    basis_indices = rng.choice(N, size=num_basis, replace=False)
    basis = np.zeros((num_basis, N), dtype=np.float64)
    for i, idx in enumerate(basis_indices):
        basis[i] = np.cos(2 * np.pi * idx * np.arange(N) / N)
        # Also add sine
    # Add sine components
    basis_sine = np.zeros((num_basis, N), dtype=np.float64)
    for i, idx in enumerate(basis_indices):
        basis_sine[i] = np.sin(2 * np.pi * idx * np.arange(N) / N)
    
    # Combined basis
    full_basis = np.vstack([basis, basis_sine])  # shape (2*num_basis, N)
    num_params = full_basis.shape[0]
    
    # Initial parameters
    params = np.zeros(num_params, dtype=np.float64)
    
    def params_to_heights(p):
        h_out = best_h + full_basis.T @ p
        return np.maximum(h_out, 0.0)
    
    # Simulated annealing
    current_h = best_h.copy()
    current_score = best_score
    params = np.zeros(num_params, dtype=np.float64)
    
    T_start = 0.05
    T_end = 0.001
    n_sa_iters = 3000
    
    # Precompute for speed
    M = 1
    while M < 2 * N - 1:
        M *= 2
    w = 0.5 / N
    
    for it in range(n_sa_iters):
        if deadline_reached():
            break
        
        # Annealing schedule
        progress = it / n_sa_iters
        T = T_start * (T_end / T_start) ** progress
        
        # Generate random perturbation in parameter space
        delta_p = rng.normal(0, 0.01, size=num_params)
        new_params = params + delta_p
        new_h = params_to_heights(new_params)
        
        if np.sum(new_h) <= 0:
            continue
        
        new_score = compute_score(new_h)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score < current_score:
            accept = True
        else:
            delta = new_score - current_score
            if delta < 0.1:  # Only consider small deteriorations
                accept = rng.random() < np.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            current_h = new_h
            current_score = new_score
            params = new_params
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # Reject, keep current
            pass
    
    # Phase 3: Local optimization using scipy
    from scipy.optimize import minimize
    
    def objective(p):
        h_out = best_h + full_basis.T @ p
        h_out = np.maximum(h_out, 0.0)
        if np.sum(h_out) <= 0:
            return 1e10
        return compute_score(h_out)
    
    # Start from best found
    x0 = np.zeros(num_params, dtype=np.float64)
    
    # Try L-BFGS-B
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if result.fun < best_score:
        best_score = result.fun
        best_h = params_to_heights(result.x)
    
    # Phase 4: Multi-start with random initializations in parameter space
    n_restarts = 30
    for r in range(n_restarts):
        if deadline_reached():
            break
        
        # Random initialization
        x0_r = rng.normal(0, 0.05, size=num_params)
        
        def objective_r(p):
            h_out = best_h + full_basis.T @ p
            h_out = np.maximum(h_out, 0.0)
            if np.sum(h_out) <= 0:
                return 1e10
            return compute_score(h_out)
        
        try:
            result_r = minimize(
                objective_r,
                x0_r,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result_r.fun < best_score:
                best_score = result_r.fun
                best_h = params_to_heights(result_r.x)
        except:
            continue
    
    # Phase 5: Fine-tuning with smaller basis (higher frequency modes)
    if time_left() > 15:
        # Add higher frequency basis for fine tuning
        extra_indices = rng.choice(N, size=10, replace=False)
        extra_basis = np.zeros((20, N), dtype=np.float64)
        for i, idx in enumerate(extra_indices):
            extra_basis[2*i] = np.cos(2 * np.pi * idx * np.arange(N) / N)
            extra_basis[2*i+1] = np.sin(2 * np.pi * idx * np.arange(N) / N)
        
        def objective_fine(p):
            h_out = best_h + extra_basis.T @ p
            h_out = np.maximum(h_out, 0.0)
            if np.sum(h_out) <= 0:
                return 1e10
            return compute_score(h_out)
        
        x0_fine = np.zeros(20, dtype=np.float64)
        try:
            result_fine = minimize(
                objective_fine,
                x0_fine,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result_fine.fun < best_score:
                best_score = result_fine.fun
                best_h = np.maximum(best_h + extra_basis.T @ result_fine.x, 0.0)
        except:
            pass
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
