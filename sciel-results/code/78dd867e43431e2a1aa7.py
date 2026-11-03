"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Spectral parameterization with global grid search over low-frequency DFT modes.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    # Precompute basis vectors for DFT modes
    i = np.arange(N)
    # Modes to include: 0 (DC), 1..M (cos/sine)
    # DC is fixed by sum constraint, so we parameterize modes 1..M
    M = 8
    modes = []
    for k in range(1, M + 1):
        modes.append(np.cos(2.0 * np.pi * k * i / N))
        modes.append(np.sin(2.0 * np.pi * k * i / N))
        
    basis = np.column_stack(modes)
    # h = 0.5 + basis @ coeffs
    # Sum constraint: sum(h) = N/2 => sum(0.5) + sum(basis @ coeffs) = N/2
    # sum(0.5) = N/2, so sum(basis @ coeffs) = 0
    # The basis vectors have mean 0, so this is automatically satisfied for any coeffs.
    # However, floating point errors might creep in, so _fix_sum handles it.
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # Phase 1: Coarse Grid Search
    # We search over a grid of coefficients.
    # To keep the grid manageable, we limit the number of dimensions.
    # With M=8, we have 16 coeffs. A grid of 3 points per dim is 3^16 = 43M points. Too many.
    # Let's use a random sampling strategy instead, or a smaller grid for the first few modes.
    
    # Let's try a grid for the first 4 modes (8 coeffs) with 5 points each.
    # 5^8 = 390625 points. This is feasible.
    # Then refine with random sampling for all modes.
    
    # Grid search for first K modes
    K_grid = 4
    num_points = 5
    range_val = 0.25  # Coefficients in [-0.25, 0.25]
    
    grid_vals = np.linspace(-range_val, range_val, num_points)
    grid_indices = np.ndindex(*([num_points] * (2 * K_grid)))
    
    count = 0
    for idx in grid_indices:
        if time.time() > _DEADLINE - 20:
            break
        coeffs = np.zeros(2 * M)
        for j, v in enumerate(idx):
            coeffs[j] = grid_vals[v]
        
        h = 0.5 + basis @ coeffs
        _update_best(h)
        count += 1
        
    # Phase 2: Random Sampling / Global Optimization
    # Sample random coefficient vectors and evaluate.
    # Use a mix of uniform and gaussian distributions.
    
    num_samples = 50000
    for _ in range(num_samples):
        if time.time() > _DEADLINE - 10:
            break
            
        # Generate random coefficients
        # Decay variance for higher modes
        variances = np.array([0.1 / (k + 1) for k in range(2 * M)])
        coeffs = np.random.normal(0, np.sqrt(variances))
        
        h = 0.5 + basis @ coeffs
        _update_best(h)
        
    # Phase 3: Local Refinement
    # If we have a good candidate, do a local search around it.
    # Simple coordinate descent or small random perturbations.
    
    if best_score < 1.25:
        # Local search
        h_curr = best_h.copy()
        # Reconstruct coeffs from h_curr?
        # Or just perturb h directly?
        # Perturbing h directly is easier but might break the spectral structure.
        # Let's stick to spectral coeffs.
        # We need to map h_curr back to coeffs.
        # h = 0.5 + basis @ coeffs
        # (h - 0.5) = basis @ coeffs
        # coeffs = (basis.T @ basis)^-1 basis.T @ (h - 0.5)
        
        try:
            B = basis
            # Solve least squares
            coeffs_curr, _, _, _ = np.linalg.lstsq(B, h_curr - 0.5, rcond=None)
        except:
            coeffs_curr = np.zeros(2 * M)
            
        for _ in range(5000):
            if time.time() > _DEADLINE - 2:
                break
            
            # Gaussian perturbation
            step_size = 0.05
            coeffs_new = coeffs_curr + np.random.normal(0, step_size, 2 * M)
            
            # Decay step size for higher modes
            # This is a simple heuristic
            
            h_new = 0.5 + basis @ coeffs_new
            s_new = _score(_fix_sum(h_new))
            
            if s_new < best_score:
                best_score = s_new
                best_h = _fix_sum(h_new)
                coeffs_curr = coeffs_new.copy()
                # Reduce step size on success
                step_size *= 0.9
            else:
                # Increase step size slightly on failure? Or keep it?
                # Let's keep it simple.
                pass
                
    return best_h
