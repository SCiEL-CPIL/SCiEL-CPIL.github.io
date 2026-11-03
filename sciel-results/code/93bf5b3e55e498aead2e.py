"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_optimization():
    """
    Use spectral decomposition of the circulant convolution matrix to optimize
    the ratio of the largest eigenvalue to the squared norm of projection onto all-ones.
    
    The autoconvolution is related to the circulant matrix C where C = circulant(h).
    The eigenvalues of C are H[k] = DFT(h)[k].
    We want to minimize max_k |H[k]|^2 / (sum h)^2.
    
    This is equivalent to finding h that minimizes the condition number of the
    spectral distribution, subject to sum(h) = 1.
    """
    t_start = time.time()
    deadline = t_start + 100.0
    
    # Start with a chirp-like initial guess
    i = np.arange(N)
    h0 = np.cos(2.0 * np.pi * i**2 / N)
    h0 = h0 / np.sum(h0)  # Normalize to sum = 1
    
    def objective(x):
        """Objective: max|autoconv| / (sum)^2, with x being the heights."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-9:
            return 1e10
        ac = _autoconv_max_abs(h)
        return ac / (s * s)
    
    # Use scipy optimization with constraint sum(h) = 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    best_h = h0.copy()
    best_score = objective(h0)
    
    # Try multiple initializations
    initializations = [
        h0.copy(),
        np.ones(N) / N,
        np.cos(np.pi * i**2 / N),
        np.cos(np.pi * i * (i + 1) / N),
    ]
    
    for h_init in initializations:
        if time.time() > deadline - 20:
            break
        
        h_init = h_init / np.sum(h_init)
        
        try:
            result = minimize(
                objective, 
                h_init, 
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if result.success:
                h_opt = result.x
                h_opt = h_opt / np.sum(h_opt)
                score_opt = objective(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception:
            continue
    
    # Local search refinement: sign flips
    if time.time() < deadline - 10:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(5):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 5:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = objective(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h, best_score

def _flat_spectrum_search():
    """
    Search for sequences with flat magnitude spectrum using spectral methods.
    """
    t_start = time.time()
    deadline = t_start + 80.0
    
    i = np.arange(N)
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Try different chirp parameters
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        if time.time() > deadline - 10:
            break
        h = np.cos(2.0 * np.pi * alpha * i**2 / N)
        h = h / np.sum(h)
        score = _score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try Zadoff-Chu like
    for alpha in [0.5, 1.0, 1.5]:
        if time.time() > deadline - 10:
            break
        h = np.cos(np.pi * alpha * i * (i + 1) / N)
        h = h / np.sum(h)
        score = _score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Local search on best
    if time.time() < deadline - 5:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(3):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 3:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Strategy 1: Spectral optimization
    if time.time() < deadline - 40:
        h_spec, score_spec = _spectral_optimization()
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Strategy 2: Flat spectrum search
    if time.time() < deadline - 20:
        h_flat, score_flat = _flat_spectrum_search()
        if score_flat < best_score:
            best_score = score_flat
            best_h = h_flat.copy()
    
    # Final local search
    if time.time() < deadline - 5:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(3):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 3:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h

# EVOLVE-BLOCK-END
