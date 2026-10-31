"""Block-structured DE + log-sum-exp SLSQP for Erdos minimum overlap."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 100

def _max_overlap_fft(h):
    """Max over all circular shifts of sum(h[i]*(1-h[(i+s)%N]))."""
    F = np.fft.fft(h)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    overlaps = N / 2.0 - autocorr
    return float(np.max(overlaps))

def _build_h_from_blocks(blocks):
    """
    Build h from block structure.
    blocks: array of shape (n_blocks, 2) where column 0 is start index, column 1 is height.
    Each block covers [start, start+width) with constant height.
    Widths are equal: N // n_blocks (with remainder handled).
    """
    n_blocks = len(blocks)
    base_width = N // n_blocks
    remainder = N - base_width * n_blocks
    h = np.zeros(N)
    pos = 0
    for i in range(n_blocks):
        w = base_width + (1 if i < remainder else 0)
        h[pos:pos+w] = blocks[i, 1]
        pos += w
    return h

def _normalize_h(h):
    """Ensure sum(h) = N/2 and h in [0,1]."""
    s = h.sum()
    if s < 1e-12:
        return np.full(N, 0.5)
    h = h * (N / 2.0) / s
    h = np.clip(h, 0.0, 1.0)
    # Exact sum fix
    diff = N / 2.0 - h.sum()
    if abs(diff) > 1e-12:
        # Distribute adjustment to elements closest to 0.5
        order = np.argsort(np.abs(h - 0.5))
        remaining = diff
        for idx in order:
            if remaining > 0:
                adj = min(remaining, 1.0 - h[idx])
                h[idx] += adj
                remaining -= adj
            elif remaining < 0:
                adj = max(remaining, -h[idx])
                h[idx] += adj
                remaining -= adj
            if abs(remaining) < 1e-12:
                break
    return h

def _objective_de(params, n_blocks):
    """
    Objective for differential evolution.
    params: array of shape (n_blocks,) representing heights of each block.
    We fix the block boundaries to be equal-width (or near-equal-width).
    The sum constraint is handled by normalization.
    """
    h = np.zeros(N)
    base_width = N // n_blocks
    remainder = N - base_width * n_blocks
    pos = 0
    for i in range(n_blocks):
        w = base_width + (1 if i < remainder else 0)
        h[pos:pos+w] = params[i]
        pos += w
    h = _normalize_h(h)
    return _max_overlap_fft(h)

def _log_sum_exp_overlap(h, beta=50.0):
    """Smooth surrogate: log-sum-exp of overlaps / beta approximates max."""
    F = np.fft.fft(h)
    autocorr = np.fft.ifft(F * np.conj(F)).real
    overlaps = N / 2.0 - autocorr
    # log-sum-exp trick
    m = np.max(overlaps)
    val = m + np.log(np.sum(np.exp(overlaps - m))) / beta
    return val

def _objective_slsqp(params, n_blocks, beta=50.0):
    """Smooth objective for SLSQP."""
    h = np.zeros(N)
    base_width = N // n_blocks
    remainder = N - base_width * n_blocks
    pos = 0
    for i in range(n_blocks):
        w = base_width + (1 if i < remainder else 0)
        h[pos:pos+w] = params[i]
        pos += w
    h = _normalize_h(h)
    return _log_sum_exp_overlap(h, beta)

def _construct_from_blocks(params, n_blocks):
    """Final construction from block heights."""
    h = np.zeros(N)
    base_width = N // n_blocks
    remainder = N - base_width * n_blocks
    pos = 0
    for i in range(n_blocks):
        w = base_width + (1 if i < remainder else 0)
        h[pos:pos+w] = params[i]
        pos += w
    h = _normalize_h(h)
    return h

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_fft(best_h)  # 0.25
    
    # Try multiple block structures
    block_configs = [4, 5, 6, 8, 10, 12]
    
    for n_blocks in block_configs:
        if time.time() > deadline:
            break
        
        # Phase 1: Differential Evolution (global search)
        bounds = [(0.0, 1.0)] * n_blocks
        
        try:
            de_result = differential_evolution(
                lambda p: _objective_de(p, n_blocks),
                bounds,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        # Evaluate DE result
        h_candidate = _construct_from_blocks(de_result.x, n_blocks)
        sc = _max_overlap_fft(h_candidate)
        
        if sc < best_score:
            best_score = sc
            best_h = h_candidate.copy()
        
        # Phase 2: SLSQP refinement on smooth surrogate
        if time.time() > deadline:
            break
        
        try:
            slsqp_result = minimize(
                lambda p: _objective_slsqp(p, n_blocks, beta=80.0),
                de_result.x,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            h_candidate2 = _construct_from_blocks(slsqp_result.x, n_blocks)
            sc2 = _max_overlap_fft(h_candidate2)
            
            if sc2 < best_score:
                best_score = sc2
                best_h = h_candidate2.copy()
        except Exception:
            pass
        
        # Phase 3: Additional SLSQP with higher beta (sharper approximation)
        if time.time() > deadline:
            break
        
        try:
            slsqp_result2 = minimize(
                lambda p: _objective_slsqp(p, n_blocks, beta=200.0),
                slsqp_result.x if 'slsqp_result' in dir() else de_result.x,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            h_candidate3 = _construct_from_blocks(slsqp_result2.x, n_blocks)
            sc3 = _max_overlap_fft(h_candidate3)
            
            if sc3 < best_score:
                best_score = sc3
                best_h = h_candidate3.copy()
        except Exception:
            pass
    
    # Phase 4: Local search around best solution - try small perturbations
    if time.time() < deadline:
        # Try to improve by adjusting individual elements
        for _ in range(5):
            if time.time() > deadline:
                break
            
            # Random perturbation of a few elements
            h_pert = best_h.copy()
            n_perturb = np.random.randint(2, 6)
            indices = np.random.choice(N, n_perturb, replace=False)
            for idx in indices:
                h_pert[idx] = np.clip(h_pert[idx] + np.random.uniform(-0.05, 0.05), 0, 1)
            h_pert = _normalize_h(h_pert)
            sc_pert = _max_overlap_fft(h_pert)
            if sc_pert < best_score:
                best_score = sc_pert
                best_h = h_pert.copy()
    
    # Final validation
    if abs(best_h.sum() - N / 2.0) > 1e-8:
        best_h = _normalize_h(best_h)
    
    return best_h
