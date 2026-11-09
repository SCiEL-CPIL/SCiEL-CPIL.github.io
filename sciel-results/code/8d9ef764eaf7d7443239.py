"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def construct():
    """
    Maximize ratio = ||f||_2^2 / (||f||_1 * sup|f*f|)
    
    Key structural insight: The ratio is scale-invariant. We can normalize
    sum(a_i) = 1 (or any constant) and optimize the shape.
    
    For step functions on [-1/4, 1/4] with N=50 equal intervals:
    - dx = (1/2)/50 = 1/100
    - ||f||_2^2 = sum(a_i^2) * dx
    - ||f||_1 = sum(a_i) * dx
    - autoconv computed via np.correlate(a, a, 'full') * dx
    
    The discrete autoconvolution has peak at lag 0 = sum(a_i^2) * dx,
    but for non-uniform shapes, off-diagonal terms can matter.
    
    Strategy:
    1. Normalize to sum(a) = N (so average height = 1)
    2. Use scipy SLSQP with bounds [0, inf) to optimize shape
    3. Try multiple initializations: uniform, edge-concentrated, sparse blocks
    4. Use the insight from score 1.13 that non-uniform shapes can beat uniform
    """
    
    dx = 0.5 / N  # total width 1/2, N=50 intervals
    
    def compute_ratio(heights):
        a = np.array(heights, dtype=np.float64)
        if len(a) == 0 or np.sum(a) < 1e-15:
            return 0.0
        norm2_sq = np.sum(a**2) * dx
        norm1 = np.sum(a) * dx
        
        # Compute autoconvolution using np.correlate
        autoconv = np.correlate(a, a, mode='full') * dx
        sup_autoconv = np.max(autoconv)
        
        if sup_autoconv < 1e-15 or norm1 < 1e-15:
            return 0.0
        
        ratio = norm2_sq / (norm1 * sup_autoconv)
        return ratio
    
    best_ratio = 0.0
    best_heights = [1.0] * N
    
    # Baseline: uniform
    r = compute_ratio([1.0] * N)
    best_ratio = r
    best_heights = [1.0] * N
    
    # Try various structural shapes as initializations
    idx = np.arange(N)
    center = (N - 1) / 2.0
    
    # Edge-concentrated (tall at edges, zero in middle)
    edge_conc = np.zeros(N)
    edge_conc[:N//5] = 1.0
    edge_conc[-(N//5):] = 1.0
    edge_conc = edge_conc / np.sum(edge_conc) * N  # normalize sum to N
    r = compute_ratio(list(edge_conc))
    if r > best_ratio:
        best_ratio = r
        best_heights = list(edge_conc)
    
    # Sparse single block
    for block_size in [1, 2, 3, 5, 10]:
        h = np.zeros(N)
        h[:block_size] = 1.0
        h = h / np.sum(h) * N
        r = compute_ratio(list(h))
        if r > best_ratio:
            best_ratio = r
            best_heights = list(h)
    
    # Two separated blocks
    for k1 in [2, 5, 10]:
        for gap in [5, 10, 20]:
            if k1 + gap + k1 < N:
                h = np.zeros(N)
                h[:k1] = 1.0
                h[k1+gap:k1+gap+k1] = 1.0
                h = h / np.sum(h) * N
                r = compute_ratio(list(h))
                if r > best_ratio:
                    best_ratio = r
                    best_heights = list(h)
    
    # Try scipy optimization with multiple initializations
    try:
        from scipy.optimize import minimize
        
        def neg_ratio(x):
            if time.time() - START_TIME > TIME_LIMIT:
                return 0.0
            h = np.maximum(x, 0.0)  # enforce nonnegativity
            # Normalize to sum = N for stability
            s = np.sum(h)
            if s < 1e-10:
                return 0.0
            h_norm = h / s * N
            return -compute_ratio(list(h_norm))
        
        # Try multiple initializations
        initializations = [
            np.ones(N),
            edge_conc.copy(),
            np.zeros(N),  # will be replaced
        ]
        
        # Add sparse block initializations
        for block_size in [1, 3, 5, 10]:
            h_init = np.zeros(N)
            h_init[:block_size] = 1.0
            h_init = h_init / np.sum(h_init) * N
            initializations.append(h_init)
        
        # Add alternating pattern
        h_alt = np.zeros(N)
        h_alt[::2] = 1.0
        h_alt = h_alt / np.sum(h_alt) * N
        initializations.append(h_alt)
        
        for x0 in initializations:
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Ensure valid starting point
            x0 = np.maximum(x0, 0.0)
            if np.sum(x0) < 1e-10:
                x0 = np.ones(N)
            
            try:
                result = minimize(neg_ratio, x0, method='SLSQP',
                                bounds=[(0, None)] * N,
                                options={'maxiter': 200, 'ftol': 1e-8})
                
                if result.fun < 0:
                    r_opt = -result.fun
                    if r_opt > best_ratio:
                        best_ratio = r_opt
                        best_heights = [max(0.0, x) for x in result.x]
                        
            except Exception:
                pass
        
        # Also try Nelder-Mead for robustness
        if time.time() - START_TIME < TIME_LIMIT - 30:
            x0 = np.ones(N)
            try:
                result = minimize(neg_ratio, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
                
                if result.fun < 0:
                    r_opt = -result.fun
                    if r_opt > best_ratio:
                        best_ratio = r_opt
                        best_heights = [max(0.0, x) for x in result.x]
                        
            except Exception:
                pass
                
    except Exception:
        pass
    
    # Ensure nonnegative and valid
    best_heights = [max(0.0, h) for h in best_heights]
    if np.sum(best_heights) < 1e-10:
        best_heights = [1.0] * N
    
    return best_heights
