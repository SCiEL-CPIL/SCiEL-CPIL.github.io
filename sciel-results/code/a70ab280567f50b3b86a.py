import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial guess: uniform distribution
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    # Objective function for L-BFGS-B: maximize L2² subject to peak constraint
    # We'll use a penalty approach: maximize L2² - λ * max(0, peak - threshold)
    # But first, let's use a direct approach with constraints
    
    def objective_and_grad(h):
        l2_sq = np.sum(h**2)
        # Gradient of L2² is 2*h
        grad = 2 * h
        
        # Compute autoconvolution peak
        h_conv = np.convolve(h, h, mode='full')
        peak = np.max(h_conv)
        
        return -l2_sq, -grad  # Negate for minimization
    
    # Use L-BFGS-B with bounds [0, inf)
    bounds = [(0.0, None)] * N
    
    # Multiple restarts with different initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Sparse (Sidon-like)
    for _ in range(10):
        h_sparse = np.zeros(N)
        num_peaks = np.random.randint(8, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_sparse[indices] = 1.0
        initializations.append(h_sparse)
    
    # 3. Gaussian-like
    for center in np.linspace(N//4, 3*N//4, 5):
        for width in [3, 5, 8, 12]:
            h_gauss = np.exp(-0.5 * ((np.arange(N) - center) / width)**2)
            initializations.append(h_gauss)
    
    # 4. Triangle-like
    for center in np.linspace(N//4, 3*N//4, 5):
        for width in [5, 10, 15, 20]:
            h_tri = np.maximum(0, 1 - np.abs(np.arange(N) - center) / width)
            if np.sum(h_tri) > 0:
                initializations.append(h_tri)
    
    # Optimize from each initialization
    for init in initializations:
        if time.time() > deadline - 10:
            break
        
        init = np.maximum(init, 0)
        if np.sum(init) == 0:
            continue
        init = init / np.sum(init)
        
        try:
            result = minimize(
                objective_and_grad,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success:
                h_candidate = np.maximum(result.x, 0)
                if np.sum(h_candidate) > 0:
                    h_candidate = h_candidate / np.sum(h_candidate)
                    score = _score_raw(h_candidate)
                    if score > best_score:
                        best_score = score
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Additional refinement using trust-region approach
    if time.time() < deadline - 15:
        # Use the best solution as starting point for trust-region
        h_start = best_h.copy()
        
        def trust_region_objective(h):
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            peak = np.max(h_conv)
            # We want to maximize L2² / (L1 * peak)
            # With L1 normalized, this is L2² / peak
            return -(l2_sq / max(peak, 1e-10))
        
        try:
            result = minimize(
                trust_region_objective,
                h_start,
                method='TRF',  # Trust Region Reflective
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            
            if result.success:
                h_candidate = np.maximum(result.x, 0)
                if np.sum(h_candidate) > 0:
                    h_candidate = h_candidate / np.sum(h_candidate)
                    score = _score_raw(h_candidate)
                    if score > best_score:
                        best_score = score
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final polish with SLSQP for very fine tuning
    if time.time() < deadline - 5:
        def final_objective(h):
            l1 = np.sum(h)
            if l1 == 0:
                return 0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            peak = np.max(h_conv)
            if peak == 0:
                return 0
            return -(l2_sq / (l1 * peak))
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        
        try:
            result = minimize(
                final_objective,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            
            if result.success:
                h_candidate = np.maximum(result.x, 0)
                if np.sum(h_candidate) > 0:
                    h_candidate = h_candidate / np.sum(h_candidate)
                    score = _score_raw(h_candidate)
                    if score > best_score:
                        best_score = score
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity and normalize
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) > 1e-10:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)
