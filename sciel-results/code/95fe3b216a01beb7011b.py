import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

# Precompute the grid points
# Equal intervals of [-1/4, 1/4]
# 50 points on [-1/4, 1/4]
_x_grid = np.linspace(-0.25, 0.25, N)

def _autoconvolution(h):
    """Compute autoconvolution of the height vector.
    
    The autoconvolution c[k] = sum_i h[i] * h[i+k] (discrete convolution).
    For a sequence of length N, the full autoconvolution has length 2N-1.
    We want the supremum norm of this autoconvolution.
    """
    # Use np.convolve for full convolution
    # h is a 1D array of length N
    conv = np.convolve(h, h, mode='full')
    return conv

def _score(h):
    """Compute the ratio of squared L2 norm to L1 norm times supremum norm of autoconvolution.
    
    We want to MAXIMIZE: ||h||_2^2 / (||h||_1 * sup|autoconv(h)|)
    
    Note: autoconv here is the full discrete autoconvolution.
    """
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        # Invalid: negative heights
        return -np.inf
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -np.inf
    
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(np.abs(conv))
    
    if sup_conv < 1e-12:
        return -np.inf
    
    return l2_sq / (l1 * sup_conv)

def _construct_from_gaussians(params):
    """Construct height vector from Gaussian basis functions.
    
    params: array of shape (K, 3) where each row is (center, width, amplitude)
    for K Gaussian bumps.
    
    Returns: array of length N with nonnegative heights.
    """
    K = params.shape[0]
    h = np.zeros(N)
    for k in range(K):
        center, width, amplitude = params[k]
        if width < 1e-8:
            width = 1e-8
        if amplitude < 0:
            amplitude = 0
        # Evaluate Gaussian on the grid
        g = amplitude * np.exp(-0.5 * ((_x_grid - center) / width)**2)
        h += g
    return np.maximum(h, 0.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Strategy: Use differential evolution to optimize Gaussian parameters
    # Then polish with SLSQP on the direct 50-dim problem
    
    # First, try a few different numbers of Gaussian bumps
    for K in [1, 2, 3]:
        if time.time() > deadline - 20:
            break
        
        # Define the objective function to MINIMIZE (negative of score)
        def obj(params):
            if time.time() > deadline - 10:
                return 0.0  # Stop early
            h = _construct_from_gaussians(params)
            s = _score(h)
            if np.isinf(s) or np.isnan(s):
                return 0.0
            return -s
        
        # Bounds for parameters: (center, width, amplitude)
        # center in [-0.25, 0.25], width in [0.01, 0.5], amplitude in [0, 2]
        bounds = [(-0.25, 0.25), (0.01, 0.5), (0.0, 2.0)] * K
        
        try:
            # Use differential evolution for global search
            result = differential_evolution(
                obj, bounds, 
                maxiter=50, popsize=15,
                seed=42, tol=1e-8,
                polish=False,  # We'll polish separately
                init='latinhypercube',
                workers=1
            )
            
            if result.success or result.fun < 0:
                h_candidate = _construct_from_gaussians(result.x.reshape(K, 3))
                s_candidate = _score(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Now try to polish the best solution using SLSQP on the direct 50-dim problem
    if time.time() < deadline - 15:
        try:
            # Objective for SLSQP: minimize negative score
            def obj_direct(h):
                if time.time() > deadline - 10:
                    return 0.0
                h = np.maximum(h, 0.0)
                s = _score(h)
                if np.isinf(s) or np.isnan(s):
                    return 0.0
                return -s
            
            # Constraints: h >= 0 (handled by bounds)
            bounds_direct = [(0.0, None)] * N
            
            result = minimize(
                obj_direct, best_h,
                method='SLSQP',
                bounds=bounds_direct,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.fun < 0:
                h_candidate = np.maximum(result.x, 0.0)
                s_candidate = _score(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Additional polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            def obj_direct2(h):
                if time.time() > deadline - 3:
                    return 0.0
                h = np.maximum(h, 0.0)
                s = _score(h)
                if np.isinf(s) or np.isnan(s):
                    return 0.0
                return -s
            
            bounds_direct = [(0.0, None)] * N
            
            result = minimize(
                obj_direct2, best_h,
                method='SLSQP',
                bounds=bounds_direct,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            
            if result.fun < 0:
                h_candidate = np.maximum(result.x, 0.0)
                s_candidate = _score(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
