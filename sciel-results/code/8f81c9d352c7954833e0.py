"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

Structural change: Differential evolution global search + vectorized FFT scoring + adaptive coordinate polish.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve

N = 50
DELTA = 1.0 / N  # = 0.02

def _score_raw(h):
    """Raw score: sum h^2 / (delta * sum h * max conv)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Use FFT for fast convolution
    h_conv = fftconvolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / (DELTA * l1 * max_conv)

def _negative_score(h):
    """Negative score for minimization."""
    return -_score_raw(h)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = 0.0
    
    def update_best(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=np.float64)
        # Ensure nonnegativity
        h = np.maximum(h, 0.0)
        if np.sum(h) <= 0:
            return
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    def coordinate_descent(h, max_iters=200, tol=1e-8):
        """Adaptive coordinate descent with line search."""
        h = np.asarray(h, dtype=np.float64).copy()
        n_active = np.sum(h > 1e-10)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try each coordinate
            for i in range(N):
                if h[i] < 1e-12:
                    # Try adding this coordinate
                    for step in [0.01, 0.05, 0.1, 0.2, 0.5]:
                        h_try = h.copy()
                        h_try[i] = step
                        s = _score_raw(h_try)
                        if s > best_score + 1e-10:
                            h = h_try.copy()
                            improved = True
                            break
                else:
                    # Try modifying this coordinate
                    orig = h[i]
                    for factor in [0.5, 0.7, 1.3, 1.5, 2.0, 0.1, 0.01]:
                        h_try = h.copy()
                        h_try[i] = orig * factor
                        s = _score_raw(h_try)
                        if s > best_score + 1e-10:
                            h = h_try.copy()
                            improved = True
                            break
                    # Also try small absolute perturbations
                    for delta in [-0.05, -0.02, 0.02, 0.05]:
                        h_try = h.copy()
                        h_try[i] = max(0.0, h[i] + delta)
                        s = _score_raw(h_try)
                        if s > best_score + 1e-10:
                            h = h_try.copy()
                            improved = True
                            break
            
            if not improved:
                break
        
        return h
    
    def nelder_mead_polish(h, maxiter=100):
        """Nelder-Mead refinement on the full vector."""
        h = np.asarray(h, dtype=np.float64).copy()
        
        def obj(x):
            x = np.maximum(x, 0.0)
            if np.sum(x) < 1e-10:
                return 1e10
            return -_score_raw(x)
        
        try:
            res = minimize(obj, h, method='Nelder-Mead', 
                          options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_score + 1e-10:
                return np.maximum(res.x, 0.0)
        except:
            pass
        return h
    
    # Phase 1: Differential evolution with multiple restarts
    # Bounds: each h_i in [0, 1] (scale-invariant, we can normalize later)
    bounds = [(0.0, 1.0)] * N
    
    # Strategy 1: Standard DE with moderate population
    np.random.seed(42)
    
    de_configs = [
        {'maxiter': 80, 'popsize': 30, 'mutation': (0.4, 0.9), 'recombination': 0.7, 'seed': 42},
        {'maxiter': 60, 'popsize': 40, 'mutation': (0.6, 0.8), 'recombination': 0.5, 'seed': 123},
        {'maxiter': 50, 'popsize': 25, 'mutation': (0.3, 0.95), 'recombination': 0.8, 'seed': 456},
        {'maxiter': 40, 'popsize': 50, 'mutation': (0.5, 0.7), 'recombination': 0.6, 'seed': 789},
    ]
    
    for cfg_idx, cfg in enumerate(de_configs):
        if time.time() > deadline - 20:
            break
        
        try:
            result = differential_evolution(
                _negative_score, bounds,
                maxiter=cfg['maxiter'],
                popsize=cfg['popsize'],
                mutation=cfg['mutation'],
                recombination=cfg['recombination'],
                seed=cfg['seed'],
                tol=1e-8,
                polish=False,  # We'll do our own polish
                init='latinhypercube',
                update_interval=10
            )
            
            if result.x is not None:
                h_de = np.maximum(result.x, 0.0)
                update_best(h_de)
                
                # Coordinate descent polish
                h_cd = coordinate_descent(h_de, max_iters=150)
                update_best(h_cd)
                
                # Nelder-Mead polish
                h_nm = nelder_mead_polish(h_cd, maxiter=80)
                update_best(h_nm)
                
        except Exception as e:
            pass
    
    # Phase 2: Targeted restarts around the best solution found
    if best_h is not None and time.time() < deadline - 15:
        # Create perturbed versions of the best solution
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            noise_scale = [0.1, 0.2, 0.3, 0.5, 0.8][restart]
            h_perturbed = best_h + np.random.normal(0, noise_scale, size=N)
            h_perturbed = np.maximum(h_perturbed, 0.0)
            
            # Short DE around this region
            bounds_local = []
            for i in range(N):
                center = h_perturbed[i]
                lo = max(0.0, center - noise_scale)
                hi = center + noise_scale
                bounds_local.append((lo, hi))
            
            try:
                result = differential_evolution(
                    _negative_score, bounds_local,
                    maxiter=30,
                    popsize=20,
                    mutation=(0.5, 0.8),
                    recombination=0.6,
                    seed=1000 + restart,
                    tol=1e-8,
                    polish=False,
                    init='latinhypercube'
                )
                
                if result.x is not None:
                    h_local = np.maximum(result.x, 0.0)
                    update_best(h_local)
                    
                    # Polish
                    h_cd = coordinate_descent(h_local, max_iters=100)
                    update_best(h_cd)
                    
                    h_nm = nelder_mead_polish(h_cd, maxiter=50)
                    update_best(h_nm)
                    
            except:
                pass
    
    # Phase 3: Final intensive polish on the global best
    if best_h is not None and time.time() < deadline - 5:
        # Multiple coordinate descent passes with decreasing step sizes
        h_final = best_h.copy()
        for pass_idx in range(3):
            h_final = coordinate_descent(h_final, max_iters=100)
            update_best(h_final)
        
        # Final Nelder-Mead
        h_final = nelder_mead_polish(h_final, maxiter=200)
        update_best(h_final)
    
    # Return best found
    if best_h is not None:
        return list(best_h)
    return list(np.ones(N))
