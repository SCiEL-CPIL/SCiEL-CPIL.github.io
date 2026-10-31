import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def _exact_ratio(h):
    """
    Compute the exact ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _objective_log(z):
    """
    Objective function for log-space optimization.
    z: log-space variables.
    h = exp(z)
    We normalize h to have L1 norm 1 to enforce scale invariance and keep values bounded.
    Minimize -ratio.
    """
    h = np.exp(z)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 1e6
    h = h / l1  # Normalize to L1=1
    
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    
    if max_conv < 1e-12:
        return 1e6
        
    ratio = l2_sq / max_conv # Since l1=1, ratio is l2_sq / max_conv
    
    return -ratio

def _objective_linear(h):
    """
    Objective function for linear-space optimization (for local refinement).
    h: non-negative vector.
    """
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 1e6
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(conv)
    
    if max_conv < 1e-12:
        return 1e6
        
    ratio = l2_sq / max_conv
    return -ratio

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _exact_ratio(best_h)
    
    # 1. Global Search: Differential Evolution in Log-Space
    # Log-space ensures h > 0. 
    # Bounds for z: exp(z) should cover a wide range.
    # If z is in [-10, 10], h is in [~0, ~22000].
    # Since we normalize to L1=1, the absolute scale doesn't matter, only relative differences.
    # However, DE works better with bounded variables.
    # Let's use bounds [-5, 5] for z. This covers h ratios of exp(10) ~ 22000, which is plenty.
    bounds = [(-5.0, 5.0) for _ in range(N)]
    
    # Initial population: mix of uniform and random
    # DE handles initialization internally, but we can specify init='latinhypercube' for coverage
    
    try:
        # Use DE to find a good basin
        # maxiter and popsize control runtime.
        # N=50 is high dimension. DE might be slow.
        # Let's try a moderate population size.
        res_de = differential_evolution(
            _objective_log, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15, 
            tol=1e-5,
            init='latinhypercube',
            polish=False # We will do local refinement
        )
        
        if time.time() < deadline - 30:
            z_de = res_de.x
            h_de = np.exp(z_de)
            h_de = h_de / np.sum(h_de) # Normalize
            
            # 2. Local Refinement: L-BFGS-B in Log-Space
            # Start from DE result
            res_local = minimize(
                _objective_log, 
                z_de, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.success:
                z_opt = res_local.x
                h_opt = np.exp(z_opt)
                h_opt = h_opt / np.sum(h_opt)
                score_opt = _exact_ratio(h_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    
    except Exception as e:
        pass # Fallback to other strategies if DE fails or times out

    # 3. Additional Local Refinements from Random Starts
    # If DE didn't find a great solution, or if we have time, try random starts in log-space
    # with L-BFGS-B.
    
    if time.time() < deadline - 20:
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(100 + restart)
            # Random log-space start
            z_rand = np.random.uniform(-3, 3, N)
            
            try:
                res_rand = minimize(
                    _objective_log, 
                    z_rand, 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                
                if res_rand.success:
                    z_opt = res_rand.x
                    h_opt = np.exp(z_opt)
                    h_opt = h_opt / np.sum(h_opt)
                    score_opt = _exact_ratio(h_opt)
                    
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt.copy()
                        
            except Exception:
                pass

    # 4. Try a few structured initializations (e.g., Gaussian bumps, sparse peaks)
    if time.time() < deadline - 5:
        # Gaussian bump
        x = np.linspace(-1, 1, N)
        for sigma in [0.2, 0.5, 1.0]:
            h_gauss = np.exp(-x**2 / (2*sigma**2))
            h_gauss = h_gauss / np.sum(h_gauss)
            z_gauss = np.log(h_gauss + 1e-10)
            
            try:
                res_g = minimize(
                    _objective_log, 
                    z_gauss, 
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'ftol': 1e-12}
                )
                if res_g.success:
                    h_opt = np.exp(res_g.x)
                    h_opt = h_opt / np.sum(h_opt)
                    score_opt = _exact_ratio(h_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt.copy()
            except:
                pass
        
        # Sparse peak (single non-zero)
        for i in range(0, N, 5):
            h_sparse = np.zeros(N)
            h_sparse[i] = 1.0
            # This is a limit case, ratio = 1 / 1 = 1? 
            # l2^2 = 1, l1 = 1, conv max = 1. Ratio = 1.
            # Uniform is also 1.
            # We are looking for > 1.
            # So sparse is not helpful as a starting point for >1, but good for baseline.
            
    # Ensure non-negative and valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-10:
        best_h = np.ones(N) / N
    
    return best_h.tolist()

if __name__ == "__main__":
    # Quick self-test
    h = construct()
    print(f"Score: {_exact_ratio(np.array(h))}")
