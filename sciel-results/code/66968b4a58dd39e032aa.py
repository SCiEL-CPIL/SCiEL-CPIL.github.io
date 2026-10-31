import numpy as np
from scipy import optimize

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    if np.sum(h) == 0:
        return 0.0
    conv = np.convolve(h, h, 'full')
    total = np.sum(h)
    return np.max(conv) / (total * total)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Baseline: Uniform distribution
    best_h = np.ones(N)
    best_score = compute_autoconv_max(best_h)
    
    # Grid points for the step function intervals
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Number of Gaussian basis functions
    n_gauss = 10
    
    # Parameters for each Gaussian: [amplitude, center, width]
    # Total parameters: 3 * n_gauss
    def gaussians_to_h(params):
        """Convert Gaussian parameters to step function heights."""
        h = np.zeros(N)
        for i in range(n_gauss):
            amp = np.exp(params[3*i])  # Positive amplitude via exp
            center = params[3*i + 1]
            width = np.exp(params[3*i + 2])  # Positive width via exp
            h += amp * np.exp(-0.5 * ((x_grid - center) / width) ** 2)
        return np.maximum(h, 0)
    
    def objective_from_params(params):
        """Compute score from Gaussian parameters."""
        h = gaussians_to_h(params)
        return compute_autoconv_max(h)
    
    # Objective for optimization (smooth version using log-sum-exp)
    def soft_objective_from_params(params, T=0.01):
        h = gaussians_to_h(params)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        conv = np.convolve(h, h, 'full')
        conv_max = np.max(conv)
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -100, 100)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)
    
    # Bounds for parameters:
    # amplitude: exp(amp_param), so amp_param in [-10, 5]
    # center: in [-0.25, 0.25]
    # width: exp(width_param), so width_param in [-5, 2]
    bounds = []
    for i in range(n_gauss):
        bounds.append((-10, 5))  # log amplitude
        bounds.append((-0.25, 0.25))  # center
        bounds.append((-5, 2))  # log width
    
    np.random.seed(42)
    
    # Stage 1: Differential evolution for global search
    if time.time() < deadline - 60:
        try:
            # Initial population bounds
            x0_min = np.array([b[0] for b in bounds])
            x0_max = np.array([b[1] for b in bounds])
            
            # Use differential evolution with moderate population
            de_result = optimize.differential_evolution(
                soft_objective_from_params,
                bounds,
                args=(0.05,),
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            best_params_de = de_result.x
            best_score_de = compute_autoconv_max(gaussians_to_h(best_params_de))
            
            if best_score_de < best_score:
                best_score = best_score_de
                best_h = gaussians_to_h(best_params_de).copy()
                
        except Exception:
            best_params_de = None
    
    # Stage 2: L-BFGS-B polish from best DE result
    if time.time() < deadline - 30:
        if best_params_de is not None:
            try:
                res1 = optimize.minimize(
                    lambda p: soft_objective_from_params(p, T=0.02),
                    best_params_de,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                h1 = gaussians_to_h(res1.x)
                score1 = compute_autoconv_max(h1)
                if score1 < best_score:
                    best_score = score1
                    best_h = h1.copy()
                    
            except Exception:
                pass
            
        # Also try L-BFGS-B from uniform-like initial guess
        if time.time() < deadline - 15:
            try:
                # Initialize with one broad Gaussian
                init_params = np.zeros(3 * n_gauss)
                init_params[0] = np.log(1.0)  # amplitude ~1
                init_params[1] = 0.0  # center
                init_params[2] = np.log(0.1)  # width ~0.1
                for i in range(1, n_gauss):
                    init_params[3*i] = np.log(0.1)
                    init_params[3*i + 1] = (i - n_gauss/2) * 0.05
                    init_params[3*i + 2] = np.log(0.05)
                
                res2 = optimize.minimize(
                    lambda p: soft_objective_from_params(p, T=0.01),
                    init_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                h2 = gaussians_to_h(res2.x)
                score2 = compute_autoconv_max(h2)
                if score2 < best_score:
                    best_score = score2
                    best_h = h2.copy()
                    
            except Exception:
                pass
    
    # Stage 3: Direct L-BFGS-B on 600-dim space from best Gaussian result
    if time.time() < deadline - 10:
        try:
            h_init = np.maximum(best_h, 0.001)
            res3 = optimize.minimize(
                lambda x: soft_objective_from_params(np.zeros(3*n_gauss), T=0.005),
                h_init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            
            # Wait, that's wrong - let me fix the objective for direct optimization
            pass
        except Exception:
            pass
    
    # Stage 3 (corrected): Direct L-BFGS-B on 600-dim space
    if time.time() < deadline - 10:
        def soft_objective_direct(x, T=0.005):
            h = np.maximum(x, 0)
            total = np.sum(h)
            if total < 1e-10:
                return 1e10
            conv = np.convolve(h, h, 'full')
            conv_max = np.max(conv)
            shifted = (conv - conv_max) / T
            shifted = np.clip(shifted, -100, 100)
            soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
            return soft_max / (total * total)
        
        try:
            h_init = np.maximum(best_h, 0.001)
            res3 = optimize.minimize(
                lambda x: soft_objective_direct(x, T=0.005),
                h_init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            
            h3 = np.maximum(res3.x, 0)
            score3 = compute_autoconv_max(h3)
            if score3 < best_score:
                best_score = score3
                best_h = h3.copy()
                
        except Exception:
            pass
    
    return list(best_h)
