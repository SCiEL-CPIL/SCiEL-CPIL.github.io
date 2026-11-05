import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    def generate_h(params):
        phi0, mu = params
        k = np.arange(N)
        h = np.cos(phi0 + mu * k**2)
        return h

    def objective(params):
        h = generate_h(params)
        sum_h = np.sum(h)
        if abs(sum_h) < 1e-8:
            return 1e10
        conv_h = np.convolve(h, h)
        max_conv = np.max(conv_h)
        value = abs(2 * N * max_conv / (sum_h ** 2))
        return value

    best_val = 1e10
    best_params = None

    # 1. Coarse Grid Search
    # Scan mu in log space and phi0 in linear space
    # The optimal mu is likely such that the chirp has several cycles
    mus = np.logspace(-5, -2, 60)
    phis = np.linspace(0, 2 * np.pi, 30)
    
    for mu in mus:
        for phi0 in phis:
            val = objective([phi0, mu])
            if val < best_val:
                best_val = val
                best_params = [phi0, mu]

    # 2. Local Refinement using L-BFGS-B
    # Start from the best grid point
    if best_params is not None:
        # Define bounds for refinement
        # phi0 is periodic, but L-BFGS-B needs bounds. 
        # We can use a wide range or just refine locally.
        # Since the function is smooth, a local search from the grid point is good.
        res = minimize(
            objective, 
            best_params, 
            method='L-BFGS-B',
            bounds=[(0, 2 * np.pi), (1e-6, 1e-1)],
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    # 3. Random Restarts for L-BFGS-B
    # Explore the parameter space more thoroughly with local optimization
    rng = np.random.default_rng(42)
    for _ in range(50):
        # Sample mu in log space
        log_mu = rng.uniform(-5, -2)
        mu = 10 ** log_mu
        phi0 = rng.uniform(0, 2 * np.pi)
        p0 = [phi0, mu]
        
        res = minimize(
            objective, 
            p0, 
            method='L-BFGS-B',
            bounds=[(0, 2 * np.pi), (1e-6, 1e-1)],
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    # Final check and construction
    h = generate_h(best_params)
    sum_h = np.sum(h)
    if abs(sum_h) < 1e-6:
        return np.ones(N)
        
    conv_h = np.convolve(h, h)
    max_conv = np.max(conv_h)
    val = abs(2 * N * max_conv / (sum_h ** 2))
    
    # Compare with seed (value 2.0)
    if val < 2.0:
        return h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
