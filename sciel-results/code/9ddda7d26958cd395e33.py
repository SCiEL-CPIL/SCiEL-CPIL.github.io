import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    def generate_h(params):
        phi0, theta, mu = params
        k = np.arange(N)
        # h[k] = cos(phi0 + theta*k + mu*k^2)
        # theta is linear frequency, mu is quadratic chirp rate
        h = np.cos(phi0 + theta * k + mu * k**2)
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

    # 1. Global Search using Differential Evolution
    # Parameters: phi0 [0, 2pi), theta [-pi, pi], mu [1e-4, 100]
    # We restrict mu to a smaller upper bound to avoid numerical issues or overly rapid oscillations 
    # that might not help with the rectangular window sidelobes as effectively as moderate chirps.
    # However, the previous run went up to 1e3. Let's keep it wide but maybe log-uniform is better.
    # DE works on linear space. Let's use a moderate range for mu to ensure stability.
    bounds = [(0, 2 * np.pi), (-np.pi, np.pi), (1e-4, 50.0)]
    
    # Run DE with a large population and many iterations
    result_de = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=300, 
        tol=1e-8, 
        popsize=40, 
        init='latinhypercube',
        polish=False 
    )
    
    best_val = result_de.fun
    best_params = result_de.x
    
    # 2. Local Refinement using Nelder-Mead
    result_nm = minimize(
        objective, 
        best_params, 
        method='Nelder-Mead',
        options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 10000}
    )
    
    if result_nm.fun < best_val:
        best_val = result_nm.fun
        best_params = result_nm.x

    # 3. Random restarts
    rng = np.random.default_rng(42)
    for _ in range(15):
        p0 = [rng.uniform(0, 2*np.pi), rng.uniform(-np.pi, np.pi), rng.uniform(1e-4, 50.0)]
        res = minimize(
            objective, 
            p0, 
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000, 'maxfev': 5000}
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
