def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    
    We minimize the product of the spatial variance and the frequency variance
    (which corresponds to the upper bound on the uncertainty constant C4)
    using a convex optimization approach over magnitudes and sign patterns.
    """
    import numpy as np
    from scipy.optimize import minimize

    # Hermite values at 0
    h0, h4, h8, h12 = 1.0, 3.0, 105.0, 10395.0
    
    # Weights for Norm (N), X moment, K moment
    # N = sum a_i^2 i!
    # X = sum a_i^2 (2i+1) i!
    # K = sum a_i^2 i * i!
    
    f = [1.0, 24.0, 40320.0, 479001600.0] # i!
    w_n = np.array(f)
    w_x = np.array([1.0, 9.0, 17.0, 25.0]) * np.array(f) # (2i+1) i!
    w_k = np.array([0.0, 4.0, 8.0, 12.0]) * np.array(f)  # i * i!
    
    h = np.array([h0, h4, h8, h12])
    
    best_val = float('inf')
    best_a = None
    
    # Sign patterns: sigma_0 is fixed to 1, sigma_1, sigma_2, sigma_3 vary
    # There are 8 patterns.
    # We solve for u (magnitudes) >= 0
    # Minimize F(u) = X(u) K(u)
    # Subject to N(u) = 1
    #           sum sigma_i h_i u_i = 0
    
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                sigma = np.array([1.0, s1, s2, s3])
                
                def obj_and_grad(u):
                    # u is array of 4 magnitudes
                    u2 = u * u
                    X = np.dot(w_x, u2)
                    K = np.dot(w_k, u2)
                    val = X * K
                    
                    # Gradient
                    # dX/du = 2 u w_x
                    # dK/du = 2 u w_k
                    # d(XK)/du = K * 2 u w_x + X * 2 u w_k
                    grad = 2 * u * (K * w_x + X * w_k)
                    return val, grad

                def obj_only(u):
                    return obj_and_grad(u)[0]

                def grad_only(u):
                    return obj_and_grad(u)[1]

                # Constraints
                # 1. N(u) = 1
                # 2. sum sigma_i h_i u_i = 0
                constraints = [
                    {'type': 'eq', 'fun': lambda u: np.dot(w_n, u*u) - 1.0,
                     'jac': lambda u: 2 * u * w_n},
                    {'type': 'eq', 'fun': lambda u: np.dot(sigma * h, u),
                     'jac': lambda u: sigma * h}
                ]
                
                bounds = [(0, None), (0, None), (0, None), (0, None)]
                
                # Initial guess: distribute weight evenly?
                # Need to satisfy sum sigma h u = 0.
                # A simple feasible start is hard to find analytically for all sigmas.
                # Let's try a few starts.
                starts = [
                    np.array([1.0, 1.0, 1.0, 1.0]),
                    np.array([0.5, 0.5, 0.5, 0.5]),
                    np.array([1.0, 0.1, 0.1, 0.1]),
                    np.array([0.1, 1.0, 0.1, 0.1]),
                    np.array([0.1, 0.1, 1.0, 0.1]),
                    np.array([0.1, 0.1, 0.1, 1.0]),
                ]
                
                for start in starts:
                    try:
                        res = minimize(obj_only, start, jac=grad_only, method='SLSQP',
                                       bounds=bounds, constraints=constraints,
                                       options={'ftol': 1e-15, 'maxiter': 1000})
                        if res.success:
                            val = res.fun
                            # Check constraint satisfaction
                            u_opt = res.x
                            c1 = np.dot(w_n, u_opt*u_opt) - 1.0
                            c2 = np.dot(sigma * h, u_opt)
                            if abs(c1) < 1e-8 and abs(c2) < 1e-8:
                                if val < best_val:
                                    best_val = val
                                    # Reconstruct a
                                    a_opt = sigma * u_opt
                                    # We need to return a0, a4, a8
                                    # Note: The sign of the whole function doesn't matter for the score,
                                    # but the relative signs are fixed by sigma.
                                    best_a = a_opt[:3]
                    except Exception:
                        pass

    if best_a is None:
        # Fallback
        best_a = np.array([-113/100, 1/25, 1/3240])
        
    return best_a
