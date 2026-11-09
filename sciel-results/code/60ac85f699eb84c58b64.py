# EVOLVE-BLOCK-START
def construct():
    """
    Derive six coefficients for the growth-index ansatz:
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    where q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    
    We optimize the coefficients to minimize the maximum relative error
    against the Linder approximation on a dense grid of cosmological parameters.
    """
    import numpy as np
    from scipy.optimize import minimize

    # Define a dense grid for evaluation
    # Om0 in [0.2, 0.4], w0 in [-1.2, 0.2], wa in [-1.0, 1.0]
    # a in [0.1, 1.0]
    
    # Use a subset for optimization to keep it fast, but dense enough
    Om0_vals = np.linspace(0.2, 0.35, 10)
    w0_vals = np.linspace(-1.0, 0.0, 10)
    wa_vals = np.linspace(-0.5, 0.5, 10)
    a_vals = np.linspace(0.1, 1.0, 20)
    
    # Precompute arrays for speed
    Om0_grid, w0_grid, wa_grid, a_grid = np.meshgrid(Om0_vals, w0_vals, wa_vals, a_vals, indexing='ij')
    Om0_flat = Om0_grid.ravel()
    w0_flat = w0_grid.ravel()
    wa_flat = wa_grid.ravel()
    a_flat = a_grid.ravel()
    
    # Compute E^2 and Om(a) for all points
    E2 = np.zeros_like(Om0_flat)
    for i in range(len(Om0_flat)):
        Om0 = Om0_flat[i]
        wa = wa_flat[i]
        a = a_flat[i]
        if abs(wa) < 1e-10:
            E2[i] = Om0 * a**3 + (1 - Om0) * a**2
        else:
            E2[i] = Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))
            
    Om_a = np.zeros_like(Om0_flat)
    valid_mask = np.zeros_like(Om0_flat, dtype=bool)
    for i in range(len(Om0_flat)):
        if E2[i] > 0:
            Om_a[i] = Om0_flat[i] * a_flat[i]**3 / E2[i]
            if 0 < Om_a[i] < 1:
                valid_mask[i] = True
                
    Om_a = Om_a[valid_mask]
    w0_v = w0_flat[valid_mask]
    wa_v = wa_flat[valid_mask]
    a_v = a_flat[valid_mask]
    
    if len(Om_a) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    q = 1 - Om_a
    u = 1 + w0_v + wa_v * (1 - a_v)
    
    # Filter out invalid u
    u_mask = u > 0
    if not np.all(u_mask):
        q = q[u_mask]
        u = u[u_mask]
        wa_v = wa_v[u_mask]
        a_v = a_v[u_mask]
        
    if len(q) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Target gamma from Linder
    gamma_target = (5.0 / 7.0) * (Om_a[u_mask] / u)**0.38
    
    # Design matrix
    A = np.column_stack([
        np.ones_like(q),
        q,
        q**2,
        u,
        wa_v * a_v,
        u * q
    ])
    
    # Initial guess: Least Squares
    coeffs_lsq, residuals, rank, sv = np.linalg.lstsq(A, gamma_target, rcond=None)
    coeffs_lsq = np.clip(coeffs_lsq, -10.0, 10.0)
    
    # Define loss function for optimization
    def loss(c):
        c = np.clip(c, -10.0, 10.0)
        gamma_pred = A @ c
        # Check bounds
        if np.any(gamma_pred < 0.1) or np.any(gamma_pred > 1.2):
            return 100.0
        rel_errors = np.abs((gamma_pred - gamma_target) / gamma_target)
        return np.max(rel_errors)
    
    # Optimize
    # Use Nelder-Mead or L-BFGS-B
    # L-BFGS-B is better for bounded problems, but loss is not smooth due to max.
    # Nelder-Mead is robust to non-smoothness.
    
    res = minimize(loss, coeffs_lsq, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    best_coeffs = res.x
    best_loss = res.fun
    
    # If optimization failed or didn't improve, use LSQ
    if best_loss > loss(coeffs_lsq):
        best_coeffs = coeffs_lsq
        best_loss = loss(coeffs_lsq)
        
    # Final validation on a separate dense grid to ensure robustness
    # We'll just check the bounds on the current grid again
    gamma_final = A @ best_coeffs
    if np.any(gamma_final < 0.1) or np.any(gamma_final > 1.2):
        # Fallback to constant
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    return [float(c) for c in best_coeffs]
# EVOLVE-BLOCK-END
