# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    k = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 10s margin before 600s limit

    # Set up quadrature points
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    
    # 130 equally spaced phi points
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Precompute sin/cos of m*phi for m = 0..10
    cos_m_phi = np.zeros((11, n_phi))
    sin_m_phi = np.zeros((11, n_phi))
    for m in range(11):
        if m == 0:
            cos_m_phi[m] = 1.0
            sin_m_phi[m] = 0.0
        else:
            cos_m_phi[m] = np.cos(m * phi)
            sin_m_phi[m] = np.sin(m * phi)
    
    # Precompute normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * math.factorial(10 - m) / math.factorial(10 + m))
    
    # Precompute associated Legendre polynomials P_10^m(x) for x in x_gl
    P_legendre = np.zeros((11, 70))
    for m in range(11):
        P_legendre[m] = lpmv(m, 10, x_gl)
    
    # Build the full matrix Y_flat of shape (70*130, 21)
    # Indexing: row = i*n_phi + j, col = m+10
    N_q = 70 * n_phi
    Y_flat = np.zeros((N_q, 21), dtype=np.float64)
    
    # Weights for each quadrature point
    w_flat = np.zeros(N_q, dtype=np.float64)
    
    for i in range(70):
        row_start = i * n_phi
        row_end = (i + 1) * n_phi
        w_flat[row_start:row_end] = w_gl[i] * dphi
        
        # m = 0
        Y_flat[row_start:row_end, 10] = N[0] * P_legendre[0, i]
        
        # m > 0 and m < 0
        for m in range(1, 11):
            coeff_m_i = np.sqrt(2.0) * N[m] * P_legendre[m, i]
            # c_m (index 10+m) corresponds to cos(m*phi)
            Y_flat[row_start:row_end, 10 + m] = coeff_m_i * cos_m_phi[m]
            # c_-m (index 10-m) corresponds to sin(m*phi)
            Y_flat[row_start:row_end, 10 - m] = coeff_m_i * sin_m_phi[m]

    def objective_J(c_vec):
        """
        Compute J = S4 / S2^2 where S2 = sum(w f^2), S4 = sum(w f^4).
        This is proportional to R^4.
        """
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        if S2 <= 1e-300:
            return 1e100
        return S4 / (S2 * S2)

    def jac_J(c_vec):
        """
        Gradient of J = S4 / S2^2.
        """
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        
        if S2 <= 1e-300:
            return np.zeros(21)
            
        # dS2/dc = 2 * Y^T (w f)
        # dS4/dc = 4 * Y^T (w f^3)
        wf = w_flat * f
        wf3 = w_flat * f * f2
        
        dS2 = 2.0 * (Y_flat.T @ wf)
        dS4 = 4.0 * (Y_flat.T @ wf3)
        
        grad = (S2 * dS4 - 2.0 * S4 * dS2) / (S2 * S2 * S2)
        return grad

    # Constraint: ||c||^2 = 1
    def norm_constraint(c_vec):
        return np.dot(c_vec, c_vec) - 1.0

    def norm_jac(c_vec):
        return 2.0 * c_vec

    cons = [{'type': 'eq', 'fun': norm_constraint, 'jac': norm_jac}]

    best_result = None
    best_obj = np.inf

    # Define starting points on the unit sphere
    starts = []
    
    # 1. m=0 only
    c_m0 = np.zeros(21)
    c_m0[10] = 1.0
    starts.append(c_m0)
    
    # 2. m=10 (cos) only
    c_m10 = np.zeros(21)
    c_m10[20] = 1.0
    starts.append(c_m10)
    
    # 3. m=-10 (sin) only
    c_mneg10 = np.zeros(21)
    c_mneg10[0] = 1.0
    starts.append(c_mneg10)
    
    # 4. Mix of m=0 and m=10
    c_mix1 = np.zeros(21)
    c_mix1[10] = 1.0 / np.sqrt(2)
    c_mix1[20] = 1.0 / np.sqrt(2)
    starts.append(c_mix1)
    
    # 5. Mix of m=0 and m=-10
    c_mix2 = np.zeros(21)
    c_mix2[10] = 1.0 / np.sqrt(2)
    c_mix2[0] = 1.0 / np.sqrt(2)
    starts.append(c_mix2)

    # 6. Random starts
    for seed in range(30):
        np.random.seed(seed)
        c_rand = np.random.randn(21)
        c_rand = c_rand / np.linalg.norm(c_rand)
        starts.append(c_rand)

    # Optimization loop
    for i, c_start in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        try:
            result = minimize(
                objective_J,
                c_start,
                method='SLSQP',
                jac=jac_J,
                constraints=cons,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12
                }
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x.copy()
                
        except Exception:
            pass
            
    if best_result is None:
        best_result = np.ones(21) / np.sqrt(21)
    
    # Polish step: Run a tighter optimization from the best result
    if time.time() < deadline - 2.0:
        try:
            result_polish = minimize(
                objective_J,
                best_result,
                method='SLSQP',
                jac=jac_J,
                constraints=cons,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-15
                }
            )
            if result_polish.fun < best_obj:
                best_obj = result_polish.fun
                best_result = result_polish.x.copy()
        except Exception:
            pass

    c = best_result
    
    # Ensure valid range
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c / max_abs * 1e-2
    elif max_abs > 1e5:
        c = c / max_abs * 1e4
        
    return c

# EVOLVE-BLOCK-END
