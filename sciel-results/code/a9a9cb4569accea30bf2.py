import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import differential_evolution, minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin for 630s limit
    
    k = 10
    n_m = 2 * k + 1 # 21 coefficients
    
    # 1. Setup Quadrature Grid
    # 70-point Gauss-Legendre in mu = cos(theta)
    mu, w = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    # Flatten grid
    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    
    n_points = len(mu_flat)
    
    # 2. Construct Basis Functions
    # Y_m for m = -10..10
    # Y_0 = N_0 P_10^0(mu)
    # Y_m = sqrt(2) N_m P_10^m(mu) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(mu) sin(|m| phi) for m < 0
    
    A = np.zeros((n_points, n_m))
    
    # Precompute factorials or use log for stability? 10! is small.
    # N_m = sqrt(21/(4 pi) * (10-m)! / (10+m)!)
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)
    
    # m = 0
    m = 0
    idx = 10 # index in coeffs: m = i - 10 => i = m + 10
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    # m > 0
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
        
    # 3. Objective Function
    def objective(c):
        # c is shape (21,)
        f_vals = A @ c # shape (n_points,)
        
        # Norms
        # ||f||_2^2 = sum w_i f_i^2 * d_phi
        # ||f||_4^4 = sum w_i f_i^4 * d_phi
        # Ratio R = (||f||_4^4)^0.25 / (||f||_2^2)^0.5
        
        f2 = f_vals ** 2
        f4 = f2 ** 2
        
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return 1e10 # Avoid division by zero, penalize zero
            
        R = (norm4_4 ** 0.25) / (norm2_sq ** 0.5)
        return R

    # 4. Optimization
    # Use Differential Evolution for global search
    # Bounds: Coefficients can be anything, but scale doesn't matter.
    # We can restrict bounds to [-10, 10] or similar.
    # Actually, since R is scale invariant, we can fix one coefficient or normalize.
    # But DE works on the vector.
    # Let's try bounds [-1, 1] * 10?
    # The "flattest" harmonic usually has coefficients of comparable magnitude.
    # All ones is a start.
    
    bounds = [(-10, 10)] * n_m
    
    # Check time
    if time.time() > deadline - 60:
        return np.ones(n_m)
        
    try:
        # DE parameters
        # popsize=20, maxiter=200 is a reasonable start
        # tol=1e-6
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=20, 
            tol=1e-8,
            polish=False, # We will polish manually
            updating='immediate'
        )
        
        best_c = res_de.x
        best_val = res_de.fun
        
    except Exception as e:
        # Fallback to all ones if DE fails or times out badly
        best_c = np.ones(n_m)
        best_val = objective(best_c)
        
    # Refine with BFGS
    # BFGS needs gradient? No, it approximates.
    # But providing gradient is faster.
    # Gradient of R(c) is complex. Let's use numerical gradient via finite differences in BFGS?
    # Or just use L-BFGS-B with approximated gradient.
    
    # Actually, let's implement the gradient analytically if possible?
    # R = N4 / N2
    # N2 = sqrt(sum w f^2)
    # N4 = (sum w f^4)^0.25
    # dR/dc = (1/N2) dN4/dc - (N4/N2^2) dN2/dc
    # dN2/dc = (1/(2 N2)) d(N2^2)/dc = (1/(2 N2)) sum 2 w f (A^T) = (1/N2) sum w f A^T
    # dN4/dc = (1/4) (sum w f^4)^(-0.75) * sum 4 w f^3 (A^T) = N4^3 * sum w f^3 A^T
    # So dR/dc = (1/N2) * N4^3 * sum w f^3 A^T - (N4/N2^2) * (1/N2) sum w f A^T
    #          = N4^3/N2 sum w f^3 A^T - N4/N2^3 sum w f A^T
    
    def objective_and_grad(c):
        f_vals = A @ c
        f2 = f_vals ** 2
        f4 = f2 ** 2
        
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradients
        # term1: N4^3/N2 * sum w f^3 A^T
        # term2: N4/N2^3 * sum w f A^T
        
        w_f = w_flat * f_vals
        w_f3 = w_flat * f_vals ** 3
        
        sum_wf_A = A.T @ w_f * d_phi
        sum_wf3_A = A.T @ w_f3 * d_phi
        
        grad = (N4**3 / N2) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        return R, grad

    # Polish
    if time.time() < deadline - 30:
        try:
            res_bfgs = minimize(
                objective_and_grad, 
                best_c, 
                jac=True, 
                method='BFGS', 
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            if res_bfgs.fun < best_val:
                best_c = res_bfgs.x
                best_val = res_bfgs.fun
        except Exception:
            pass

    # Final check on validity
    if not np.all(np.isfinite(best_c)):
        return np.ones(n_m)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Normalize if out of bounds?
        # The problem says invalid if max|c| < 1e-3 or > 1e5.
        # Since R is scale invariant, we can scale c to be within bounds.
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c

# EVOLVE-BLOCK-END
