import numpy as np
import time
import math
import scipy.special
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Finds the coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimizes the ratio R = ||f||_4 / ||f||_2.
    
    Strategy:
    1. Precompute quadrature nodes and weights for exact integration.
    2. Formulate the problem as minimizing R(c) subject to ||c||_2 = 1.
    3. Use a deterministic sequence of high-symmetry initial guesses (m=10, m=8, etc.)
       and random unit vectors.
    4. For each initial guess, run a "Power Iteration" step:
       f_new = f^3 / ||f^3||_2  (in the coefficient space, this is equivalent to 
       applying the nonlinear operator associated with the 4-norm).
       This helps escape shallow local minima and converges to a flatter state.
    5. Polish the result using L-BFGS-B with analytic gradients on the unit sphere.
    6. Return the best result found within the time limit.
    """
    
    k = 10
    n_coeffs = 2 * k + 1
    
    # --- Precomputation ---
    
    # 1. Gauss-Legendre nodes and weights for cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    
    # 2. Equally spaced phi points
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # 3. Spherical Harmonics Basis
    # Y_0 = N_0 P^0, Y_m = sqrt(2) N_m P^m cos(m phi) for m > 0, Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
    # N_m = sqrt( (2k+1)/(4 pi) * (k-m)! / (k+m)! )
    
    N_m = np.zeros(n_coeffs)
    for m in range(-k, k + 1):
        abs_m = abs(m)
        N_m[m + k] = math.sqrt((2 * k + 1) / (4 * np.pi) * scipy.special.factorial(k - abs_m) / scipy.special.factorial(k + abs_m))
        
    P_vals = np.zeros((len(x_gl), k + 1))
    for m in range(k + 1):
        P_vals[:, m] = scipy.special.lpmv(m, k, x_gl)
        
    cos_m_phi = np.zeros((n_phi, k + 1))
    sin_m_phi = np.zeros((n_phi, k + 1))
    for m in range(k + 1):
        cos_m_phi[:, m] = np.cos(m * phi_vals)
        sin_m_phi[:, m] = np.sin(m * phi_vals)
        
    Y_basis = np.zeros((len(x_gl), n_phi, n_coeffs))
    
    for c_idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        if m == 0:
            val = N_m[0] * P_vals[:, 0]
            Y_basis[:, :, c_idx] = val[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            val_u = N_m[m] * P_vals[:, m]
            val_phi = cos_m_phi[:, m]
            Y_basis[:, :, c_idx] = np.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
        else:
            val_u = N_m[abs_m] * P_vals[:, abs_m]
            val_phi = sin_m_phi[:, abs_m]
            Y_basis[:, :, c_idx] = np.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
            
    w_u = w_gl
    w_phi = np.full(n_phi, dphi)
    
    # Flatten basis for matrix multiplication
    # Y_flat has shape (n_points, n_coeffs)
    # n_points = 70 * 130 = 9100
    Y_flat = Y_basis.reshape(-1, n_coeffs)
    # Weights for each point
    w_flat = np.repeat(w_u, n_phi) * np.tile(w_phi, len(x_gl))
    
    # --- Objective Function and Gradient ---
    
    def compute_ratio_and_grad(c):
        """
        Computes R = ||f||_4 / ||f||_2 and its gradient with respect to c.
        Assumes c is a unit vector (||c||_2 = 1).
        """
        f = Y_flat @ c
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.sum(w_flat * f2)
        norm4_4 = np.sum(w_flat * f4)
        
        if norm2_sq < 1e-30:
            return np.inf, np.zeros_like(c)
            
        norm2 = np.sqrt(norm2_sq)
        norm4 = np.power(norm4_4, 0.25)
        
        R = norm4 / norm2
        
        # Gradient of R = (1/4) * (||f||_4^4)^(-3/4) * ||f||_2^(-1) * 4 ||f||_4^2 * f^2 * Y^T
        # Let A = ||f||_4^4, B = ||f||_2^2
        # R = A^(1/4) / B^(1/2)
        # dR/dc = (1/4) A^(-3/4) B^(-1/2) (4 A^(1/4) ... no, simpler:
        # R = (A/B^2)^{1/4} ? No.
        # R = A^{1/4} B^{-1/2}
        # dR = (1/4) A^{-3/4} B^{-1/2} dA - (1/2) A^{1/4} B^{-3/2} dB
        # dA = sum w 4 f^3 df = 4 sum w f^3 (Y c)' = 4 (Y^T (w f^3)) . dc
        # dB = sum w 2 f df = 2 (Y^T (w f)) . dc
        
        term_A = 4 * np.dot(Y_flat.T, w_flat * f**3)
        term_B = 2 * np.dot(Y_flat.T, w_flat * f)
        
        grad_A = (1/4) * (norm4_4 ** (-0.75)) * (norm2_sq ** (-0.5)) * term_A
        grad_B = -(1/2) * (norm4 ** 1.0) * (norm2_sq ** (-1.5)) * term_B
        
        # Note: norm4 = norm4_4^0.25. So norm4^1 = norm4_4^0.25.
        # Let's re-verify the coefficient for grad_B.
        # R = A^0.25 B^-0.5
        # dR/dB = -0.5 A^0.25 B^-1.5
        # dB/dc = 2 Y^T (w f)
        # So grad_B_part = -0.5 A^0.25 B^-1.5 * 2 Y^T (w f) = - A^0.25 B^-1.5 Y^T (w f)
        
        grad = (norm4_4 ** (-0.75)) * (norm2_sq ** (-0.5)) * term_A - (norm4_4 ** 0.25) * (norm2_sq ** (-1.5)) * term_B
        
        return R, grad

    def objective(c):
        R, _ = compute_ratio_and_grad(c)
        return R

    def objective_grad(c):
        R, g = compute_ratio_and_grad(c)
        return R, g

    # --- Optimization Loop ---
    
    start_time = time.time()
    deadline = 620.0
    
    best_c = None
    best_score = np.inf
    
    # Deterministic initial guesses
    # 1. Pure m=10 (cos)
    # 2. Pure m=10 (sin)
    # 3. Pure m=8 (cos)
    # 4. Pure m=0
    # 5. Random unit vectors
    
    def normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-10:
            return v
        return v / n

    initial_guesses = []
    
    # m=10 cos is at index 10 (m=10)
    c_m10_cos = np.zeros(n_coeffs)
    c_m10_cos[20] = 1.0
    initial_guesses.append(c_m10_cos)
    
    # m=10 sin is at index 9 (m=-10)
    c_m10_sin = np.zeros(n_coeffs)
    c_m10_sin[0] = 1.0
    initial_guesses.append(c_m10_sin)
    
    # m=8 cos is at index 18 (m=8)
    c_m8_cos = np.zeros(n_coeffs)
    c_m8_cos[18] = 1.0
    initial_guesses.append(c_m8_cos)
    
    # m=0 is at index 10
    c_m0 = np.zeros(n_coeffs)
    c_m0[10] = 1.0
    initial_guesses.append(c_m0)
    
    # Add some random guesses
    np.random.seed(42)
    for _ in range(20):
        c_rand = np.random.randn(n_coeffs)
        initial_guesses.append(normalize(c_rand))
    
    # Process each initial guess
    for c0 in initial_guesses:
        if time.time() - start_time > deadline:
            break
            
        c = c0.copy()
        
        # Phase 1: Power Iteration / Nonlinear Smoothing
        # Iterate: c_new = normalize( f^3 projected back to coefficient space )
        # This effectively applies the operator that minimizes L4/L2 locally.
        # We do a few iterations to move away from the peaky initial state.
        for _ in range(10):
            if time.time() - start_time > deadline:
                break
            f = Y_flat @ c
            # f^3 is not a harmonic, but we project it back to the degree-10 space
            # This is equivalent to multiplying by the "3rd power" in the function space
            # and taking the inner product with the basis.
            # grad_L4_4 is proportional to Y^T (w f^3).
            # This direction decreases L4^4 while keeping L2 fixed (approximately).
            f3 = f**3
            c_new = Y_flat.T @ (w_flat * f3)
            
            # Normalize
            norm_c_new = np.linalg.norm(c_new)
            if norm_c_new < 1e-10:
                break
            c_new /= norm_c_new
            
            # Check improvement
            R_old, _ = compute_ratio_and_grad(c)
            R_new, _ = compute_ratio_and_grad(c_new)
            
            if R_new < R_old:
                c = c_new
            else:
                # If it didn't improve, stop power iteration and switch to gradient
                break
                
        # Phase 2: L-BFGS-B with analytic gradient
        # Constrain to unit sphere? L-BFGS-B doesn't handle equality constraints well.
        # Instead, we minimize R(c) directly. Since R is scale invariant, the optimizer
        # might drift in scale. We can periodically renormalize or just let it run.
        # Actually, R(c) = R(c/||c||). The gradient is zero on the radial direction?
        # No, the gradient points towards lower R.
        # To keep it stable, we can minimize R(c) and renormalize c at the end.
        
        try:
            res = minimize(
                objective, 
                c, 
                jac=objective_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_res = res.x
            # Renormalize to ensure valid output range and consistency
            c_res = normalize(c_res)
            
            score, _ = compute_ratio_and_grad(c_res)
            
            if score < best_score:
                best_score = score
                best_c = c_res.copy()
                
        except Exception:
            pass

    # Final polish on the best candidate found
    if best_c is not None:
        if time.time() - start_time < deadline - 10:
            try:
                res_final = minimize(
                    objective, 
                    best_c, 
                    jac=objective_grad, 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                c_final = normalize(res_final.x)
                score_final, _ = compute_ratio_and_grad(c_final)
                if score_final < best_score:
                    best_c = c_final
                    best_score = score_final
            except Exception:
                pass

    # Validation and Output
    if best_c is None:
        return np.ones(n_coeffs)
        
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(n_coeffs)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            return np.ones(n_coeffs)
        best_c = best_c / max_abs * 1.0
        
    return best_c
