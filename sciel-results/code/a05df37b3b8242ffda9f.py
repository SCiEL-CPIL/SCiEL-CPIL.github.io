# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import differential_evolution, minimize

def rudin_problem10() -> np.ndarray:
    """
    Minimizes the L4/L2 norm ratio for degree-10 spherical harmonics using 
    Differential Evolution (global stochastic search) followed by L-BFGS-B polish.
    Returns the optimal coefficients c_m for m = -10..10.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # 1. Setup Grid
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    cos_theta_vals = x_gl
    # sin_theta_vals not explicitly needed for weights if we use dOmega = dcos_theta dphi
    # Wait, dOmega = sin(theta) dtheta dphi.
    # x = cos(theta), dx = -sin(theta) dtheta.
    # So sin(theta) dtheta = -dx.
    # Integral over theta 0 to pi corresponds to x 1 to -1.
    # Gauss-Legendre integrates over [-1, 1].
    # So integral = sum w_i f(x_i).
    # The weight for the surface measure is just w_gl_i * dphi.
    
    # 2. Pre-compute Basis Functions Y_m
    # Y_m values at all grid points. Shape: (n_coeffs, n_grid)
    n_grid = 70 * n_phi
    Y = np.zeros((n_coeffs, n_grid), dtype=np.float64)
    
    # Quadrature weights for the surface integral
    # Integral over sphere: sum_{i,j} f(x_i, phi_j) w_gl_i dphi
    # Weights shape: (n_grid,)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    for m_idx in range(n_coeffs):
        m = m_idx - k
        abs_m = abs(m)
        
        # Normalization constant N_m
        # N_m = sqrt( (21)/(4*pi) * (10-m)! / (10+m)! )
        num = factorial(10 - abs_m)
        den = factorial(10 + abs_m)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * (num / den))
        
        # Associated Legendre Polynomial P_10^|m|(cos_theta)
        P_vals = lpmv(abs_m, 10, cos_theta_vals)
        
        if m == 0:
            Y_vals = np.tile(P_vals[:, np.newaxis], (1, n_phi)) * N_m
        elif m > 0:
            cos_term = np.cos(m * phi)
            Y_vals = (np.sqrt(2.0) * N_m) * (P_vals[:, np.newaxis] * cos_term[np.newaxis, :])
        else:
            sin_term = np.sin(abs_m * phi)
            Y_vals = (np.sqrt(2.0) * N_m) * (P_vals[:, np.newaxis] * sin_term[np.newaxis, :])
        
        Y[m_idx, :] = Y_vals.flatten()

    # 3. Define Objective and Gradient
    # We want to minimize R(c) = ||f||_4 / ||f||_2
    # R(c) is scale invariant.
    # For Differential Evolution, we define a function on the box [-1, 1]^n_coeffs.
    # Inside the function, we normalize c to unit L2 norm (Euclidean) to stay on the sphere.
    # This makes the objective well-defined and bounded.
    
    def objective_and_grad(c):
        # c is shape (n_coeffs,)
        # Normalize c to unit Euclidean norm to ensure we are on the sphere S^{20}
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            # Handle zero vector edge case
            c_norm = np.ones(n_coeffs) / np.sqrt(n_coeffs)
        else:
            c_norm = c / norm_c
            
        f = Y.T @ c_norm
        
        f_sq = f**2
        f_four = f_sq**2
        
        A = np.sum(weights * f_sq)  # ||f||_2^2
        B = np.sum(weights * f_four) # ||f||_4^4
        
        if A < 1e-15:
            return 1e10, np.zeros_like(c)
            
        R = (B ** 0.25) / (A ** 0.5)
        
        # Gradient with respect to c (unnormalized input to DE)
        # We need dR/dc.
        # R(c) = R(c/norm_c).
        # Let u = c / ||c||.
        # dR/dc = (dR/du) * (du/dc)
        # du/dc = (I - u u^T) / ||c||
        # dR/du is the gradient on the sphere.
        
        # First compute grad_R_u = gradient of R with respect to u
        grad_A_u = 2.0 * (Y @ (weights * f))
        grad_B_u = 4.0 * (Y @ (weights * f**3))
        
        # grad R_u = R * (1/4 * grad_B_u / B - 1/2 * grad_A_u / A)
        grad_R_u = R * (0.25 * grad_B_u / B - 0.5 * grad_A_u / A)
        
        # Project gradient to tangent space (optional for correctness of chain rule if we treat u as variable)
        # Actually, for the chain rule dR/dc, we just need the Jacobian of the mapping c -> u.
        # J = (I - u u^T) / norm_c
        # grad_R_c = J^T @ grad_R_u = (I - u u^T) / norm_c @ grad_R_u
        # Note: (I - u u^T) is symmetric.
        
        # However, since we are using DE which doesn't use gradients, we only need the objective.
        # But for the final polish (L-BFGS-B), we will use a constrained or normalized approach.
        # For DE, we just return R.
        
        return R

    def objective_for_de(c):
        # c is from DE, in box [-1, 1]
        return objective_and_grad(c)[0]

    # 4. Global Search with Differential Evolution
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    # Bounds for DE: [-1, 1] for each coefficient
    bounds = [(-1.0, 1.0)] * n_coeffs
    
    best_R = np.inf
    best_c = np.ones(n_coeffs, dtype=np.float64) / np.sqrt(n_coeffs)
    
    # Run DE
    # maxiter and popsize need to be balanced against time.
    # 21 dimensions. popsize=200, maxiter=200 is a reasonable start.
    # We can adjust based on time.
    
    try:
        # Using 'best1bin' strategy
        res_de = differential_evolution(
            objective_for_de,
            bounds,
            maxiter=100,
            popsize=200,
            tol=1e-10,
            seed=42,
            polish=False, # We will polish separately to control time
            init='sobol', # Better initial spread
            updating='deferred', # More efficient memory
            workers=1
        )
        
        if res_de.success or res_de.fun < best_R:
            best_R = res_de.fun
            # res_de.x is the optimal c in the box. We need to normalize it for the final answer.
            c_de = res_de.x
            norm_c_de = np.linalg.norm(c_de)
            if norm_c_de > 1e-12:
                best_c = c_de / norm_c_de
            else:
                best_c = np.ones(n_coeffs) / np.sqrt(n_coeffs)
                
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random if DE fails completely
        rng = np.random.default_rng(42)
        best_c = rng.normal(0, 1, n_coeffs)
        best_c /= np.linalg.norm(best_c)
        best_R = objective_and_grad(best_c)[0]

    # 5. Local Refinement with L-BFGS-B on the sphere
    # We refine best_c.
    # To use L-BFGS-B effectively on the sphere, we can optimize in the tangent space or just use the gradient derived above.
    # Since R is scale invariant, L-BFGS-B on unconstrained c might drift.
    # We will use a projected gradient step or just constrain the norm.
    # Simpler: Use L-BFGS-B with the gradient derived in objective_and_grad.
    # The gradient derived in objective_and_grad is for the unnormalized c?
    # Wait, in objective_and_grad above, I calculated grad_R_u but didn't chain rule to c.
    # Let's create a specific objective/grad for the polish that handles the normalization correctly.
    
    def polish_objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros_like(c)
        u = c / norm_c
        f = Y.T @ u
        f_sq = f**2
        f_four = f_sq**2
        A = np.sum(weights * f_sq)
        B = np.sum(weights * f_four)
        if A < 1e-15:
            return 1e10, np.zeros_like(c)
        R = (B ** 0.25) / (A ** 0.5)
        
        grad_A_u = 2.0 * (Y @ (weights * f))
        grad_B_u = 4.0 * (Y @ (weights * f**3))
        grad_R_u = R * (0.25 * grad_B_u / B - 0.5 * grad_A_u / A)
        
        # Chain rule: grad_R_c = (I - u u^T) / norm_c * grad_R_u
        # (I - u u^T) is the projection matrix onto the tangent space.
        # grad_R_c = (grad_R_u - u * (u @ grad_R_u)) / norm_c
        
        u_dot_grad = np.dot(u, grad_R_u)
        grad_R_c = (grad_R_u - u * u_dot_grad) / norm_c
        
        return R, grad_R_c

    # Time check before polish
    elapsed = time.time() - start_time
    if elapsed < time_limit - safety_margin:
        try:
            # Use L-BFGS-B to polish
            # We start from best_c (which is on the sphere)
            # L-BFGS-B will move off the sphere, but the gradient is projected?
            # No, the gradient calculated above is the true gradient of the extended function R(c) = R(c/||c||).
            # So L-BFGS-B will follow the gradient flow which respects the scale invariance (gradient is orthogonal to c).
            res_polish = minimize(
                lambda c: polish_objective_and_grad(c)[0],
                best_c,
                jac=lambda c: polish_objective_and_grad(c)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if res_polish.fun < best_R:
                best_R = res_polish.fun
                best_c = res_polish.x
                
        except Exception as e:
            print(f"Polish failed: {e}")

    # 6. Final Validity Check and Normalization
    if not np.all(np.isfinite(best_c)):
        return np.ones(n_coeffs, dtype=np.float64)
        
    # Ensure max|c| is within bounds
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1e5 / max_abs_c)
        
    # Final normalization to unit Euclidean norm for stability, though not strictly required by contract
    # as long as max|c| is in bounds. But it's good practice.
    norm_final = np.linalg.norm(best_c)
    if norm_final > 1e-12:
        best_c = best_c / norm_final
        # Re-check bounds after normalization
        max_abs_c = np.max(np.abs(best_c))
        if max_abs_c < 1e-3:
             best_c = best_c * (1.0 / max_abs_c)
        elif max_abs_c > 1e5:
             best_c = best_c * (1e5 / max_abs_c)

    return best_c

# EVOLVE-BLOCK-END
