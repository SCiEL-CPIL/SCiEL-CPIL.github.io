import numpy as np
import time
import math
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution, line_search

def rudin_problem10() -> np.ndarray:
    """
    Finds the coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    that minimizes the ratio R = ||f||_4 / ||f||_2.
    
    Strategy:
    1. Precompute quadrature nodes, weights, and the spherical harmonic basis matrix.
    2. Define a scale-invariant objective function that normalizes the input vector.
    3. Use Differential Evolution (a global, derivative-free optimizer) to explore the 
       parameter space broadly. This avoids the local minima traps of pure gradient descent.
    4. Polish the best candidate found by DE using L-BFGS-B with analytic gradients for precision.
    5. Time-guarded to return the best result before the 630s limit.
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
    N_m = np.zeros(n_coeffs)
    for m in range(-k, k + 1):
        abs_m = abs(m)
        N_m[m + k] = math.sqrt((2 * k + 1) / (4 * np.pi) * factorial(k - abs_m) / factorial(k + abs_m))
        
    P_vals = np.zeros((len(x_gl), k + 1))
    for m in range(k + 1):
        P_vals[:, m] = lpmv(m, k, x_gl)
        
    cos_m_phi = np.zeros((n_phi, k + 1))
    sin_m_phi = np.zeros((n_phi, k + 1))
    for m in range(k + 1):
        cos_m_phi[:, m] = np.cos(m * phi_vals)
        sin_m_phi[:, m] = np.sin(m * phi_vals)
        
    # Build the basis matrix Y
    # Y_basis shape: (n_u, n_phi, n_coeffs)
    Y_basis = np.zeros((len(x_gl), n_phi, n_coeffs))
    
    for c_idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        if m == 0:
            val = N_m[0] * P_vals[:, 0]
            Y_basis[:, :, c_idx] = val[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            val_u = N_m[m] * P_vals[:, m]
            val_phi = cos_m_phi[:, m]
            Y_basis[:, :, c_idx] = math.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
        else:
            val_u = N_m[abs_m] * P_vals[:, abs_m]
            val_phi = sin_m_phi[:, abs_m]
            Y_basis[:, :, c_idx] = math.sqrt(2) * (val_u[:, np.newaxis] * val_phi[np.newaxis, :])
            
    w_u = w_gl
    w_phi = np.full(n_phi, dphi)
    
    # Precompute 2D weights
    weight_2d = w_u[:, np.newaxis] * w_phi[np.newaxis, :]  # (70, 130)
    
    # --- Objective Functions ---
    
    def compute_ratio(c):
        """
        Computes R = ||f||_4 / ||f||_2 for a given coefficient vector c.
        c is normalized to unit L2 norm on the sphere before computation to ensure
        scale invariance, which helps global optimizers like DE.
        """
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            return np.inf
        c = c / c_norm
        
        # f shape (70, 130)
        f = np.tensordot(Y_basis, c, axes=([2], [0]))
        
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.sum(f2 * weight_2d)
        norm4_4 = np.sum(f4 * weight_2d)
        
        if norm2_sq < 1e-30 or norm4_4 < 1e-30:
            return np.inf
            
        norm2 = math.sqrt(norm2_sq)
        norm4 = norm4_4 ** 0.25
        
        return norm4 / norm2

    def compute_ratio_and_grad(c):
        """
        Computes R and its gradient for L-BFGS-B polishing.
        Assumes c is already normalized or handles normalization internally for consistency.
        """
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            return np.inf, np.zeros(n_coeffs)
        c = c / c_norm
        
        f = np.tensordot(Y_basis, c, axes=([2], [0]))
        
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.sum(f2 * weight_2d)
        norm4_4 = np.sum(f4 * weight_2d)
        
        if norm2_sq < 1e-30 or norm4_4 < 1e-30:
            return np.inf, np.zeros(n_coeffs)
            
        norm2 = math.sqrt(norm2_sq)
        norm4 = norm4_4 ** 0.25
        R = norm4 / norm2
        
        # Gradient calculation
        f3 = f2 * f
        g3 = weight_2d * f3
        g1 = weight_2d * f
        
        grad_n4 = np.tensordot(g3, Y_basis, axes=([0,1], [0,1]))
        grad_n2 = np.tensordot(g1, Y_basis, axes=([0,1], [0,1]))
        
        factor1 = 1.0 / (norm4**3 * norm2)
        factor2 = norm4 / (norm2**3)
        
        grad = factor1 * grad_n4 - factor2 * grad_n2
        
        return R, grad

    def objective_de(c):
        """
        Objective for Differential Evolution.
        Normalizes c internally to handle the sphere constraint implicitly.
        """
        return compute_ratio(c)

    def objective_lf(c):
        """
        Objective for L-BFGS-B.
        """
        val, _ = compute_ratio_and_grad(c)
        return val

    def objective_lf_jac(c):
        """
        Objective and Jacobian for L-BFGS-B.
        """
        val, grad = compute_ratio_and_grad(c)
        return val, grad

    # --- Optimization ---
    
    start_time = time.time()
    deadline = 620.0  # Leave 10s margin for safety
    
    # Initial Best: Try a few structured starts
    best_c = np.ones(n_coeffs)
    best_c /= np.linalg.norm(best_c)
    best_score = compute_ratio(best_c)
    
    # Try m=10 (index 20)
    c_m10 = np.zeros(n_coeffs)
    c_m10[k] = 1.0
    s = compute_ratio(c_m10)
    if s < best_score:
        best_score = s
        best_c = c_m10.copy()
        
    # Try m=0 (index 10)
    c_m0 = np.zeros(n_coeffs)
    c_m0[k] = 1.0
    s = compute_ratio(c_m0)
    if s < best_score:
        best_score = s
        best_c = c_m0.copy()

    # 1. Global Search with Differential Evolution
    # DE works on a box. We use [-1, 1]^21.
    # Since the objective is scale-invariant (normalizes c), the magnitude of c doesn't matter,
    # only the direction. This effectively searches the sphere.
    
    if time.time() - start_time < deadline - 120.0:
        try:
            # Use a moderate population and maxiter to balance exploration and time
            # tol and atol control convergence
            de_result = differential_evolution(
                objective_de,
                bounds=[(-1.0, 1.0)] * n_coeffs,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=True,  # DE has a built-in polish step (Nelder-Mead)
                init='sobol'
            )
            
            if de_result.success or de_result.fun < best_score:
                # de_result.x is the best point found
                c_de = de_result.x
                # Normalize and evaluate with high precision
                c_de_norm = c_de / np.linalg.norm(c_de)
                score_de = compute_ratio(c_de_norm)
                if score_de < best_score:
                    best_score = score_de
                    best_c = c_de_norm.copy()
        except Exception as e:
            pass

    # 2. Riemannian Optimization on the Unit Sphere
    # Parameterize c on the sphere using the exponential map.
    # The Riemannian gradient is the Euclidean gradient projected onto the tangent space.
    # This avoids the ill-conditioning of constrained Euclidean optimizers.
    
    def riemannian_step(c, grad, step_size):
        """
        Perform a Riemannian gradient step on the unit sphere.
        c: current point on sphere (||c|| = 1)
        grad: Euclidean gradient of the objective
        step_size: step size alpha
        Returns: new point on sphere
        """
        # Project gradient onto tangent space: grad_tan = grad - <grad, c> * c
        grad_tan = grad - np.dot(grad, c) * c
        
        # Exponential map on sphere: exp_c(v) = c * cos(||v||) + (v/||v||) * sin(||v||)
        v = -step_size * grad_tan
        v_norm = np.linalg.norm(v)
        
        if v_norm < 1e-15:
            return c.copy()
        
        # Use Taylor expansion for small angles to avoid numerical issues
        if v_norm < 1e-6:
            # exp_c(v) ≈ c + v + 0.5 * (v - <v,c>*c) ... but since v is tangent, <v,c>=0
            # So exp_c(v) ≈ c + v + 0.5*v (approximately, for very small v)
            # Actually, for small v: cos(v_norm) ≈ 1 - v_norm²/2, sin(v_norm)/v_norm ≈ 1 - v_norm²/6
            c_new = c * (1 - v_norm**2 / 2) + v * (1 - v_norm**2 / 6)
        else:
            c_new = c * math.cos(v_norm) + (v / v_norm) * math.sin(v_norm)
        
        # Renormalize to ensure we're exactly on the sphere
        c_new_norm = np.linalg.norm(c_new)
        if c_new_norm > 1e-15:
            c_new = c_new / c_new_norm
        
        return c_new
    
    def riemannian_optimize(c0, max_iter=500, tol=1e-12):
        """
        Riemannian gradient descent with line search on the unit sphere.
        """
        c = c0 / np.linalg.norm(c0)
        
        for it in range(max_iter):
            # Check time
            if time.time() - start_time > deadline - 10.0:
                break
            
            f, grad = compute_ratio_and_grad(c)
            
            # Project gradient onto tangent space
            grad_tan = grad - np.dot(grad, c) * c
            grad_norm = np.linalg.norm(grad_tan)
            
            # Convergence check
            if grad_norm < tol:
                break
            
            # Line search: find step size that satisfies Wolfe conditions
            # We want to minimize f along the Riemannian direction
            direction = -grad_tan / grad_norm  # Unit direction in tangent space
            
            def line_obj(alpha):
                c_trial = riemannian_step(c, grad, alpha * grad_norm)
                f_trial, _ = compute_ratio_and_grad(c_trial)
                return f_trial
            
            # Backtracking line search
            alpha = 1.0
            f0 = f
            grad_dir = np.dot(grad_tan, direction)  # Should be -grad_norm
            
            for ls_iter in range(30):
                c_trial = riemannian_step(c, grad, alpha)
                f_trial, _ = compute_ratio_and_grad(c_trial)
                
                # Armijo condition: f_trial <= f0 + c1 * alpha * grad_dir
                if f_trial <= f0 + 1e-4 * alpha * grad_dir:
                    break
                alpha *= 0.5
            
            c = c_trial
            f = f_trial
        
        return c, f
    
    if time.time() - start_time < deadline - 30.0:
        try:
            # Run Riemannian optimization from the current best point
            c_riem, score_riem = riemannian_optimize(best_c, max_iter=300, tol=1e-13)
            if score_riem < best_score:
                best_score = score_riem
                best_c = c_riem.copy()
        except Exception:
            pass
    
    # 2b. Final high-precision polish with L-BFGS-B (tighter tolerance)
    if time.time() - start_time < deadline - 15.0:
        try:
            res_lf = minimize(
                objective_lf,
                best_c,
                jac=objective_lf_jac,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            score_lf, _ = compute_ratio_and_grad(res_lf.x)
            if score_lf < best_score:
                best_score = score_lf
                best_c = res_lf.x.copy()
        except Exception:
            pass

    # 3. Multi-start with Riemannian + L-BFGS-B Polish if time remains
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 20.0:
        n_restarts = int(min(8, remaining_time / 12.0))
        
        for i in range(n_restarts):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Random start on the sphere
            c0 = np.random.randn(n_coeffs)
            c0 /= np.linalg.norm(c0)
            
            try:
                # First: Riemannian optimization
                c_riem, score_riem = riemannian_optimize(c0, max_iter=200, tol=1e-12)
                if score_riem < best_score:
                    best_score = score_riem
                    best_c = c_riem.copy()
                
                # Then: L-BFGS-B polish
                if time.time() - start_time < deadline - 5.0:
                    res = minimize(
                        objective_lf,
                        best_c,
                        jac=objective_lf_jac,
                        method='L-BFGS-B',
                        options={'maxiter': 2000, 'gtol': 1e-13, 'ftol': 1e-15}
                    )
                    score, _ = compute_ratio_and_grad(res.x)
                    if score < best_score:
                        best_score = score
                        best_c = res.x.copy()
            except Exception:
                pass

    # Final Validation
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        return np.ones(n_coeffs)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs == 0:
            return np.ones(n_coeffs)
        best_c = best_c / max_abs * 1.0
        
    return best_c
