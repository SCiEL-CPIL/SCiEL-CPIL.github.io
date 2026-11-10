import numpy as np
import time
import math
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize, differential_evolution

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
        # Use gammaln for numerical stability
        log_N_sq = np.log((2 * k + 1) / (4 * np.pi)) + gammaln(k + 1 - abs_m) - gammaln(k + 1 + abs_m)
        N_m[m + k] = math.sqrt(math.exp(log_N_sq))
        
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
                maxiter=300,
                popsize=25,
                tol=1e-8,
                seed=42,
                polish=False,  # We do our own polish
                init='sobol'
            )
            
            if de_result.fun < best_score:
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

    # 2. Local Polish with L-BFGS-B
    # Use the best candidate from DE (or initial) as the start
    
    if time.time() - start_time < deadline - 30.0:
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

    # 3. Multi-start Polish if time remains
    # If DE didn't find something significantly better, or if we have time,
    # try a few random starts polished with L-BFGS-B to ensure we didn't miss a basin.
    
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 20.0:
        # Estimate time per L-BFGS-B run. If we have 100s, we can do ~5-10 runs.
        # Let's do up to 10 random restarts.
        n_restarts = int(min(10, remaining_time / 10.0))
        
        rng = np.random.default_rng(123)
        
        for i in range(n_restarts):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Random start on the sphere
            c0 = rng.standard_normal(n_coeffs)
            c0 /= np.linalg.norm(c0)
            
            try:
                res = minimize(
                    objective_lf,
                    c0,
                    jac=objective_lf_jac,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14}
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
