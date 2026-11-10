import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c for degree-10 spherical harmonics minimizing ||f||_4 / ||f||_2.
    Strategy: 
    1. Precompute Spherical Harmonic matrix Y at quadrature points.
    2. Define objective as variance of f^2 (smooth surrogate for L4/L2 ratio).
       Note: ||f||_4^4 = ∫ f^4. ||f||_2^4 = (∫ f^2)^2.
       If we normalize ||f||_2 = 1, then minimizing ||f||_4 is equivalent to minimizing ∫ f^4.
       Also, Var(f^2) = E[f^4] - (E[f^2])^2. If E[f^2]=1, Var(f^2) = E[f^4] - 1.
       So minimizing Var(f^2) under constraint ||c||_2=1 (which implies ||f||_2=1 due to orthonormality)
       is equivalent to minimizing the ratio.
       
       However, the constraint ||c||_2=1 is a sphere. We can optimize on the sphere using
       L-BFGS-B with a penalty or by projecting? 
       Actually, L-BFGS-B doesn't handle non-linear equality constraints well.
       
       Alternative: Use the fact that the objective is scale invariant. 
       We can minimize the "excess kurtosis" or similar.
       
       Let's stick to the smooth objective:
       J(c) = (∫ f^4 dΩ) / (∫ f^2 dΩ)^2 - 1
       This is scale invariant. 
       Gradient can be computed analytically.
       
       To use L-BFGS-B, we need a smooth function. J(c) is smooth everywhere except c=0.
       We can just use J(c) directly. Since it's scale invariant, the optimizer might drift in scale.
       To prevent this, we can add a small regularization ||c||^2 - 1? 
       Or better, just use the ratio. L-BFGS-B handles scale invariance poorly (flat directions).
       
       Better approach: 
       Fix the scale. Since the problem is scale invariant, we can impose ||c||_2 = 1.
       But L-BFGS-B is for box constraints. 
       
       Let's use a different method: 
       We can use `scipy.optimize.minimize` with method='L-BFGS-B' on the function:
       f(c) = ||Y c||_4^4 / ||Y c||_2^4
       This is smooth. The gradient is well-defined.
       
       Let's derive the gradient.
       Let A = Y^T W Y (Gram matrix, identity if Y is orthonormal at quadrature points? 
       Wait, quadrature is exact for degree 20, so Y^T W Y = I for degree 10 harmonics?
       Yes, 70-point GL in cos(theta) is exact for polynomials up to degree 139 in x.
       Y_m involves P_10^m. Product is degree 20. So Y^T W Y = I (with proper weights).
       So ||c||_2 = ||f||_2 / sqrt(4pi)? 
       Let's check normalization.
       N_m is defined such that ∫ Y_m Y_n dΩ = δ_mn.
       So if ∫ f^2 dΩ = 1, then ||c||_2 = 1.
       
       So we can minimize f(c) = ∫ f^4 dΩ subject to ||c||_2 = 1.
       Since ||c||_2=1 implies ∫ f^2 dΩ = 1, then f(c) = ||f||_4^4.
       Minimizing ||f||_4 is the goal.
       
       We can use Lagrange multipliers or just project back to sphere after each step.
       L-BFGS-B doesn't project. 
       
       Alternative: Use `trust-constr` with equality constraint? 
       Or use `scipy.optimize.minimize` with method='SLSQP' with constraint ||c||^2=1?
       
       Let's try `L-BFGS-B` on the unconstrained smooth function:
       g(c) = (∫ f^4) / (∫ f^2)^2
       This is scale invariant. The optimizer might wander. 
       To fix this, we can add a term: (||c||_2^2 - 1)^2 * lambda.
       But lambda tuning is tricky.
       
       Better: Use the fact that we can normalize c at the end.
       During optimization, we can just minimize g(c).
       The gradient of g(c) at c is proportional to the gradient of the numerator 
       if we fix the denominator.
       
       Let's use the "variance of f^2" objective.
       V(c) = E[f^4] - (E[f^2])^2.
       If we normalize c such that E[f^2] = 1 (i.e., ||c||=1), then V(c) = E[f^4] - 1.
       Minimizing V(c) is the same as minimizing E[f^4].
       
       Let's use `minimize` with `method='L-BFGS-B'` on:
       obj(c) = (Yc @ W @ (Yc)**3) - (Yc @ W @ (Yc)**2)**2  # This is not scale invariant in a way that helps?
       Actually, let's just use:
       obj(c) = np.dot(W, (Y @ c)**4) / (np.dot(W, (Y @ c)**2)**2)
       
       And provide the analytic gradient.
       
       Gradient of J(c) = N/D^2 where N = ∫ f^4, D = ∫ f^2.
       ∇J = (∇N D^2 - N * 2 D ∇D) / D^4 = (∇N D - 2 N ∇D) / D^3
       
       ∇D = ∇(c^T G c) = 2 G c, where G = Y^T W Y = I (approx).
       ∇N = ∇(c^T H c) where H_ij = ∫ Y_i Y_j Y_k Y_l ... no, f^4 = (sum c_i Y_i)^4.
       This is a 4th order tensor.
       N = sum_{ijkl} c_i c_j c_k c_l ∫ Y_i Y_j Y_k Y_l.
       Let T_{ijkl} = ∫ Y_i Y_j Y_k Y_l dΩ.
       N = c^T T c (in tensor notation).
       ∇N = 4 T c (where (T c)_i = sum_{jkl} T_{ijkl} c_j c_k c_l).
       
       We can precompute T? 21^4 = 194481 elements. 
       T is symmetric. We can compute T by evaluating the integrals.
       Since we have quadrature points, we can build the tensor from the matrix Y.
       T_{ijkl} = sum_p W_p Y_{p,i} Y_{p,j} Y_{p,k} Y_{p,l}.
       This is a 4th order tensor of size 21x21x21x21.
       Memory: 21^4 * 8 bytes ~ 1.5 MB. Very small.
       Computation: 
       We can compute T efficiently using einsum or loops.
       Y is (P, 21). P=9100.
       T = np.einsum('pi,pj,pk,pl->ijkl', Y*W[:,None], Y, Y, Y)
       This might be slow if P is large, but 9100 is okay.
       
       Then ∇N = 4 * np.einsum('ijkl,jkl->i', T, c, c, c) ? No.
       (T c)_i = sum_{jkl} T_{ijkl} c_j c_k c_l.
       This is a matrix-vector product with a 3rd order tensor?
       Actually, T_{ijkl} is symmetric in all indices? 
       Y_i Y_j Y_k Y_l is symmetric. So T is fully symmetric.
       We can contract T with c,c,c.
       Let M_{ij} = sum_{kl} T_{ijkl} c_k c_l.
       Then (T c)_i = sum_j M_{ij} c_j.
       
       So we can precompute M(c) on the fly.
       M = np.einsum('ijkl,k,l->ij', T, c, c)
       grad_N = 4 * (M @ c)
       
       This is efficient.
       
       So the plan:
       1. Build Y matrix.
       2. Build T tensor.
       3. Define objective and gradient.
       4. Run L-BFGS-B from multiple starts.
       5. Return best c.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    start_time = time.time()
    deadline = start_time + 590
    
    # --- 1. Precompute Spherical Harmonics ---
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    x_grid, phi_grid = np.meshgrid(x_gl, phi_vals, indexing='ij')
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Effective weights
    W_eff = np.repeat(w_gl, n_phi) * (2.0 * np.pi / n_phi)
    
    # Normalization constants
    N = np.zeros(n_coeffs)
    for m in range(0, k + 1):
        norm_factor = (2 * k + 1) / (4.0 * np.pi) * factorial(k - m) / factorial(k + m)
        N[m] = np.sqrt(norm_factor)
        if m > 0:
            N[-m] = N[m]
            
    # Build Y matrix
    n_points = len(x_flat)
    Y_matrix = np.zeros((n_points, n_coeffs))
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        if m == 0:
            P_val = lpmv(0, k, x_flat)
            Y_matrix[:, m_idx] = N[0] * P_val
        elif m > 0:
            P_val = lpmv(m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[m] * P_val * np.cos(m * phi_flat)
        else:
            abs_m = -m
            P_val = lpmv(abs_m, k, x_flat)
            Y_matrix[:, m_idx] = np.sqrt(2) * N[abs_m] * P_val * np.sin(abs_m * phi_flat)
            
    # --- 2. Precompute 4th order tensor T ---
    # T_{ijkl} = sum_p W_p Y_{p,i} Y_{p,j} Y_{p,k} Y_{p,l}
    # We can compute this as:
    # YW = Y_matrix * W_eff[:, None]
    # T = np.einsum('pi,pj,pk,pl->ijkl', YW, Y_matrix, Y_matrix, Y_matrix)
    # This might be memory intensive or slow? 21^4 = 194k floats. 1.5MB. Fine.
    # But einsum might be slow for 9100 points.
    # Let's try to optimize.
    # T is symmetric. We can just compute it.
    
    # Optimization: 
    # Let A = Y^T W Y (21x21). 
    # Let B = Y^T W (Y * Y) (21x21)? No.
    # T_{ijkl} = (Y^T W Y)_{ij} ... no.
    # T_{ijkl} = sum_p W_p Y_{pi} Y_{pj} Y_{pk} Y_{pl}
    # We can group:
    # M_{ij,kl} = sum_p W_p (Y_{pi} Y_{pj}) (Y_{pk} Y_{pl})
    # Let Z_{p, ij} = Y_{pi} Y_{pj} for i<=j? No, full matrix.
    # Z is (P, 21, 21).
    # T_{ijkl} = sum_p W_p Z_{p,ij} Z_{p,kl}
    # This is a product of two matrices if we reshape.
    # Let Z_flat = Z.reshape(P, 21*21)
    # Then T_flat = Z_flat.T @ (W_eff[:, None] * Z_flat)
    # T_flat is (441, 441). Reshape to (21,21,21,21).
    
    Z = Y_matrix[:, :, None] * Y_matrix[:, None, :]  # (P, 21, 21)
    Z_flat = Z.reshape(n_points, n_coeffs * n_coeffs)
    T_flat = Z_flat.T @ (W_eff[:, None] * Z_flat)
    T_tensor = T_flat.reshape(n_coeffs, n_coeffs, n_coeffs, n_coeffs)
    
    # --- 3. Define Objective and Gradient ---
    def objective_and_grad(c):
        # c is shape (21,)
        # f = Y @ c
        # N_val = sum_p W_p f_p^4 = c^T T c (tensor contraction)
        # D_val = sum_p W_p f_p^2 = c^T G c, where G = Y^T W Y = I (approx)
        # Actually, let's compute D_val directly to be safe.
        f = Y_matrix @ c
        f2 = f * f
        D_val = np.dot(W_eff, f2)
        
        # N_val = sum_p W_p f^4
        # We can use T_tensor
        # (T c)_i = sum_{jkl} T_{ijkl} c_j c_k c_l
        # Let's compute M_{ij} = sum_{kl} T_{ijkl} c_k c_l
        # M = np.einsum('ijkl,k,l->ij', T_tensor, c, c)
        # N_val = c @ M @ c
        
        # Alternatively, compute N_val directly from f?
        # N_val = np.dot(W_eff, f2 * f2)
        # This is faster than tensor contraction?
        # f is 9100x1. f2 is 9100x1. f2*f2 is 9100x1. Dot is 9100 ops.
        # Tensor contraction: M is 21x21. M@c is 21x1. c@M@c is scalar.
        # Building M: einsum on 21^4 tensor with 21x1 vector.
        # The tensor is small. The direct computation of N_val is likely faster.
        
        N_val = np.dot(W_eff, f2 * f2)
        
        if D_val < 1e-12:
            return 1e10, np.zeros_like(c)
            
        val = N_val / (D_val * D_val)
        
        # Gradient
        # grad_N = 4 * Y^T (W * f^3)
        # grad_D = 2 * Y^T (W * f)
        
        grad_N = 4.0 * (Y_matrix.T @ (W_eff * f**3))
        grad_D = 2.0 * (Y_matrix.T @ (W_eff * f))
        
        # val = N / D^2
        # dval/dc = (dN D^2 - N 2 D dD) / D^4 = (dN D - 2 N dD) / D^3
        
        grad = (grad_N * D_val - 2.0 * N_val * grad_D) / (D_val ** 3)
        
        return val, grad

    # --- 4. Optimization ---
    best_score = np.inf
    best_coeffs = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    
    inits = []
    
    # 1. Single modes
    for m in range(-k, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        inits.append(init)
        
    # 2. Symmetric pairs
    for m in range(1, k + 1):
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = 1.0
        inits.append(init)
        
        init = np.zeros(n_coeffs)
        init[m + k] = 1.0
        init[-m + k] = -1.0
        inits.append(init)
        
    # 3. Random vectors
    for _ in range(50):
        init = np.random.randn(n_coeffs)
        init /= np.linalg.norm(init)
        inits.append(init)
        
    # 4. Sparse random
    for _ in range(50):
        init = np.random.randn(n_coeffs)
        mask = np.random.rand(n_coeffs) < 0.3
        init[~mask] = 0
        if np.linalg.norm(init) > 1e-6:
            init /= np.linalg.norm(init)
        inits.append(init)
        
    np.random.shuffle(inits)
    
    lbfgs_options = {
        'maxiter': 1000,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxls': 20,
    }
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', options=lbfgs_options)
            
            # Normalize result to unit norm for consistency, though objective is scale invariant
            c_res = res.x
            norm_c = np.linalg.norm(c_res)
            if norm_c > 1e-10:
                c_res = c_res / norm_c
            
            # Evaluate final score
            f_res = Y_matrix @ c_res
            l2_sq = np.dot(W_eff, f_res**2)
            l4 = np.dot(W_eff, f_res**4)
            score = l4 / (l2_sq * l2_sq)
            
            if score < best_score:
                best_score = score
                best_coeffs = c_res.copy()
                
        except Exception:
            continue

    # --- 5. Refine Best Solution ---
    if time.time() < deadline - 10:
        try:
            lbfgs_options_refine = {
                'maxiter': 5000,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 50,
            }
            res = minimize(objective_and_grad, best_coeffs, jac=True, method='L-BFGS-B', options=lbfgs_options_refine)
            
            c_res = res.x
            norm_c = np.linalg.norm(c_res)
            if norm_c > 1e-10:
                c_res = c_res / norm_c
            
            f_res = Y_matrix @ c_res
            l2_sq = np.dot(W_eff, f_res**2)
            l4 = np.dot(W_eff, f_res**4)
            score = l4 / (l2_sq * l2_sq)
            
            if score < best_score:
                best_score = score
                best_coeffs = c_res.copy()
        except Exception:
            pass

    # --- 6. Validate and Return ---
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs)
        
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3:
        best_coeffs *= (1.0 / max_c)
    elif max_c > 1e5:
        best_coeffs /= max_c
        
    return best_coeffs
