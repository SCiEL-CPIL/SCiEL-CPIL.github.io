import numpy as np
from scipy.special import lpmv, factorial
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    
    Strategy:
    1. Precompute the basis matrix A on the specified quadrature grid.
    2. Formulate the problem as minimizing ||f||_4 subject to ||f||_2 = 1.
       Since R = ||f||_4 / ||f||_2, minimizing R is equivalent to minimizing ||f||_4
       on the unit sphere of the coefficient space (under the L2 norm induced by the basis).
       Note: The L2 norm of f is 1 if and only if c^T G c = 1, where G is the Gram matrix.
       However, the basis functions are orthonormal with respect to the surface measure,
       so G = I. Thus, the constraint is simply ||c||_2 = 1.
    3. Use Riemannian Steepest Descent on the unit sphere S^20.
       The objective is J(c) = ||A c||_4^4 (minimizing this minimizes ||f||_4).
       Gradient of J: d/dc ||Ac||_4^4 = 4 ||Ac||_4^0 * (A^T (Ac)^3) * 4? 
       Let's be precise.
       Let f = A c.
       J(c) = (sum w_i |f_i|^4)^{1/4} is the L4 norm.
       Minimizing L4 norm is equivalent to minimizing J(c) = ||f||_4^4 = sum w_i f_i^4.
       dJ/dc = 4 A^T (f^3 * w)
       
       Riemannian Gradient: Proj_c (Euclidean Gradient) = Euclidean Grad - (c^T Euclidean Grad) c
       
       Retraction: c_new = c - step * RiemGrad, then normalize to unit length.
       
    4. To escape local minima, we use multiple random starts on the sphere.
       The "flattest" harmonic likely has high symmetry. We will bias initializations
       towards symmetric patterns (e.g., only even m, or specific sign patterns) but
       primarily rely on the robustness of Riemannian SD for this smooth convex-like
       (on the sphere, locally) landscape. Actually, the problem is non-convex, but 
       Riemannian methods are very effective for ratio norms.
       
    5. We will also try a "power method" variant: 
       c_{k+1} = normalize( A^T ( (A c_k)^3 ) )
       This is the gradient ascent of the L4 norm? 
       Wait, we want to MINIMIZE L4/L2. 
       Minimizing L4 norm means making f as "flat" as possible.
       The gradient of L4^4 is proportional to A^T f^3.
       Steepest descent: c <- c - alpha * Proj(A^T f^3).
       
       Let's implement Riemannian Steepest Descent.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    k = 10
    n_m = 2 * k + 1

    # 1. Setup Quadrature Grid
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # 2. Construct Basis Matrix A
    # A[i, j] is the value of the j-th harmonic at the i-th grid point.
    A = np.zeros((n_points, n_m))

    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)

    # m = 0
    m = 0
    idx = 10
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

    # Precompute weights for objective
    # Objective: Minimize R = ||f||_4 / ||f||_2
    # Since we constrain ||c||_2 = 1, ||f||_2 = 1 (because basis is orthonormal).
    # So we minimize ||f||_4.
    # Equivalent to minimizing J(c) = sum(w_flat * (A c)^4)
    # Note: The quadrature is exact for degree 20 (4*10), so this is the true integral.
    
    def compute_f(c):
        return A @ c

    def objective_value(c):
        f = compute_f(c)
        # ||f||_4
        return np.sum(w_flat * f**4) ** 0.25

    def objective_and_riemannian_grad(c):
        """
        Returns objective value (L4 norm) and Riemannian gradient for minimizing L4^4.
        We minimize J = L4^4 to avoid the 0.25 power in gradient.
        J(c) = sum w_i f_i^4
        dJ/dc = 4 A^T (w * f^3)
        """
        f = compute_f(c)
        J = np.sum(w_flat * f**4)
        
        # Euclidean gradient of J
        # dJ/dc_j = sum_i w_i * 4 f_i^3 * A_ij
        grad_euclid = 4.0 * (A.T @ (w_flat * f**3))
        
        # Project onto tangent space of unit sphere at c
        # Proj_c(v) = v - (c^T v) c
        proj_scalar = np.dot(c, grad_euclid)
        grad_riemann = grad_euclid - proj_scalar * c
        
        # The actual R value is J**0.25
        R_val = J ** 0.25
        
        return R_val, grad_riemann

    # 3. Optimization Strategy
    # We will run Riemannian Steepest Descent with backtracking line search.
    # We will use multiple initializations.
    
    best_c = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate Initial Candidates
    candidates = []
    
    # 1. Random uniform on sphere
    for _ in range(20):
        v = rng.normal(size=n_m)
        v /= np.linalg.norm(v)
        candidates.append(v)
        
    # 2. Structured: Single modes
    for i in range(n_m):
        c = np.zeros(n_m)
        c[i] = 1.0
        candidates.append(c)
        
    # 3. Structured: Pairs of m and -m (cosine/sine combinations)
    # Y_m cos + Y_-m sin corresponds to a rotated harmonic.
    # The norm is invariant under rotation, but the shape might vary?
    # Actually, for a fixed |m|, the shape is the same up to rotation.
    # So single modes are equivalent.
    # But combinations of different |m| are distinct.
    
    # 4. Sparse combinations
    for _ in range(50):
        c = np.zeros(n_m)
        n_nonzero = rng.integers(2, 6)
        idxs = rng.choice(n_m, size=n_nonzero, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_nonzero)
        c[idxs] = signs
        norm = np.linalg.norm(c)
        if norm > 0:
            c /= norm
        candidates.append(c)
        
    # 5. Symmetric real patterns (c[m] = c[-m] implies cos terms only? No, index mapping)
    # Index i corresponds to m = i - 10.
    # m > 0 is cos, m < 0 is sin.
    # If we set c[m] = c[-m], we get cos + sin which is a rotated cos.
    # If we set c[m] = -c[-m], we get cos - sin.
    # These are all rotations.
    # The distinct shapes come from mixing different |m| values.
    
    # Sort candidates by initial objective value
    scored_candidates = []
    for c in candidates:
        val, _ = objective_and_riemannian_grad(c)
        scored_candidates.append((val, c))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 30 candidates for optimization
    top_candidates = [c for _, c in scored_candidates[:30]]
    
    for c0 in top_candidates:
        if time.time() > deadline - 5:
            break
            
        c = c0.copy()
        # Ensure unit norm
        c /= np.linalg.norm(c)
        
        # Riemannian Steepest Descent
        step_size = 0.01
        max_iter = 2000
        tol = 1e-10
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
                
            val, grad_r = objective_and_riemannian_grad(c)
            
            # Check convergence
            if np.linalg.norm(grad_r) < tol:
                break
                
            # Backtracking line search
            # We want to decrease J.
            # J(c - step * grad_r) < J(c)
            # Since grad_r is Riemannian gradient, for small step, this holds.
            # Retraction: c_new = normalize(c - step * grad_r)
            
            current_J = val**4  # Use J for comparison
            
            alpha = step_size
            improved = False
            for _ in range(20):
                c_new = c - alpha * grad_r
                norm_new = np.linalg.norm(c_new)
                if norm_new < 1e-12:
                    alpha *= 0.5
                    continue
                c_new /= norm_new
                
                val_new, _ = objective_and_riemannian_grad(c_new)
                J_new = val_new**4
                
                if J_new < current_J * (1 - 1e-4): # Armijo condition simplified
                    c = c_new
                    improved = True
                    break
                else:
                    alpha *= 0.5
            
            if not improved:
                # If we couldn't improve, try a smaller fixed step or break
                # If grad is tiny, we are at a minimum.
                if np.linalg.norm(grad_r) < 1e-8:
                    break
                # Otherwise, force a small step
                c_new = c - 1e-4 * grad_r
                c_new /= np.linalg.norm(c_new)
                c = c_new
                
        val_final, _ = objective_and_riemannian_grad(c)
        if val_final < best_val:
            best_val = val_final
            best_c = c.copy()
            
        # If we found a very good value, we can stop early?
        # The theoretical minimum for L4/L2 on sphere is 1 (constant function).
        # But degree 10 cannot be constant.
        # The "flattest" degree 10 harmonic will have R > 1.
        # For Legendre P_10, R is known.
        # If best_val is close to the theoretical lower bound, we might stop.
        # But we don't know the exact bound.
        
    # Fallback if no optimization ran or failed
    if best_c is None:
        best_c = np.zeros(n_m)
        best_c[10] = 1.0 # m=0
        
    # Final validity check
    if not np.all(np.isfinite(best_c)):
        return np.ones(n_m)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)
        
    return best_c
