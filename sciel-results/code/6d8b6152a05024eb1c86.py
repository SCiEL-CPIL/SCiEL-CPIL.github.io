"""
Strategy:
1. Formulate the C4 upper bound exactly using the spectral theory of Laguerre polynomials.
   The bound is the smallest eigenvalue $\lambda$ of the problem:
   $\int_0^\infty P(x)^2 e^{-x} dx / \int_0^\infty (P(x)/x)^2 e^{-x} dx \ge \lambda$
   subject to $P(r_j) = P'(r_j) = 0$.
   Using the Laguerre basis, this is a generalized eigenvalue problem $A c = \lambda B c$
   where $A$ is the Gram matrix $\langle L_i, L_j \rangle$ and $B$ is the moment matrix
   $\langle L_i/x, L_j/x \rangle$.
   We restrict to the null space of the constraint matrix $C$ (double roots).
   The bound is the smallest eigenvalue of $N^T B N$ relative to $N^T A N$ (or just
   the smallest eigenvalue if we normalize $N^T A N = I$).
   
   Actually, for the uncertainty principle, the constant is related to the inverse of the
   maximum of $\frac{\int |f|^2}{\int |x f|^2}$ or similar. In Cohn-Goncalves, the bound
   on the constant $C$ is given by the smallest eigenvalue of the operator $x$ restricted
   to the orthogonal complement of the polynomials vanishing at the roots.
   Wait, the task says "upper bound on C4". In the Fourier sign uncertainty, the constant
   is bounded by a function of the roots. The "tuning" minimizes this bound.
   
   Let's use the standard result: The optimal constant is the smallest eigenvalue of the
   Jacobi matrix $J$ (which represents multiplication by $x$) restricted to the subspace
   orthogonal to the constraints.
   
   The Jacobi matrix $J$ for Laguerre polynomials $L_n(x)$ with weight $e^{-x}$ is:
   $J_{i,i} = 2i + 1$
   $J_{i,i+1} = \sqrt{i+1}$
   $J_{i+1,i} = \sqrt{i+1}$
   
   We construct a basis $V$ for the null space of the constraint matrix $C$.
   The bound is the smallest eigenvalue of $V^T J V$.
   
2. Optimize the roots $r_1, \dots, r_{12}$ to minimize this smallest eigenvalue.
   We use L-BFGS-B with gradients computed via the eigenvalue sensitivity theorem.
   $\frac{\partial \lambda}{\partial r_k} = v^T \frac{\partial (V^T J V)}{\partial r_k} v$
   where $v$ is the eigenvector corresponding to $\lambda$.
   Since $J$ does not depend on $r_k$, only $V$ does.
   $\frac{\partial (V^T J V)}{\partial r_k} = \frac{\partial V^T}{\partial r_k} J V + V^T J \frac{\partial V}{\partial r_k}$
   
   This requires computing $\frac{\partial V}{\partial r_k}$.
   $V$ is the null space of $C(r)$.
   $C(r) V = 0 \implies \frac{\partial C}{\partial r_k} V + C \frac{\partial V}{\partial r_k} = 0$.
   $C \frac{\partial V}{\partial r_k} = - \frac{\partial C}{\partial r_k} V$.
   
   This is a linear system for $\frac{\partial V}{\partial r_k}$, but $C$ is singular.
   We can solve for the component of $\frac{\partial V}{\partial r_k}$ in the range of $C$.
   Alternatively, we can use automatic differentiation or finite differences for the gradient
   if analytic is too complex, but finite differences might be slow.
   
   Given the 600s limit, let's try a robust approach:
   - Compute the bound exactly.
   - Use Nelder-Mead or L-BFGS-B with numerical gradients (complex-step or central diff)
     if analytic is tricky. 12 variables, 24 constraints.
     Numerical gradient: 24 evaluations per gradient step.
     Each evaluation: SVD of 24x24 matrix + Eigen of 12x12 matrix. Very fast.
     1000 steps * 24 evals = 24000 evals. Each eval ~1ms. Total ~24s. Feasible.
     
   Let's implement:
   1. `compute_bound(roots)`: Returns smallest eigenvalue of $V^T J V$.
   2. `gradient(roots)`: Numerical gradient using central differences.
   3. Optimization loop with L-BFGS-B.
"""

import numpy as np
import time
from scipy.optimize import minimize

def _build_constraint_matrix(roots):
    """
    Build the 2m x (2m) constraint matrix C.
    Rows: P(r_j) = 0, P'(r_j) = 0 for j=1..m.
    Columns: Coefficients c_0..c_{2m-1} of P(x) = sum c_k L_k(x).
    """
    m = 12
    max_deg = 2 * m - 1
    C = np.zeros((2 * m, max_deg + 1))
    
    # Precompute Laguerre values and derivatives for efficiency?
    # For now, direct evaluation.
    
    for j, r in enumerate(roots):
        # P(r) = 0
        for k in range(max_deg + 1):
            # L_k(r)
            # Use recurrence or scipy
            # scipy.special.eval_genlaguerre(k, 0, r)
            C[2 * j, k] = _eval_laguerre(k, 0, r)
            
        # P'(r) = 0
        # L_k'(x) = -L_{k-1}^{(1)}(x)
        for k in range(1, max_deg + 1):
            C[2 * j + 1, k] = -_eval_laguerre(k - 1, 1, r)
        # L_0' = 0, so C[2*j+1, 0] = 0
        
    return C

def _eval_laguerre(n, alpha, x):
    """
    Evaluate generalized Laguerre polynomial L_n^(alpha)(x).
    Uses scipy.special if available, else recurrence.
    """
    try:
        from scipy.special import eval_genlaguerre
        return eval_genlaguerre(n, alpha, x)
    except ImportError:
        # Fallback recurrence
        if n == 0:
            return 1.0
        if n == 1:
            return alpha + 1 - x
        L_nm1 = 1.0
        L_n = alpha + 1 - x
        for i in range(2, n + 1):
            L_np1 = ((2 * i - 1 + alpha - x) * L_n - (i - 1 + alpha) * L_nm1) / i
            L_nm1 = L_n
            L_n = L_np1
        return L_n

def _compute_bound(roots):
    """
    Compute the smallest eigenvalue of V^T J V.
    V is the null space of the constraint matrix C.
    """
    m = 12
    max_deg = 2 * m - 1
    dim = max_deg + 1
    
    C = _build_constraint_matrix(roots)
    
    # Find null space of C
    # C is 2m x dim. Rank should be 2m. Null space dimension = dim - 2m = 1.
    # Wait, dim = 24. 2m = 24. Rank 24. Null space dimension 1.
    # If null space dimension is 1, V is a vector.
    # Then V^T J V is a scalar. The bound is that scalar.
    
    U, s, Vt = np.linalg.svd(C)
    # The last row of Vt is the null vector (assuming rank 24)
    v = Vt[-1, :]
    
    # Normalize v? The bound is scale invariant for the ratio, but here we are
    # looking at the eigenvalue of the operator x restricted to the subspace.
    # If the subspace is 1D, the "eigenvalue" is just <v, J v> / <v, v>.
    
    # Construct J (Jacobi matrix for Laguerre)
    J = np.zeros((dim, dim))
    for i in range(dim):
        J[i, i] = 2 * i + 1
        if i < dim - 1:
            J[i, i + 1] = np.sqrt(i + 1)
            J[i + 1, i] = np.sqrt(i + 1)
    
    # Rayleigh quotient
    num = v @ J @ v
    den = v @ v
    
    if den < 1e-12:
        return np.inf
        
    bound = num / den
    return bound

def _gradient(roots, eps=1e-6):
    """
    Compute gradient of the bound with respect to roots using central differences.
    """
    grad = np.zeros_like(roots)
    f0 = _compute_bound(roots)
    
    for i in range(len(roots)):
        roots_plus = roots.copy()
        roots_minus = roots.copy()
        roots_plus[i] += eps
        roots_minus[i] -= eps
        
        # Ensure positivity and ordering if needed, but small eps should be fine
        # if initial roots are well-separated.
        
        fp = _compute_bound(roots_plus)
        fm = _compute_bound(roots_minus)
        
        if np.isinf(fp) or np.isinf(fm):
            grad[i] = 0.0
        else:
            grad[i] = (fp - fm) / (2 * eps)
            
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 590.0
    
    m = 12
    # Initial guess: spread out roots
    # The seed was [4, 8, ..., 48]. Let's try a wider spread or geometric.
    # Roots must be positive.
    
    # Start with the seed-like configuration
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Evaluate initial
    f0 = _compute_bound(x0)
    best_roots = x0.copy()
    best_score = f0
    
    # Optimization
    # Use L-BFGS-B with numerical gradients
    # Bounds: roots must be positive and increasing? 
    # The problem doesn't strictly require increasing, but distinct.
    # Let's enforce x > 0.
    
    bounds = [(1e-6, None) for _ in range(m)]
    
    # To handle the distinctness constraint, we can optimize in log-space or
    # just hope L-BFGS-B stays in the valid region.
    # If roots collide, the constraint matrix becomes rank deficient,
    # null space dimension > 1, and the "smallest eigenvalue" logic needs adjustment.
    # For robustness, we'll just return inf if rank < 2m.
    
    def objective(x):
        if time.time() > deadline:
            return np.inf
        # Check validity
        if np.any(x <= 0):
            return np.inf
        # Check distinctness (loose)
        if np.any(np.diff(np.sort(x)) < 1e-5):
            return np.inf
        return _compute_bound(x)
        
    def jac(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        return _gradient(x)
        
    # Multi-start strategy
    starts = [
        x0,
        x0 * 1.5,
        x0 * 0.5,
        np.array([i**2 for i in range(1, m+1)]),
        np.array([2**i for i in range(m)]),
    ]
    
    for x_start in starts:
        if time.time() > deadline - 60:
            break
            
        # Ensure start is valid
        if np.any(x_start <= 0) or np.any(np.diff(np.sort(x_start)) < 1e-5):
            continue
            
        try:
            res = minimize(
                objective, 
                x_start, 
                jac=jac, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_roots = res.x.copy()
                
        except Exception:
            continue
            
    # Final check
    if time.time() < deadline - 10:
        # Polish with Nelder-Mead if L-BFGS-B got stuck?
        # Nelder-Mead doesn't use gradients, might escape local minima.
        try:
            res_nm = minimize(
                objective, 
                best_roots, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_roots = res_nm.x.copy()
        except Exception:
            pass

    # Ensure output is a list of floats
    return best_roots.tolist()

# EVOLVE-BLOCK-END
