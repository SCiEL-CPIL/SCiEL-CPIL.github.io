import numpy as np
import time
from scipy.optimize import minimize, root

_DEADLINE = time.time() + 205

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
H0_c = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H4_c = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
H8_c = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
H12_c = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

# a12 is linear in a: a12 = c12_0*a0 + c12_4*a4 + c12_8*a8
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

# Precompute moment matrices for fast evaluation
# Mx[n] = sum_{i,j} p_i p_j I[i+j+n] for n=0,1
# Mxi[n] = sum_{i,j} q_i q_j I[i+j+n] for n=0,1
# where q_k = p_k * (-1)^k

# For position: Mx[n] = p^T Kx[n] p where Kx[n]_{ij} = I[i+j+n]
# For momentum: Mxi[n] = q^T Kxi[n] q where Kxi[n]_{ij} = I[i+j+n]
# But q = D p where D = diag([1, -1, 1, -1, 1, -1, 1])
# So Mxi[n] = p^T D Kxi[n] D p

D = np.diag(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]))

# Precompute Kx[0], Kx[1], Kxi[0], Kxi[1] (7x7 matrices)
Kx = np.zeros((2, 7, 7))
Kxi = np.zeros((2, 7, 7))
for n in range(2):
    for i in range(7):
        for j in range(7):
            Kx[n, i, j] = I[i+j+n]
            Kxi[n, i, j] = I[i+j+n]

# Effective matrices: Mx[n] = p^T Kx[n] p, Mxi[n] = p^T (D Kxi[n] D) p
Kxi_eff = np.zeros((2, 7, 7))
for n in range(2):
    Kxi_eff[n] = D @ Kxi[n] @ D

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    p = _get_p(a)
    
    # Compute moments using precomputed matrices
    Mx0 = p @ Kx[0] @ p
    Mx1 = p @ Kx[1] @ p
    Mxi0 = p @ Kxi_eff[0] @ p
    Mxi1 = p @ Kxi_eff[1] @ p
    
    if Mx0 <= 1e-14 or Mxi0 <= 1e-14:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-14 or vxi <= 1e-14:
        return 1e18, np.zeros(3)
        
    prod = vx * vxi
    if prod <= 0:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(prod)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation using matrix calculus
    # dMx0 = 2 * p^T Kx[0] da = 2 * (Kx[0] p)^T da
    # dMx1 = 2 * (Kx[1] p)^T da
    # dMxi0 = 2 * (Kxi_eff[0] p)^T da
    # dMxi1 = 2 * (Kxi_eff[1] p)^T da
    
    # dM_m (for m=0,1 in x and xi) = 2 * (K_m p)^T A_mat (since da = A_mat da_coeff)
    # Actually: p = A_mat @ a, so dp = A_mat @ da
    # dM = 2 * p^T K dp = 2 * (K p)^T A_mat da
    
    Kx0p = Kx[0] @ p
    Kx1p = Kx[1] @ p
    Kxi0p = Kxi_eff[0] @ p
    Kxi1p = Kxi_eff[1] @ p
    
    dMx0 = 2.0 * (A_mat.T @ Kx0p)
    dMx1 = 2.0 * (A_mat.T @ Kx1p)
    dMxi0 = 2.0 * (A_mat.T @ Kxi0p)
    dMxi1 = 2.0 * (A_mat.T @ Kxi1p)
    
    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)
    
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) * inv_Mx0_sq
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) * inv_Mxi0_sq
        
    factor = 0.5 / bound
    grad = factor * (vxi * dvx + vx * dvxi)
    
    if not np.all(np.isfinite(grad)):
        return bound, np.zeros(3)
        
    return bound, grad

def _stationary_equations(a):
    """
    Returns the gradient of F = Vx * Vxi.
    """
    p = _get_p(a)
    
    Mx0 = p @ Kx[0] @ p
    Mx1 = p @ Kx[1] @ p
    Mxi0 = p @ Kxi_eff[0] @ p
    Mxi1 = p @ Kxi_eff[1] @ p
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    Kx0p = Kx[0] @ p
    Kx1p = Kx[1] @ p
    Kxi0p = Kxi_eff[0] @ p
    Kxi1p = Kxi_eff[1] @ p
    
    dMx0 = 2.0 * (A_mat.T @ Kx0p)
    dMx1 = 2.0 * (A_mat.T @ Kx1p)
    dMxi0 = 2.0 * (A_mat.T @ Kxi0p)
    dMxi1 = 2.0 * (A_mat.T @ Kxi1p)
    
    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)
    
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) * inv_Mx0_sq
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) * inv_Mxi0_sq
        
    grad_F = vxi * dvx + vx * dvxi
    
    return grad_F

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 5
    
    # Generate initial guesses
    starts = [seed_params.copy()]
    
    # Grid-based starts
    a0_range = np.linspace(-2.5, 0.5, 6)
    a4_range = np.linspace(-0.3, 0.3, 6)
    a8_range = np.linspace(-0.02, 0.02, 5)
    
    for a0 in a0_range:
        for a4 in a4_range:
            for a8 in a8_range:
                starts.append(np.array([a0, a4, a8]))
                
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(80):
        a0 = rng.uniform(-4, 2)
        a4 = rng.uniform(-0.8, 0.8)
        a8 = rng.uniform(-0.1, 0.1)
        starts.append(np.array([a0, a4, a8]))
        
    # Solve stationary point equations for each start
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            res = root(_stationary_equations, start, method='hybr', tol=1e-12)
            
            if res.success:
                a_sol = res.x
                val, _ = _compute_bound_and_grad(a_sol)
                
                if np.isfinite(val) and val < best_val:
                    if time.time() < _DEADLINE - deadline_margin:
                        res_min = minimize(_compute_bound_and_grad, a_sol, method='L-BFGS-B', jac=True, options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-12})
                        if res_min.success and res_min.fun < val:
                            val = res_min.fun
                            a_sol = res_min.x
                            
                    if val < best_val:
                        best_val = val
                        best_params = a_sol.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res_min = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res_min.success and res_min.fun < best_val:
                best_val = res_min.fun
                best_params = res_min.x
        except Exception:
            pass
        
        if time.time() < _DEADLINE - deadline_margin:
            try:
                res_min2 = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B', jac=True, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res_min2.success and res_min2.fun < best_val:
                    best_val = res_min2.fun
                    best_params = res_min2.x
            except Exception:
                pass
        
        # Perturbation-based polish
        if time.time() < _DEADLINE - 2 * deadline_margin:
            rng_polish = np.random.default_rng(123)
            for scale in [1e-6, 1e-8, 1e-10]:
                if time.time() > _DEADLINE - deadline_margin:
                    break
                for _ in range(5):
                    if time.time() > _DEADLINE - deadline_margin:
                        break
                    try:
                        perturbation = rng_polish.normal(0, scale, size=3)
                        start_polish = best_params + perturbation
                        res_polish = minimize(_compute_bound_and_grad, start_polish, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
                        if res_polish.success and res_polish.fun < best_val:
                            best_val = res_polish.fun
                            best_params = res_polish.x
                    except Exception:
                        pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
