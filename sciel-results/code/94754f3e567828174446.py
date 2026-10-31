import numpy as np
import time
from scipy.optimize import minimize, root
from scipy.linalg import eigh

_DEADLINE = time.time() + 200

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
# I[0] = sqrt(pi)
# I[n] = I[n-1] * (2n-1)/(2n)
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# p_k is the coefficient of x^{2k} in P(x)
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# a12 = -(a0 + 3*a4 + 105*a8) / 945

# Matrix A such that p = A * a, where a = [a0, a4, a8]
# p has 7 components (k=0..6)
# Row k of A corresponds to coefficient of x^{2k}

# H0 coeffs: [1, 0, 0, 0, 0, 0, 0]
# H4 coeffs: [3, -6, 1, 0, 0, 0, 0] (order x^0, x^2, x^4)
# H8 coeffs: [105, -420, 210, -28, 1, 0, 0]
# H12 coeffs: [945, -7560, 13860, -6930, 1155, -66, 1]

H0_c = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H4_c = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
H8_c = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
H12_c = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

# a12 is linear in a: a12 = c12_0*a0 + c12_4*a4 + c12_8*a8
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# p = a0*H0_c + a4*H4_c + a8*H8_c + a12*H12_c
# p = a0*(H0_c + c12_0*H12_c) + a4*(H4_c + c12_4*H12_c) + a8*(H8_c + c12_8*H12_c)

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    p = _get_p(a)
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # Compute moments M_x[n] and M_xi[n] for n=0,1
    # M[n] = sum_{i,j} p_i p_j I[i+j+n]
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            pj = p[j]
            qj = q[j]
            idx = i + j
            Mx0 += pi * pj * I[idx]
            Mx1 += pi * pj * I[idx + 1]
            Mxi0 += qi * qj * I[idx]
            Mxi1 += qi * qj * I[idx + 1]
            
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
        
    # Gradient computation
    # Let S[n]_i = sum_j p_j I[i+j+n]
    Sx0 = np.zeros(7)
    Sx1 = np.zeros(7)
    Sxi0 = np.zeros(7)
    Sxi1 = np.zeros(7)
    
    for i in range(7):
        Sx0[i] = np.dot(p, I[i:i+7])
        Sx1[i] = np.dot(p, I[i+1:i+8])
        Sxi0[i] = np.dot(q, I[i:i+7])
        Sxi1[i] = np.dot(q, I[i+1:i+8])
            
    # dM[n]_m = 2 * sum_i A[i,m] S[n, i]
    # Vectorized: dM = 2 * A_mat.T @ S
    dMx0 = 2.0 * A_mat.T @ Sx0
    dMx1 = 2.0 * A_mat.T @ Sx1
    dMxi0 = 2.0 * A_mat.T @ Sxi0
    dMxi1 = 2.0 * A_mat.T @ Sxi1
    
    # dvx_m = (dMx1[m]*Mx0 - Mx1*dMx0[m]) / Mx0^2
    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)
    
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) * inv_Mx0_sq
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) * inv_Mxi0_sq
        
    # d(bound) = (1/(2*bound)) * (vxi*dvx + vx*dvxi)
    factor = 0.5 / bound
    grad = factor * (vxi * dvx + vx * dvxi)
    
    if not np.all(np.isfinite(grad)):
        return bound, np.zeros(3)
        
    return bound, grad

def _stationary_equations(a):
    """
    Returns the gradient of F = Vx * Vxi.
    We want to find roots of this system.
    """
    p = _get_p(a)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # Vectorized moment computation
    Mx0 = np.dot(p, np.dot(p, I))
    Mx1 = np.dot(p, np.dot(p, I[1:]))
    Mxi0 = np.dot(q, np.dot(q, I))
    Mxi1 = np.dot(q, np.dot(q, I[1:]))
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    # Sx[n, i] = sum_j p[j] * I[i+j+n]
    # This is a Toeplitz-like structure
    Sx0 = np.array([np.dot(p, I[i:i+7]) for i in range(7)])
    Sx1 = np.array([np.dot(p, I[i+1:i+8]) for i in range(7)])
    Sxi0 = np.array([np.dot(q, I[i:i+7]) for i in range(7)])
    Sxi1 = np.array([np.dot(q, I[i+1:i+8]) for i in range(7)])
    
    dMx0 = 2.0 * A_mat.T @ Sx0
    dMx1 = 2.0 * A_mat.T @ Sx1
    dMxi0 = 2.0 * A_mat.T @ Sxi0
    dMxi1 = 2.0 * A_mat.T @ Sxi1
    
    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)
    
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) * inv_Mx0_sq
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) * inv_Mxi0_sq
    
    # Gradient of F = Vx * Vxi
    # dF = Vxi * dVx + Vx * dVxi
    grad_F = vxi * dvx + vx * dvxi
    
    return grad_F

def _jacobian_stationary(a):
    """
    Compute the Jacobian of the stationary equations using finite differences.
    """
    eps = 1e-7
    J = np.zeros((3, 3))
    f0 = _stationary_equations(a)
    for i in range(3):
        a_plus = a.copy()
        a_plus[i] += eps
        f_plus = _stationary_equations(a_plus)
        J[:, i] = (f_plus - f0) / eps
    return J

def _newton_polish(a, max_iter=50, tol=1e-14):
    """
    Newton's method to polish a stationary point.
    """
    for _ in range(max_iter):
        f = _stationary_equations(a)
        if np.max(np.abs(f)) < tol:
            break
        J = _jacobian_stationary(a)
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            # Singular Jacobian, use pseudo-inverse
            step = np.linalg.lstsq(J, -f, rcond=None)[0]
        # Damping to ensure convergence
        alpha = 1.0
        for _ in range(20):
            a_new = a + alpha * step
            f_new = _stationary_equations(a_new)
            if np.max(np.abs(f_new)) < np.max(np.abs(f)):
                break
            alpha *= 0.5
        a = a + alpha * step
    return a

def _is_local_min(a):
    """
    Check if a is a local minimum by checking Hessian positive definiteness.
    Approximate Hessian using finite differences of the gradient.
    """
    eps = 1e-6
    grad0 = _stationary_equations(a)
    if np.any(np.abs(grad0) > 1e-4):
        return False
        
    hess = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a_plus = a.copy()
            a_minus = a.copy()
            a_plus[i] += eps
            a_minus[i] -= eps
            if i == j:
                hess[i, j] = (_stationary_equations(a_plus)[i] - _stationary_equations(a_minus)[i]) / (2*eps)
            else:
                # Cross derivatives
                a_pp = a.copy(); a_pm = a.copy(); a_mp = a.copy(); a_mm = a.copy()
                a_pp[i] += eps; a_pp[j] += eps
                a_pm[i] += eps; a_pm[j] -= eps
                a_mp[i] -= eps; a_mp[j] += eps
                a_mm[i] -= eps; a_mm[j] -= eps
                hess[i, j] = (_stationary_equations(a_pp)[i] - _stationary_equations(a_pm)[i] - _stationary_equations(a_mp)[i] + _stationary_equations(a_mm)[i]) / (4*eps*eps)
                
    try:
        eigvals = np.linalg.eigvalsh(hess)
        return np.all(eigvals > -1e-6)
    except:
        return False

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 5  # seconds
    
    # Generate initial guesses for root finding
    starts = []
    
    # More structured grid-based starts
    a0_range = np.linspace(-2.5, 1.5, 8)
    a4_range = np.linspace(-0.3, 0.3, 8)
    a8_range = np.linspace(-0.02, 0.02, 8)
    
    for a0 in a0_range:
        for a4 in a4_range:
            for a8 in a8_range:
                starts.append(np.array([a0, a4, a8]))
                
    # Random starts with different distributions
    rng = np.random.default_rng(42)
    for _ in range(100):
        a0 = rng.uniform(-4, 3)
        a4 = rng.uniform(-0.8, 0.8)
        a8 = rng.uniform(-0.1, 0.1)
        starts.append(np.array([a0, a4, a8]))
        
    # Include seed
    starts.append(seed_params)
    
    # Solve stationary point equations for each start
    found_solutions = []
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            # Use hybr (MINPACK) which is robust for small systems
            res = root(_stationary_equations, start, method='hybr', tol=1e-10)
            
            if res.success:
                a_sol = res.x
                # Verify it's a valid point
                val, _ = _compute_bound_and_grad(a_sol)
                
                if np.isfinite(val) and val < 1e15:
                    # Check if we've already found this solution
                    is_new = True
                    for sol in found_solutions:
                        if np.linalg.norm(a_sol - sol) < 1e-6:
                            is_new = False
                            break
                    if is_new:
                        found_solutions.append(a_sol.copy())
                        
                    # Polish with Newton's method
                    if time.time() < _DEADLINE - deadline_margin:
                        a_polished = _newton_polish(a_sol, max_iter=30, tol=1e-13)
                        val_polished, _ = _compute_bound_and_grad(a_polished)
                        
                        if np.isfinite(val_polished) and val_polished < val:
                            a_sol = a_polished
                            val = val_polished
                    
                    if val < best_val:
                        best_val = val
                        best_params = a_sol.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < _DEADLINE - deadline_margin:
        try:
            # Newton polish on the best solution
            a_final = _newton_polish(best_params, max_iter=100, tol=1e-15)
            val_final, _ = _compute_bound_and_grad(a_final)
            
            if np.isfinite(val_final) and val_final < best_val:
                best_val = val_final
                best_params = a_final.copy()
        except Exception:
            pass
        
        # Try L-BFGS-B as a backup
        if time.time() < _DEADLINE - deadline_margin:
            try:
                res_min = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B', jac=True, options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
                if res_min.success and res_min.fun < best_val:
                    best_val = res_min.fun
                    best_params = res_min.x
            except Exception:
                pass
        
        # Final Newton polish after L-BFGS-B
        if time.time() < _DEADLINE - deadline_margin:
            try:
                a_final2 = _newton_polish(best_params, max_iter=50, tol=1e-16)
                val_final2, _ = _compute_bound_and_grad(a_final2)
                
                if np.isfinite(val_final2) and val_final2 < best_val:
                    best_val = val_final2
                    best_params = a_final2.copy()
            except Exception:
                pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
