import numpy as np
import time
from scipy.optimize import root, minimize
from scipy.linalg import solve

_DEADLINE = time.time() + 200

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
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
# H4 coeffs: [3, -6, 1, 0, 0, 0, 0]
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

def _compute_bound(a):
    p = _get_p(a)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    Mx0 = np.dot(p, np.dot(p, I))
    Mx1 = np.dot(p, np.dot(p, I[1:]))
    Mxi0 = np.dot(q, np.dot(q, I))
    Mxi1 = np.dot(q, np.dot(q, I[1:]))
    
    if Mx0 <= 1e-14 or Mxi0 <= 1e-14:
        return 1e18
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    if vx <= 1e-14 or vxi <= 1e-14:
        return 1e18
    prod = vx * vxi
    if prod <= 0:
        return 1e18
    return np.sqrt(prod)

def _compute_bound_and_grad(a):
    p = _get_p(a)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    Mx0 = np.dot(p, np.dot(p, I))
    Mx1 = np.dot(p, np.dot(p, I[1:]))
    Mxi0 = np.dot(q, np.dot(q, I))
    Mxi1 = np.dot(q, np.dot(q, I[1:]))
    
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
    
    grad_F = vxi * dvx + vx * dvxi
    
    return grad_F

def _jacobian_stationary(a):
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
    for _ in range(max_iter):
        f = _stationary_equations(a)
        if np.max(np.abs(f)) < tol:
            break
        J = _jacobian_stationary(a)
        try:
            step = solve(J, -f)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -f, rcond=None)[0]
        alpha = 1.0
        for _ in range(20):
            a_new = a + alpha * step
            f_new = _stationary_equations(a_new)
            if np.max(np.abs(f_new)) < np.max(np.abs(f)):
                break
            alpha *= 0.5
        a = a + alpha * step
    return a

def construct():
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 10  # seconds
    
    # Generate initial guesses
    starts = []
    
    # Grid search
    a0_range = np.linspace(-3.0, 2.0, 15)
    a4_range = np.linspace(-0.5, 0.5, 15)
    a8_range = np.linspace(-0.05, 0.05, 15)
    
    for a0 in a0_range:
        for a4 in a4_range:
            for a8 in a8_range:
                starts.append(np.array([a0, a4, a8]))
                
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(200):
        a0 = rng.uniform(-5, 5)
        a4 = rng.uniform(-1.0, 1.0)
        a8 = rng.uniform(-0.2, 0.2)
        starts.append(np.array([a0, a4, a8]))
        
    starts.append(seed_params)
    
    found_solutions = []
    
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            # Use hybr (MINPACK)
            res = root(_stationary_equations, start, method='hybr', tol=1e-10)
            
            if res.success:
                a_sol = res.x
                val, _ = _compute_bound_and_grad(a_sol)
                
                if np.isfinite(val) and val < 1e15:
                    is_new = True
                    for sol in found_solutions:
                        if np.linalg.norm(a_sol - sol) < 1e-6:
                            is_new = False
                            break
                    if is_new:
                        found_solutions.append(a_sol.copy())
                        
                    if time.time() < _DEADLINE - deadline_margin:
                        a_polished = _newton_polish(a_sol, max_iter=50, tol=1e-14)
                        val_polished, _ = _compute_bound_and_grad(a_polished)
                        
                        if np.isfinite(val_polished) and val_polished < val:
                            a_sol = a_polished
                            val = val_polished
                    
                    if val < best_val:
                        best_val = val
                        best_params = a_sol.copy()
        except Exception:
            pass

    # Final polish
    if time.time() < _DEADLINE - deadline_margin:
        try:
            a_final = _newton_polish(best_params, max_iter=100, tol=1e-15)
            val_final, _ = _compute_bound_and_grad(a_final)
            
            if np.isfinite(val_final) and val_final < best_val:
                best_val = val_final
                best_params = a_final.copy()
        except Exception:
            pass
        
        # L-BFGS-B backup
        if time.time() < _DEADLINE - deadline_margin:
            try:
                res_min = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
                if res_min.success and res_min.fun < best_val:
                    best_val = res_min.fun
                    best_params = res_min.x
            except Exception:
                pass
        
        # Final Newton
        if time.time() < _DEADLINE - deadline_margin:
            try:
                a_final2 = _newton_polish(best_params, max_iter=100, tol=1e-16)
                val_final2, _ = _compute_bound_and_grad(a_final2)
                
                if np.isfinite(val_final2) and val_final2 < best_val:
                    best_val = val_final2
                    best_params = a_final2.copy()
            except Exception:
                pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
