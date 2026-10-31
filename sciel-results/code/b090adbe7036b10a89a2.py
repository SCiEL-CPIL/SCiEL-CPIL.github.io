import numpy as np
import time
from scipy.optimize import minimize, fsolve, newton

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
H0_c = np.array([1.0, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3.0, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105.0, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945.0, -7560, 13860, -6930, 1155, -66, 1])

c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

sign_vec = np.array([1.0, -1, 1, -1, 1, -1, 1])

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    p = _get_p(a)
    q = p * sign_vec
    
    Mx = np.zeros(2)
    Mxi = np.zeros(2)
    
    for i in range(7):
        pi = p[i]
        if pi == 0: continue
        for j in range(7):
            idx0 = i + j
            idx1 = i + j + 1
            Mx[0] += pi * p[j] * I[idx0]
            Mx[1] += pi * p[j] * I[idx1]
            
            idx0_q = i + j
            idx1_q = i + j + 1
            Mxi[0] += q[i] * q[j] * I[idx0_q]
            Mxi[1] += q[i] * q[j] * I[idx1_q]
            
    if Mx[0] <= 1e-15 or Mxi[0] <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx[1] / Mx[0]
    vxi = Mxi[1] / Mxi[0]
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx[0] - Mx[1]*dMx[0, m]) / (Mx[0]*Mx[0])
        dvxi[m] = (dMxi[1, m]*Mxi[0] - Mxi[1]*dMxi[0, m]) / (Mxi[0]*Mxi[0])
        
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi*dvx[m] + vx*dvxi[m])
        
    return bound, grad

def _objective(a):
    val, _ = _compute_bound_and_grad(a)
    return val

def _objective_grad(a):
    _, grad = _compute_bound_and_grad(a)
    return grad

def _stationary_equations(a):
    """
    Return the gradient of the uncertainty product (the stationary conditions).
    At a stationary point, this vector is zero.
    """
    _, grad = _compute_bound_and_grad(a)
    return grad

def _is_local_min(a, grad, h=1e-6):
    """
    Check if a is a local minimum by verifying the Hessian is positive definite
    (or at least the directional curvature is positive in random directions).
    """
    if not np.all(np.isfinite(grad)):
        return False
    if np.linalg.norm(grad) > 1e-8:
        return False
    
    # Check Hessian via finite differences in a few directions
    np.random.seed(0)
    for _ in range(5):
        d = np.random.randn(3)
        d /= np.linalg.norm(d)
        f0 = _objective(a)
        f1 = _objective(a + h*d)
        f2 = _objective(a - h*d)
        curvature = (f1 - 2*f0 + f2) / (h*h)
        if curvature < -1e-10:
            return False
    return True

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 15  # seconds
    
    # Strategy 1: Multi-start root-finding on stationary equations
    # This is structurally different from gradient descent - we solve grad=0 directly
    np.random.seed(42)
    
    # Generate diverse starting points for fsolve
    starts = [seed_params.copy()]
    
    # Random starts in a reasonable range
    for _ in range(40):
        start = np.random.uniform(-5, 5, 3)
        starts.append(start)
    
    # Structured starts around the seed
    for a0 in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        for a4 in [-0.2, -0.1, 0, 0.1, 0.2]:
            for a8 in [-0.05, -0.01, 0, 0.01, 0.05]:
                starts.append(np.array([a0, a4, a8]))
    
    # Run fsolve from each start to find stationary points
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
        
        try:
            sol, info, ier, msg = fsolve(_stationary_equations, start, full_output=True, 
                                         maxfev=200, xtol=1e-14)
            
            if ier == 1:  # Success
                if np.all(np.isfinite(sol)):
                    val, grad = _compute_bound_and_grad(sol)
                    if np.isfinite(val) and np.linalg.norm(grad) < 1e-8:
                        if val < best_val:
                            best_val = val
                            best_params = sol.copy()
        except Exception:
            pass
    
    # Strategy 2: L-BFGS-B from the best stationary point found
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_objective, best_params, jac=_objective_grad, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Try Newton's method from multiple points for high-precision refinement
    if time.time() < _DEADLINE - deadline_margin:
        # Use the best params as a starting point for Newton
        try:
            # Newton iteration on the gradient
            a_newton = best_params.copy()
            for _ in range(50):
                if time.time() > _DEADLINE - deadline_margin:
                    break
                grad = _stationary_equations(a_newton)
                if np.linalg.norm(grad) < 1e-14:
                    break
                
                # Compute Jacobian of the gradient (Hessian of the objective)
                J = np.zeros((3, 3))
                h = 1e-7
                for i in range(3):
                    e_i = np.zeros(3)
                    e_i[i] = h
                    J[:, i] = (_stationary_equations(a_newton + e_i) - _stationary_equations(a_newton - e_i)) / (2*h)
                
                try:
                    step = np.linalg.solve(J, -grad)
                    a_newton = a_newton + step
                except np.linalg.LinAlgError:
                    break
                
                if not np.all(np.isfinite(a_newton)):
                    break
            
            if np.all(np.isfinite(a_newton)):
                val, grad = _compute_bound_and_grad(a_newton)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_params = a_newton.copy()
        except Exception:
            pass
    
    # Strategy 4: Final polish with very tight tolerance
    if time.time() < _DEADLINE - deadline_margin and best_val < seed_val:
        try:
            res_local = minimize(_objective, best_params, jac=_objective_grad, method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-15})
            if np.isfinite(res_local.fun) and res_local.fun < best_val:
                best_val = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
