import numpy as np
import time
from scipy.optimize import minimize

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

def _compute_hessian(a):
    """Compute the Hessian of the objective function via finite differences."""
    h = 1e-7
    H = np.zeros((3, 3))
    grad0 = _objective_grad(a)
    
    for i in range(3):
        e_i = np.zeros(3)
        e_i[i] = h
        grad_plus = _objective_grad(a + e_i)
        grad_minus = _objective_grad(a - e_i)
        H[:, i] = (grad_plus - grad_minus) / (2*h)
        
    return H

def _trust_region_step(a, grad, H, delta=0.1):
    """
    Compute a trust-region step using the exact Hessian.
    Solves the subproblem: min 0.5*d^T H d + g^T d s.t. ||d|| <= delta
    """
    # Eigen-decomposition of the Hessian
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        # Fallback to steepest descent if Hessian is not symmetric/definite
        norm_g = np.linalg.norm(grad)
        if norm_g < 1e-14:
            return np.zeros(3)
        d = -grad / norm_g
        if np.linalg.norm(d) > delta:
            d = d * delta / np.linalg.norm(d)
        return d
    
    # Project gradient onto eigenbasis
    g_eig = eigvecs.T @ grad
    
    # For each eigen-direction, the optimal step size within the trust region
    # is constrained by the trust radius. We use a simple approach:
    # d_i = -g_i / H_ii, clamped to satisfy ||d|| <= delta
    
    d_eig = np.zeros(3)
    for i in range(3):
        if abs(eigvals[i]) > 1e-12:
            d_eig[i] = -g_eig[i] / eigvals[i]
        else:
            d_eig[i] = 0
    
    # Transform back to original coordinates
    d = eigvecs @ d_eig
    
    # Check if the step is within the trust region
    norm_d = np.linalg.norm(d)
    if norm_d > delta:
        # Scale down to the trust boundary
        d = d * (delta / norm_d)
        
    return d

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
    
    # Strategy 1: Trust-region method with exact Hessian
    # This is structurally different from L-BFGS (quasi-Newton) - we use the exact Hessian
    # and solve a trust-region subproblem at each step
    
    a = best_params.copy()
    delta = 0.5  # initial trust radius
    
    for iteration in range(200):
        if time.time() > _DEADLINE - deadline_margin:
            break
        
        f, grad = _compute_bound_and_grad(a)
        
        if not np.isfinite(f):
            a = seed_params.copy()
            delta = 0.5
            continue
        
        if f < best_val:
            best_val = f
            best_params = a.copy()
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Compute Hessian
        try:
            H = _compute_hessian(a)
        except Exception:
            H = np.eye(3) * 1e-6
        
        # Compute trust-region step
        try:
            d = _trust_region_step(a, grad, H, delta)
        except Exception:
            d = -grad / max(np.linalg.norm(grad), 1e-14)
            if np.linalg.norm(d) > delta:
                d = d * delta / np.linalg.norm(d)
        
        # Try the step
        a_new = a + d
        f_new, grad_new = _compute_bound_and_grad(a_new)
        
        if not np.isfinite(f_new):
            delta *= 0.5
            continue
        
        # Acceptance criterion
        predicted_reduction = -grad @ d - 0.5 * d @ H @ d
        actual_reduction = f - f_new
        
        if actual_reduction > 0.25 * predicted_reduction and predicted_reduction > 0:
            # Accept the step
            a = a_new
            # Increase trust radius
            if actual_reduction > 0.75 * predicted_reduction:
                delta = min(2.0 * delta, 1.0)
            else:
                delta = 1.5 * delta
        else:
            # Reject the step, decrease trust radius
            delta *= 0.5
        
        # If the step was too small, we might be at a minimum
        if np.linalg.norm(d) < 1e-14:
            break
    
    # Strategy 2: L-BFGS-B from the best point found
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_objective, best_params, jac=_objective_grad, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Newton's method for high-precision refinement
    if time.time() < _DEADLINE - deadline_margin:
        try:
            a_newton = best_params.copy()
            for _ in range(50):
                if time.time() > _DEADLINE - deadline_margin:
                    break
                grad = _objective_grad(a_newton)
                if np.linalg.norm(grad) < 1e-14:
                    break
                
                try:
                    H = _compute_hessian(a_newton)
                    # Add regularization if needed
                    eigvals = np.linalg.eigvalsh(H)
                    if np.min(eigvals) < 1e-8:
                        H = H + (1e-8 - np.min(eigvals)) * np.eye(3)
                    
                    step = np.linalg.solve(H, -grad)
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
