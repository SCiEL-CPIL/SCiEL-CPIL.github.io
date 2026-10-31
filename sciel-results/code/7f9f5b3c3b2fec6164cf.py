import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, root

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Hermite polynomial coefficients in power basis x^{2k}
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))

def _get_p(a):
    return A_mat @ a

def _compute_bound(a):
    """Compute the uncertainty bound for given coefficients a = [a0, a4, a8]."""
    p = _get_p(a)
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            Mx0 += pi * p[j] * I[i+j]
            Mx1 += pi * p[j] * I[i+j+1]
            Mxi0 += qi * q[j] * I[i+j]
            Mxi1 += qi * q[j] * I[i+j+1]
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18
        
    return bound

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    p = _get_p(a)
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            Mx0 += pi * p[j] * I[i+j]
            Mx1 += pi * p[j] * I[i+j+1]
            Mxi0 += qi * q[j] * I[i+j]
            Mxi1 += qi * q[j] * I[i+j+1]
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    # J = 0.5 * (vx * vxi)^0.5
    # dJ = 0.5 * (vx * vxi)^-0.5 * (d(vx*vxi))
    # d(vx*vxi) = vxi * dvx + vx * dvxi
    # dvx = d(Mx1/Mx0) = (dMx1*Mx0 - Mx1*dMx0) / Mx0^2
    # dvxi = d(Mxi1/Mxi0) = (dMxi1*Mxi0 - Mxi1*dMxi0) / Mxi0^2
    
    # dM[n] = 2 * sum_{i,j} (dp_i * p_j) I[i+j+n]
    # dp_i = sum_m A[i,m] * da_m
    # So dM[n]_m = 2 * sum_i A[i,m] sum_j p_j I[i+j+n]
    
    # Let S[n]_i = sum_j p_j I[i+j+n]
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    # dM[n] is a vector of 3 components (for each a_m)
    # dM[n]_m = 2 * sum_i A[i,m] S[n, i]
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    # dvx_m = (dMx[1,m]*Mx0 - Mx1*dMx[0,m]) / Mx0^2
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx0 - Mx1*dMx[0, m]) / (Mx0*Mx0)
        dvxi[m] = (dMxi[1, m]*Mxi0 - Mxi1*dMxi[0, m]) / (Mxi0*Mxi0)
        
    # d(bound) = (1/(2*bound)) * (vxi*dvx + vx*dvxi)
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi*dvx[m] + vx*dvxi[m])
        
    return bound, grad

def _stationary_equations(a):
    """
    Computes the gradient of the squared objective (Vx * Vxi).
    F(a) = Vxi * grad_Vx + Vx * grad_Vxi = 0
    This is the condition for stationary points of the product.
    """
    p = _get_p(a)
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            Mx0 += pi * p[j] * I[i+j]
            Mx1 += pi * p[j] * I[i+j+1]
            Mxi0 += qi * q[j] * I[i+j]
            Mxi1 += qi * q[j] * I[i+j+1]
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    # Compute gradients of Mx0, Mx1, Mxi0, Mxi1
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
            
    # grad Vx = (dMx1 * Mx0 - Mx1 * dMx0) / Mx0^2
    # grad Vxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) / Mxi0^2
    
    grad_vx = np.zeros(3)
    grad_vxi = np.zeros(3)
    
    for m in range(3):
        grad_vx[m] = (dMx[1, m]*Mx0 - Mx1*dMx[0, m]) / (Mx0*Mx0)
        grad_vxi[m] = (dMxi[1, m]*Mxi0 - Mxi1*dMxi[0, m]) / (Mxi0*Mxi0)
        
    # System: Vxi * grad_vx + Vx * grad_vxi = 0
    F = vxi * grad_vx + vx * grad_vxi
    
    return F

def _is_local_min(a):
    """
    Check if a is a local minimum by checking Hessian positive definiteness.
    Approximate Hessian using finite differences of the gradient of the bound.
    """
    eps = 1e-5
    val0, grad0 = _compute_bound_and_grad(a)
    if not np.isfinite(val0):
        return False
        
    # Check if gradient is small enough to be considered a stationary point
    if np.linalg.norm(grad0) > 1e-4:
        return False
        
    hess = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            a_plus = a.copy()
            a_minus = a.copy()
            a_plus[i] += eps
            a_minus[i] -= eps
            if i == j:
                hess[i, j] = (grad0[i] - (lambda g: g[i])(_compute_bound_and_grad(a_minus)[1])) / (2*eps) # Rough diagonal estimate, better to use central diff of grad
                # Let's do proper central difference for Hessian elements
                # H[i,j] = d(grad_i)/da_j
                # We need grad at a +/- eps e_j
                pass
                
    # Redo Hessian calculation properly
    hess = np.zeros((3, 3))
    for j in range(3): # derivative with respect to a_j
        a_plus = a.copy()
        a_minus = a.copy()
        a_plus[j] += eps
        a_minus[j] -= eps
        _, grad_plus = _compute_bound_and_grad(a_plus)
        _, grad_minus = _compute_bound_and_grad(a_minus)
        for i in range(3): # gradient component i
            hess[i, j] = (grad_plus[i] - grad_minus[i]) / (2*eps)
            
    try:
        eigvals = np.linalg.eigvalsh(hess)
        return np.all(eigvals > -1e-4)
    except:
        return False

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _compute_bound(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 15  # seconds safety margin
    
    # Phase 1: Grid search around the seed to find good initial points
    # Use a finer grid with more points in the neighborhood
    grid_centers = [seed_params]
    
    # Add perturbation points at multiple scales
    scales = [0.005, 0.01, 0.02, 0.05, 0.1]
    for scale in scales:
        for da0 in [-1, -0.5, 0, 0.5, 1]:
            for da4 in [-1, -0.5, 0, 0.5, 1]:
                for da8 in [-1, -0.5, 0, 0.5, 1]:
                    if abs(da0) == 1 and abs(da4) == 1 and abs(da8) == 1:
                        continue  # skip corners for some scales to save time
                    grid_centers.append(seed_params + np.array([da0 * scale, da4 * scale, da8 * scale * 0.01]))
    
    # Also add some heuristic guesses
    heuristic_guesses = [
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-1.1, 0.03, 0.0002]),
        np.array([-1.15, 0.05, 0.0004]),
        np.array([-1.05, 0.045, 0.00035]),
        np.array([-1.18, 0.035, 0.00025]),
        np.array([-0.95, 0.055, 0.00025]),
        np.array([-1.25, 0.025, 0.00045]),
    ]
    
    all_starts = grid_centers + heuristic_guesses
    
    # Phase 2: Try L-BFGS-B from multiple starting points
    for start in all_starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            res = minimize(_compute_bound, start, method='L-BFGS-B',
                          jac=lambda x: _compute_bound_and_grad(x)[1],
                          options={'gtol': 1e-10, 'maxiter': 200})
            if res.success and np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Try root finding from the best candidate and a few nearby points
    root_starts = [best_params]
    for scale in [0.001, 0.005, 0.01]:
        for da0 in [-1, 1]:
            for da4 in [-1, 1]:
                for da8 in [-1, 1]:
                    root_starts.append(best_params + np.array([da0 * scale, da4 * scale, da8 * scale * 0.01]))
    
    for start in root_starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        # Try Hybrid method
        try:
            res = root(_stationary_equations, start, method='hybr', tol=1e-14)
            if res.success and np.all(np.isfinite(res.x)):
                val = _compute_bound(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
        
        # Try Levenberg-Marquardt method
        if time.time() > _DEADLINE - deadline_margin:
            break
        try:
            res = root(_stationary_equations, start, method='lm', tol=1e-14)
            if res.success and np.all(np.isfinite(res.x)):
                val = _compute_bound(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: High-precision polish using BFGS with very tight tolerance
    if time.time() < _DEADLINE - 10:
        try:
            res_min = minimize(_compute_bound, best_params, method='BFGS', 
                               jac=lambda x: _compute_bound_and_grad(x)[1], 
                               options={'gtol': 1e-12, 'maxiter': 500})
            if res_min.success and np.isfinite(res_min.fun) and res_min.fun < best_val:
                best_val = res_min.fun
                best_params = res_min.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final Newton-type polish using the stationary equations
    if time.time() < _DEADLINE - 5:
        try:
            res = root(_stationary_equations, best_params, method='hybr', tol=1e-15)
            if res.success and np.all(np.isfinite(res.x)):
                val = _compute_bound(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Final check: ensure we return a valid array
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
