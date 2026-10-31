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
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))  # Shape (7, 3)

def _get_p(a):
    return A_mat @ a

def _compute_bound(a):
    """
    Compute the uncertainty bound for given coefficients a = [a0, a4, a8].
    Returns the bound value.
    """
    p = _get_p(a)
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    # Compute Mx[0], Mx[1], Mxi[0], Mxi[1]
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
            pj = p[j]
            qj = q[j]
            Mx0 += pi * pj * I[i+j]
            Mx1 += pi * pj * I[i+j+1]
            Mxi0 += qi * qj * I[i+j]
            Mxi1 += qi * qj * I[i+j+1]
    
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
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    # Compute moments
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
            pj = p[j]
            qj = q[j]
            Mx0 += pi * pj * I[i+j]
            Mx1 += pi * pj * I[i+j+1]
            Mxi0 += qi * qj * I[i+j]
            Mxi1 += qi * qj * I[i+j+1]
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation using chain rule
    # dM[n]_m = 2 * sum_i A[i,m] * sum_j p_j I[i+j+n]
    
    # Precompute S[n, i] = sum_j p_j I[i+j+n]
    Sx0 = np.zeros(7)
    Sx1 = np.zeros(7)
    Sxi0 = np.zeros(7)
    Sxi1 = np.zeros(7)
    
    for i in range(7):
        Sx0[i] = np.dot(p, I[i:i+7])
        Sx1[i] = np.dot(p, I[i+1:i+8])
        Sxi0[i] = np.dot(q, I[i:i+7])
        Sxi1[i] = np.dot(q, I[i+1:i+8])
    
    # dM[n]_m = 2 * dot(A[:, m], S[n, :])
    dMx0 = np.zeros(3)
    dMx1 = np.zeros(3)
    dMxi0 = np.zeros(3)
    dMxi1 = np.zeros(3)
    
    for m in range(3):
        Am = A_mat[:, m]
        dMx0[m] = 2.0 * np.dot(Am, Sx0)
        dMx1[m] = 2.0 * np.dot(Am, Sx1)
        dMxi0[m] = 2.0 * np.dot(Am, Sxi0)
        dMxi1[m] = 2.0 * np.dot(Am, Sxi1)
    
    # dvx_m = (dMx1[m]*Mx0 - Mx1*dMx0[m]) / Mx0^2
    # dvxi_m = (dMxi1[m]*Mxi0 - Mxi1*dMxi0[m]) / Mxi0^2
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    
    for m in range(3):
        dvx_m = (dMx1[m]*Mx0 - Mx1*dMx0[m]) / (Mx0*Mx0)
        dvxi_m = (dMxi1[m]*Mxi0 - Mxi1*dMxi0[m]) / (Mxi0*Mxi0)
        grad[m] = factor * (vxi*dvx_m + vx*dvxi_m)
        
    return bound, grad

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _compute_bound(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 5  # seconds
    
    # Use direct optimization in a-space with L-BFGS-B
    def _obj_and_grad(a):
        val, grad = _compute_bound_and_grad(a)
        return val, grad
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    
    starts = [seed_params.copy()]
    
    # Grid starts around seed
    for da0 in [-0.5, -0.2, 0.0, 0.2, 0.5]:
        for da4 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            for da8 in [-0.0001, -0.00005, 0.0, 0.00005, 0.0001]:
                candidate = seed_params + np.array([da0, da4, da8])
                starts.append(candidate)
    
    # Random starts
    for _ in range(200):
        a0 = rng.uniform(-3, 1)
        a4 = rng.uniform(-0.2, 0.2)
        a8 = rng.uniform(-0.001, 0.002)
        starts.append(np.array([a0, a4, a8]))
    
    # Some structured starts with different sign patterns
    for sign0 in [-1, 1]:
        for sign4 in [-1, 1]:
            for sign8 in [-1, 1]:
                for mag0 in [0.5, 1.0, 2.0]:
                    for mag4 in [0.01, 0.05, 0.1]:
                        for mag8 in [0.0001, 0.0005, 0.001]:
                            starts.append(np.array([
                                sign0 * mag0,
                                sign4 * mag4,
                                sign8 * mag8
                            ]))
    
    # Shuffle starts (keep seed first)
    rng2 = np.random.default_rng(99)
    perm = np.arange(len(starts))
    perm[1:] = rng2.permutation(perm[1:])
    starts = [starts[i] for i in perm]
    
    # Optimization loop
    for idx, start in enumerate(starts):
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            res = minimize(_obj_and_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _compute_bound(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_obj_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _compute_bound(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead for local refinement
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(lambda a: _compute_bound(a), best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _compute_bound(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
    
    # One more round of L-BFGS-B from the Nelder-Mead result
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_obj_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _compute_bound(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
