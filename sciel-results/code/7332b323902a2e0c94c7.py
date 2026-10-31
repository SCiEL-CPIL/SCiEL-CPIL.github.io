import numpy as np
import time
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 200

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(30)
I[0] = np.sqrt(np.pi)
for n in range(1, 30):
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

A_mat = np.column_stack((col0, col1, col2))  # Shape (7, 3)

sign_vec = np.array([1, -1, 1, -1, 1, -1, 1])  # (-1)^k

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound sqrt(vx * vxi) and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    p = A_mat @ a
    q = p * sign_vec
    
    # Compute Mx[0], Mx[1], Mxi[0], Mxi[1]
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
            idx0 = i + j
            idx1 = i + j + 1
            Mx0 += pi * pj * I[idx0]
            Mx1 += pi * pj * I[idx1]
            Mxi0 += qi * qj * I[idx0]
            Mxi1 += qi * qj * I[idx1]
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient
    # Sx[n, i] = sum_j p_j I[i+j+n]
    # Sxi[n, i] = sum_j q_j I[i+j+n]
    Sx0 = np.zeros(7)
    Sx1 = np.zeros(7)
    Sxi0 = np.zeros(7)
    Sxi1 = np.zeros(7)
    
    for i in range(7):
        Sx0[i] = np.dot(p, I[i:i+7])
        Sx1[i] = np.dot(p, I[i+1:i+8])
        Sxi0[i] = np.dot(q, I[i:i+7])
        Sxi1[i] = np.dot(q, I[i+1:i+8])
    
    # dMx[0,m] = 2 * sum_i A[i,m] Sx0[i]
    # dMx[1,m] = 2 * sum_i A[i,m] Sx1[i]
    # etc.
    dMx0 = 2.0 * (A_mat.T @ Sx0)
    dMx1 = 2.0 * (A_mat.T @ Sx1)
    dMxi0 = 2.0 * (A_mat.T @ Sxi0)
    dMxi1 = 2.0 * (A_mat.T @ Sxi1)
    
    # dvx[m] = (dMx1[m]*Mx0 - Mx1*dMx0[m]) / Mx0^2
    # dvxi[m] = (dMxi1[m]*Mxi0 - Mxi1*dMxi0[m]) / Mxi0^2
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) / (Mx0 * Mx0)
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) / (Mxi0 * Mxi0)
    
    factor = 1.0 / (2.0 * bound)
    grad = factor * (vxi * dvx + vx * dvxi)
    
    return bound, grad

def _objective_and_grad(u):
    """
    Objective in transformed coordinates.
    u = [u0, u4, u8] where a_i = sign(u_i) * exp(|u_i|) ... 
    Actually let's use a simpler transform: a_i = u_i directly but with a scaling.
    
    Let's use: a_i = exp(m_i) * tanh(t_i) where m_i, t_i are the variables.
    But that's 6 variables. Let's stick with 3 variables but use a smart transform.
    
    Alternative: just use a_i directly but with a good initial guess and multiple starts.
    The log-sign transform is 6D which might be harder. Let's try a different approach.
    
    Actually, let's use the transform: a_i = s_i * exp(m_i) where s_i in {-1, 1} is fixed per start,
    and we optimize over m_i in R^3. This is a 3D optimization for each sign combination.
    """
    m = u  # log magnitudes
    a = np.zeros(3)
    for i in range(3):
        a[i] = np.sign(u[i]) * np.exp(np.abs(u[i])) if abs(u[i]) > 1e-15 else 0.0
    # Wait, this doesn't work well because sign(u) * exp(|u|) = u if u>0, -exp(-u) if u<0... no.
    # Let me use: a_i = exp(m_i) * tanh(t_i) is 6D.
    # Or: a_i = m_i * tanh(m_i) ... no.
    # Let's just use a_i = u_i directly and rely on multiple starts.
    # But the log-sign idea was to avoid local minima. Let me use a_i = exp(u_i) for positive, 
    # and handle sign separately.
    
    # Actually, the simplest robust approach: use a_i directly, with many random starts
    # and L-BFGS-B with analytic gradient.
    return _compute_bound_and_grad(u)

def _obj(u):
    val, _ = _compute_bound_and_grad(u)
    return val

def _obj_grad(u):
    val, grad = _compute_bound_and_grad(u)
    return val, grad

def construct():
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 8  # seconds
    
    def _try_optimize(start, maxiter=3000):
        nonlocal best_val, best_params
        if time.time() > _DEADLINE - deadline_margin:
            return
        try:
            res = minimize(_obj_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if np.isfinite(res.fun) and res.fun < best_val:
                val_check, _ = _compute_bound_and_grad(res.x)
                if val_check < best_val:
                    best_val = val_check
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Generate diverse starting points
    rng = np.random.default_rng(12345)
    starts = [seed_params.copy()]
    
    # Scale the seed in various ways
    for scale in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        starts.append(seed_params * scale)
        starts.append(-seed_params * scale)
    
    # Random starts in a reasonable box
    for _ in range(200):
        a0 = rng.uniform(-5, 2)
        a4 = rng.uniform(-0.5, 0.5)
        a8 = rng.uniform(-0.05, 0.05)
        starts.append(np.array([a0, a4, a8]))
    
    # Some structured starts based on the constraint that g(0)=0
    # The constraint is automatically satisfied by the H12 coefficient
    for _ in range(100):
        # Try to make p have specific properties
        a0 = rng.uniform(-3, 1)
        a4 = rng.uniform(-0.2, 0.2)
        a8 = rng.uniform(-0.02, 0.02)
        starts.append(np.array([a0, a4, a8]))
    
    # Shuffle starts (keep seed first)
    rng2 = np.random.default_rng(777)
    perm = np.arange(len(starts))
    perm[1:] = rng2.permutation(perm[1:])
    starts = [starts[i] for i in perm]
    
    # Optimize from each start
    for i, start in enumerate(starts):
        if time.time() > _DEADLINE - deadline_margin:
            break
        # Use fewer iterations for early starts, more for later ones
        if i < 20:
            maxiter = 1000
        else:
            maxiter = 500
        _try_optimize(start, maxiter)
    
    # Final polish on best found
    if time.time() < _DEADLINE - deadline_margin:
        _try_optimize(best_params.copy(), maxiter=10000)
    
    # Nelder-Mead polish
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_obj, best_params.copy(), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                val_check, _ = _compute_bound_and_grad(res.x)
                if val_check < best_val:
                    best_val = val_check
                    best_params = res.x.copy()
        except Exception:
            pass
        
        # One more L-BFGS-B after Nelder-Mead
        if time.time() < _DEADLINE - deadline_margin:
            _try_optimize(best_params.copy(), maxiter=10000)
    
    if not np.all(np.isfinite(best_params)):
        return seed_params
    
    return np.array(best_params)
