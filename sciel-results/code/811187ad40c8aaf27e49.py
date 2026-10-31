import numpy as np
import time
from scipy.optimize import minimize, root

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Hermite polynomial coefficients in power basis x^{2k}
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# Constraint: g(0) = 0 => a0 + 3*a4 + 105*a8 + 945*a12 = 0
# a12 = -(a0 + 3*a4 + 105*a8) / 945
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    """
    Computes the uncertainty bound and its gradient with respect to a.
    Bound = sqrt(Vx * Vxi)
    """
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
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Compute gradients of Mx0, Mx1, Mxi0, Mxi1
    # dMx0/da_m = 2 * sum_i p_i A_mat[i,m] I[i+j] ... wait, let's use the chain rule properly.
    # p = A a
    # Mx0 = sum_i,j p_i p_j I[i+j]
    # dMx0/da_m = sum_i,j (dp_i/da_m p_j + p_i dp_j/da_m) I[i+j]
    #            = 2 sum_i,j p_j I[i+j] A_mat[i,m]
    # Let Sx[n, i] = sum_j p_j I[i+j+n] for n=0,1
    # Then dMx0/da_m = 2 sum_i A_mat[i,m] Sx[0, i]
    # dMx1/da_m = 2 sum_i A_mat[i,m] Sx[1, i]
    
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
        
    # grad Bound = (1/(2*sqrt(Vx*Vxi))) * (Vxi * grad_Vx + Vx * grad_Vxi)
    #             = (1/(2*bound)) * (Vxi * grad_Vx + Vx * grad_Vxi)
    
    grad_bound = (vxi * grad_vx + vx * grad_vxi) / (2.0 * bound)
    
    return bound, grad_bound

def _stationary_equations(a):
    """
    Computes the gradient of the squared objective (Vx * Vxi).
    F(a) = Vxi * grad_Vx + Vx * grad_Vxi = 0
    """
    bound, grad_bound = _compute_bound_and_grad(a)
    # grad_bound is proportional to the stationary condition
    # If bound is huge, gradient is unreliable
    if bound > 1e10:
        return np.zeros(3)
    
    # The stationary condition is grad_Vx/Vx + grad_Vxi/Vxi = 0 for log(product)
    # Or Vxi * grad_Vx + Vx * grad_Vxi = 0
    # grad_bound = (Vxi * grad_Vx + Vx * grad_Vxi) / (2 * bound)
    # So stationary condition is grad_bound = 0
    return 2.0 * bound * grad_bound

def _newton_polish(a, max_iter=50, tol=1e-14):
    """
    High-precision Newton polish using numerical Jacobian.
    """
    a = a.copy()
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        F = _stationary_equations(a)
        if not np.all(np.isfinite(F)):
            break
        norm_F = np.linalg.norm(F)
        if norm_F < tol:
            break
        
        # Numerical Jacobian
        h = 1e-8
        J = np.zeros((3, 3))
        for j in range(3):
            a_plus = a.copy()
            a_plus[j] += h
            F_plus = _stationary_equations(a_plus)
            if not np.all(np.isfinite(F_plus)):
                break
            J[:, j] = (F_plus - F) / h
        if not np.all(np.isfinite(J)):
            break
        
        # Solve J * delta = -F
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -F, rcond=None)[0]
        
        # Damped step
        step_size = 1.0
        a_trial = a + delta
        F_trial = _stationary_equations(a_trial)
        while step_size > 1e-12:
            if np.all(np.isfinite(F_trial)) and np.linalg.norm(F_trial) < norm_F:
                break
            step_size *= 0.5
            delta = step_size * delta
            a_trial = a + delta
            F_trial = _stationary_equations(a_trial)
        
        a = a_trial
        
    return a

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 15
    
    # Generate a set of diverse initial guesses
    # 1. Seed
    # 2. Perturbations
    # 3. Structured guesses based on Hermite properties
    initial_guesses = [
        seed_params,
        seed_params + np.array([0.01, 0.01, 0.0001]),
        seed_params - np.array([0.01, 0.01, 0.0001]),
        seed_params + np.array([-0.01, 0.01, 0.0001]),
        seed_params + np.array([0.01, -0.01, 0.0001]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-1.13, 0.04, 0.000308642]) + np.array([0.05, -0.02, 0.0002]),
        np.array([-1.13, 0.04, 0.000308642]) + np.array([-0.05, 0.02, -0.0002]),
        np.array([-0.9, 0.06, 0.0002]),
        np.array([-1.3, 0.02, 0.0005]),
        np.array([-1.15, 0.03, 0.0003]),
        np.array([-1.10, 0.05, 0.0003]),
    ]
    
    candidates = []
    
    for start in initial_guesses:
        if time.time() > _DEADLINE - deadline_margin:
            break
        
        # Use L-BFGS-B for robust local optimization
        try:
            res = minimize(
                _compute_bound_and_grad, 
                start, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if np.all(np.isfinite(res.x)):
                val, _ = _compute_bound_and_grad(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
                candidates.append(res.x.copy())
        except Exception:
            pass

    # Phase 2: High-precision Newton polish on best candidates
    if time.time() < _DEADLINE - 10:
        polished = set()
        # Sort candidates by their objective value to polish the best ones first
        scored_candidates = []
        for cand in candidates:
            val, _ = _compute_bound_and_grad(cand)
            scored_candidates.append((val, cand))
        scored_candidates.sort(key=lambda x: x[0])
        
        for val, cand in scored_candidates[:5]: # Polish top 5
            key = tuple(np.round(cand, 8))
            if key in polished:
                continue
            polished.add(key)
            
            if time.time() > _DEADLINE - 8:
                break
            
            try:
                polished_a = _newton_polish(cand, max_iter=50, tol=1e-14)
                if np.all(np.isfinite(polished_a)):
                    p_val, _ = _compute_bound_and_grad(polished_a)
                    if p_val < best_val:
                        best_val = p_val
                        best_params = polished_a.copy()
            except Exception:
                pass
    
    # Final polish on the best result
    if time.time() < _DEADLINE - 5:
        try:
            final_a = _newton_polish(best_params, max_iter=100, tol=1e-15)
            if np.all(np.isfinite(final_a)):
                f_val, _ = _compute_bound_and_grad(final_a)
                if f_val < best_val:
                    best_val = f_val
                    best_params = final_a.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
