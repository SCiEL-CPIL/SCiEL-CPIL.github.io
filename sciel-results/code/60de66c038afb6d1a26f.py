import numpy as np
import time
import math
from scipy.optimize import minimize, basin_hopping

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill

# Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..12
_I = np.zeros(13)
_I[0] = np.sqrt(np.pi)
for n in range(1, 13):
    _I[n] = (2.0 * n - 1.0) / 2.0 * _I[n - 1]

def _coeffs_from_params(a0, a4, a8):
    """
    Compute polynomial coefficients p_k of P(x) = sum p_k x^(2k)
    where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    and a12 = -(a0 + 3*a4 + 105*a8)/945
    Returns p (length 7) and dp_da (3x7 matrix, gradients of p wrt a0, a4, a8)
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    # p[k] is coeff of x^(2k)
    # H0: p0 += a0
    # H4: p0 += 3a4, p1 += -6a4, p2 += a4
    # H8: p0 += 105a8, p1 += -420a8, p2 += 210a8, p3 += -28a8, p4 += a8
    # H12: p0 += 945a12, p1 += -7560a12, p2 += 13860a12, p3 += -6930a12, p4 += 1155a12, p5 += -66a12, p6 += a12
    
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    
    # Gradients dp/da
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    dp_da = np.zeros((3, 7))
    
    # d/da0
    dp_da[0, 0] = 1.0 - 945.0/945.0  # 1 - 1 = 0
    dp_da[0, 1] = -7560.0 * (-1.0/945.0)
    dp_da[0, 2] = 13860.0 * (-1.0/945.0)
    dp_da[0, 3] = -6930.0 * (-1.0/945.0)
    dp_da[0, 4] = 1155.0 * (-1.0/945.0)
    dp_da[0, 5] = -66.0 * (-1.0/945.0)
    dp_da[0, 6] = 1.0 * (-1.0/945.0)
    
    # d/da4
    dp_da[1, 0] = 3.0 - 945.0*(3.0/945.0) # 3 - 3 = 0
    dp_da[1, 1] = -6.0 - 7560.0 * (-3.0/945.0)
    dp_da[1, 2] = 1.0 + 13860.0 * (-3.0/945.0)
    dp_da[1, 3] = -6930.0 * (-3.0/945.0)
    dp_da[1, 4] = 1155.0 * (-3.0/945.0)
    dp_da[1, 5] = -66.0 * (-3.0/945.0)
    dp_da[1, 6] = 1.0 * (-3.0/945.0)
    
    # d/da8
    dp_da[2, 0] = 105.0 - 945.0*(105.0/945.0) # 105 - 105 = 0
    dp_da[2, 1] = -420.0 - 7560.0 * (-105.0/945.0)
    dp_da[2, 2] = 210.0 + 13860.0 * (-105.0/945.0)
    dp_da[2, 3] = -28.0 - 6930.0 * (-105.0/945.0)
    dp_da[2, 4] = 1.0 + 1155.0 * (-105.0/945.0)
    dp_da[2, 5] = -66.0 * (-105.0/945.0)
    dp_da[2, 6] = 1.0 * (-105.0/945.0)
    
    return p, dp_da

def _compute_bound_and_grad(params):
    """
    Compute bound and analytic gradient.
    """
    a0, a4, a8 = params
    p, dp_da = _coeffs_from_params(a0, a4, a8)
    
    # Q coefficients: q_k = (-1)^k p_k
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    dq_da = dp_da * signs # signs broadcast over columns? No, signs is length 7, dp_da is 3x7.
    # dq_da[i, k] = dp_da[i, k] * signs[k]
    dq_da = dp_da * signs[None, :]
    
    # Compute m_k and n_k for k=0,1,2
    # m_k = sum_{i,j} p_i p_j I_{k+i+j}
    # n_k = 2*pi * sum_{i,j} q_i q_j I_{k+i+j}
    
    m = np.zeros(3)
    n = np.zeros(3)
    dm_da = np.zeros((3, 3)) # dm_da[k, a] = d m_k / d a_a
    dn_da = np.zeros((3, 3))
    
    # Efficient calculation using matrix multiplication if possible, but 7x7 is small
    # I_matrix[k]_{i,j} = I[k+i+j]
    
    for k in range(3):
        # Build I_mat for this k
        # I_mat[i,j] = _I[k+i+j]
        # We need sum p_i p_j I_mat[i,j]
        # and sum dp_da[a,i] p_j I_mat[i,j] + p_i dp_da[a,j] I_mat[i,j]
        
        # Since I_mat is symmetric, sum p_i dp_i I_ij = p^T (dp * I_mat) = (dp * I_mat^T) dot p?
        # Let's just loop for clarity and correctness, it's only 3*7*7*3 operations
        
        for i in range(7):
            pi = p[i]
            qi = q[i]
            if pi == 0.0 and qi == 0.0:
                continue
            for j in range(7):
                pij = pi * p[j]
                qij = qi * q[j]
                idx = k + i + j
                if idx < 13:
                    m[k] += pij * _I[idx]
                    n[k] += qij * _I[idx]
                    
                    # Gradient contribution
                    # d(p_i p_j)/da_a = dp_da[a,i] p_j + p_i dp_da[a,j]
                    for a in range(3):
                        dp_i = dp_da[a, i]
                        dp_j = dp_da[a, j]
                        ddm = (dp_i * p[j] + p[i] * dp_j) * _I[idx]
                        dm_da[k, a] += ddm
                        
                        dq_i = dq_da[a, i]
                        dq_j = dq_da[a, j]
                        ddn = (dq_i * q[j] + q[i] * dq_j) * _I[idx]
                        dn_da[k, a] += ddn
                        
    n *= 2.0 * np.pi
    dn_da *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
    if m2 <= 1e-15 or n2 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient of bound
    # B = sqrt(vf * vF)
    # dB/da = 1/(2*sqrt(vf*vF)) * (vF * dvf/da + vf * dvF/da)
    
    # dvf/da = (m0 * dm2/da - m2 * dm0/da) / m0^2
    # dvF/da = (n0 * dn2/da - n2 * dn0/da) / n0^2
    
    dvf_da = (m0 * dm_da[1] - m2 * dm_da[0]) / (m0 * m0)
    dvF_da = (n0 * dn_da[1] - n2 * dn_da[0]) / (n0 * n0)
    
    grad = (vF * dvf_da + vf * dvF_da) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    return _compute_bound_and_grad(params)

def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    # Initial evaluation
    try:
        v, _ = _compute_bound_and_grad(best_params)
        if v < best_val:
            best_val = v
    except Exception:
        pass
    
    # Strategy:
    # 1. Generate diverse starting points.
    # 2. Run Basin Hopping from the best few starts.
    # 3. Polish with L-BFGS-B.
    
    rng = np.random.default_rng(42)
    
    starts = []
    # Seed neighborhood
    for _ in range(10):
        s = best_params + rng.normal(0, 0.1, 3)
        starts.append(s)
        
    # Wider exploration
    for _ in range(20):
        s = np.array([rng.uniform(-3, 2), rng.uniform(-0.5, 0.5), rng.uniform(-0.01, 0.01)])
        starts.append(s)
        
    # Structured grid
    for a0 in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5]:
        for a4 in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            a8 = 0.0005 * rng.normal()
            starts.append(np.array([a0, a4, a8]))
            
    # Evaluate starts to find good seeds for Basin Hopping
    scored_starts = []
    for s in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            val = _objective(s)
            if np.isfinite(val):
                scored_starts.append((val, s))
        except Exception:
            pass
            
    scored_starts.sort(key=lambda x: x[0])
    
    # Run Basin Hopping from top 5 starts
    for i, (val, s) in enumerate(scored_starts[:5]):
        if time.time() > _DEADLINE - 30:
            break
        try:
            # Define step function for Basin Hopping
            def step(x):
                return x + rng.normal(0, 0.1, 3)
                
            # Basin hopping options
            # min_steps: number of local minimizations
            # T: temperature
            # niter: number of Monte Carlo steps
            bh_res = basin_hopping(
                _objective_grad, 
                s, 
                stepfunc=step, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True, 'options': {'maxiter': 500, 'ftol': 1e-12}},
                niter=20, 
                T=1.0, 
                seed=42+i
            )
            
            if bh_res.fun < best_val and np.isfinite(bh_res.fun):
                best_val = bh_res.fun
                best_params = bh_res.x.copy()
                
        except Exception:
            pass
            
    # Final Polish with L-BFGS-B
    if time.time() < _DEADLINE - 5:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
            
    # One more round of perturbation and polish
    if time.time() < _DEADLINE - 2:
        try:
            for _ in range(3):
                perturbed = best_params + rng.normal(0, 1e-3, 3)
                res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                             jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass
            
    return np.array(best_params)
