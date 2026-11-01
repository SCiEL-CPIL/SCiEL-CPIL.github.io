"""Combined program: exact polynomial-moment formulation (inspiration) + robust multi-start search (current)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # Safety margin before 210s kill

def _precompute_I():
    """Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..13."""
    sp = math.sqrt(math.pi)
    I = np.zeros(14)
    for n in range(14):
        # (2n)! / (4^n * n!) * sqrt(pi)
        if n == 0:
            I[n] = sp
        else:
            # Use log-space to avoid overflow for larger n, though n<=13 is fine
            log_val = 0.0
            for m in range(1, 2 * n + 1):
                log_val += math.log(m)
            log_val -= n * math.log(4.0)
            for m in range(1, n + 1):
                log_val -= math.log(m)
            log_val += 0.5 * math.log(math.pi)
            I[n] = math.exp(log_val)
    return I

_I = _precompute_I()

def _bound_and_grad(params):
    """
    Compute the uncertainty bound sqrt(vf * vF) and its analytic gradient.
    
    g(x) = P(x) * exp(-x^2/2)
    P(x) = sum_{k=0}^6 p_k x^{2k}
    
    Constraints: g(0) = 0 => p0 = 0.
    We parameterize by (a0, a4, a8) where P is a linear combination of H0, H4, H8, H12.
    H0 = 1
    H4 = x^4 - 6x^2 + 3
    H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    p0 = a0 + 3a4 + 105a8 + 945a12 = 0  => a12 = -(a0 + 3a4 + 105a8)/945
    
    We compute moments m_k = int x^{2k} |P|^2 e^{-x^2} dx for k=0,1,2
    And n_k = int xi^{2k} |Q|^2 e^{-xi^2} dx * 2*pi for k=0,1,2
    where Q(xi) = P(i*xi) essentially (Fourier transform of P(x)e^{-x^2/2} relates to P(i xi) e^{-xi^2/2})
    Specifically, if g(x) = P(x) e^{-x^2/2}, then G(xi) = sqrt(2*pi) * P(-i xi) e^{-xi^2/2} (up to convention)
    Since P is even with real coeffs, P(-i xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k} = Q(xi)
    |G|^2 = 2*pi |Q|^2 e^{-xi^2}
    
    vf = m2/m0, vF = n2/n0
    bound = sqrt(vf * vF)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Coefficients of P(x) = sum p_k x^{2k}
    # p_k = a0*coeff_H0 + a4*coeff_H4 + a8*coeff_H8 + a12*coeff_H12
    # a12 is determined by p0=0.
    
    # H0 coeffs: [1, 0, 0, 0, 0, 0, 0]
    # H4 coeffs: [3, -6, 1, 0, 0, 0, 0]
    # H8 coeffs: [105, -420, 210, -28, 1, 0, 0]
    # H12 coeffs: [945, -7560, 13860, -6930, 1155, -66, 1]
    
    # p0 = a0 + 3a4 + 105a8 + 945a12 = 0
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12  # Should be 0
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    
    # Compute m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            if pij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pij * _I[idx]
    
    # Q(xi) coeffs: q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0) ** k)
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    factor = 2.0 * math.pi
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            if qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += factor * qij * _I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = math.sqrt(vf * vF)
    
    # Gradient calculation
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    
    # Derivatives of p_k with respect to a0, a4, a8
    # a12 = -(a0 + 3a4 + 105a8)/945
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    
    da12_da = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c = coeff_k_c + coeff_k_12 * da12_da_c
    # coeff_k_c is the coefficient of x^{2k} in H_c (where c is index 0,1,2 corresponding to deg 0,4,8)
    # coeff_k_12 is the coefficient of x^{2k} in H12
    
    H_coeffs = np.array([
        [1, 0, 0, 0, 0, 0, 0],      # H0
        [3, -6, 1, 0, 0, 0, 0],     # H4
        [105, -420, 210, -28, 1, 0, 0], # H8
        [945, -7560, 13860, -6930, 1155, -66, 1] # H12
    ])
    
    grad = np.zeros(3)
    
    for c in range(3):
        # dp = H_coeffs[c] + H_coeffs[3] * da12_da[c]
        dp = H_coeffs[c] + H_coeffs[3] * da12_da[c]
        
        # dm_k/da_c = 2 * sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s
        
        # dq_k/da_c = dp_k * (-1)^k
        dq = dp * np.array([1, -1, 1, -1, 1, -1, 1])
        
        # dn_k/da_c = 2 * 2*pi * sum_i dq_i * sum_j q_j I[k+i+j]
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += q[j] * _I[idx]
                dn[k] += 2.0 * factor * dqi * s
        
        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Strict internal deadline
    
    # Initial guesses
    # The seed program used: -1.13, 0.04, 0.000308642
    # Let's use a set of diverse starts
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.001]),
        np.array([-2.0, 0.1, 0.002]),
        np.array([-0.5, 0.02, 0.0005]),
        np.array([1.0, -0.05, 0.001]),
        np.array([-3.0, 0.2, 0.005]),
        np.array([0.5, -0.02, 0.0005]),
    ]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        starts.append(rng.normal(0, 1.0, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    # Phase 2: Refine best result with high precision
    if time.time() < deadline - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Phase 3: Nelder-Mead polish
        if time.time() < deadline - 5:
            try:
                res = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    return best_params

# EVOLVE-BLOCK-END
