import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200

# Precompute moments I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_moments():
    sp = np.sqrt(np.pi)
    I = np.zeros(16)
    I[0] = sp
    for n in range(1, 16):
        I[n] = I[n-1] * (2*n - 1) / (4*n)
    return I

_I_TABLE = _precompute_moments()
_J_TABLE = _I_TABLE / np.sqrt(2.0)
_K_TABLE = _I_TABLE / 2.0

# Precompute Hermite polynomial coefficients (physicists' convention)
# Coefficients are indexed by power: coeff[i] * x^i
def _hermite_coeffs(n):
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif n == 4:
        return np.array([12.0, 0.0, -48.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif n == 8:
        return np.array([4032.0, 0.0, -13824.0, 0.0, 11520.0, 0.0, -3072.0, 0.0, 256.0, 0.0, 0.0, 0.0, 0.0])
    elif n == 12:
        return np.array([10321920.0, 0.0, -17694720.0, 0.0, 15975168.0, 0.0, -5996736.0, 0.0, 1032192.0, 0.0, -98304.0, 0.0, 4096.0])
    else:
        raise ValueError(f"Unsupported Hermite degree: {n}")

_H0_coeffs = _hermite_coeffs(0)
_H4_coeffs = _hermite_coeffs(4)
_H8_coeffs = _hermite_coeffs(8)
_H12_coeffs = _hermite_coeffs(12)

_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 4032.0
_H12_0 = 10321920.0

def _get_c12(c0, c4, c8):
    return -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0

def _compute_polynomial_moments(c0, c4, c8):
    c12 = _get_c12(c0, c4, c8)
    
    p_coeffs = np.zeros(13)
    p_coeffs += c0 * _H0_coeffs
    p_coeffs += c4 * _H4_coeffs
    p_coeffs += c8 * _H8_coeffs
    p_coeffs += c12 * _H12_coeffs
    
    # Only even powers are non-zero, so we can optimize
    even_powers = np.array([0, 2, 4, 6, 8, 10, 12])
    p_even = p_coeffs[even_powers]
    
    norm2_sq = 0.0
    for i_idx in range(7):
        pi_val = p_even[i_idx]
        if pi_val == 0:
            continue
        for j_idx in range(7):
            pj_val = p_even[j_idx]
            if pj_val == 0:
                continue
            total_power = even_powers[i_idx] + even_powers[j_idx]
            n = total_power // 2
            if n < len(_I_TABLE):
                norm2_sq += pi_val * pj_val * _I_TABLE[n]
    
    norm4_4 = 0.0
    for i_idx in range(7):
        pi_val = p_even[i_idx]
        if pi_val == 0:
            continue
        for j_idx in range(7):
            pj_val = p_even[j_idx]
            if pj_val == 0:
                continue
            for k_idx in range(7):
                pk_val = p_even[k_idx]
                if pk_val == 0:
                    continue
                for l_idx in range(7):
                    pl_val = p_even[l_idx]
                    if pl_val == 0:
                        continue
                    total_power = even_powers[i_idx] + even_powers[j_idx] + even_powers[k_idx] + even_powers[l_idx]
                    n = total_power // 2
                    if n < len(_J_TABLE):
                        norm4_4 += pi_val * pj_val * pk_val * pl_val * _J_TABLE[n]
    
    return norm2_sq, norm4_4

def _compute_fourier_moments(c0, c4, c8):
    c12 = _get_c12(c0, c4, c8)
    
    q_coeffs = np.zeros(13)
    q_coeffs += c0 * _H0_coeffs
    q_coeffs += c4 * _H4_coeffs
    q_coeffs += c8 * _H8_coeffs
    q_coeffs += (-c12) * _H12_coeffs
    
    even_powers = np.array([0, 2, 4, 6, 8, 10, 12])
    q_even = q_coeffs[even_powers]
    
    norm4_hat_4 = 0.0
    for i_idx in range(7):
        qi_val = q_even[i_idx]
        if qi_val == 0:
            continue
        for j_idx in range(7):
            qj_val = q_even[j_idx]
            if qj_val == 0:
                continue
            for k_idx in range(7):
                qk_val = q_even[k_idx]
                if qk_val == 0:
                    continue
                for l_idx in range(7):
                    ql_val = q_even[l_idx]
                    if ql_val == 0:
                        continue
                    total_power = even_powers[i_idx] + even_powers[j_idx] + even_powers[k_idx] + even_powers[l_idx]
                    n = total_power // 2
                    if n < len(_K_TABLE):
                        norm4_hat_4 += qi_val * qj_val * qk_val * ql_val * _K_TABLE[n]
    
    norm4_hat_4 *= (2 * np.pi) ** 2
    
    return norm4_hat_4

def _bound_and_grad(coeffs):
    c0, c4, c8 = coeffs
    
    norm2_sq, norm4_4 = _compute_polynomial_moments(c0, c4, c8)
    norm4_hat_4 = _compute_fourier_moments(c0, c4, c8)
    
    if norm2_sq < 1e-10 or norm4_4 < 1e-10 or norm4_hat_4 < 1e-10:
        return 1e10, np.zeros(3)
    
    R = (norm4_4 ** 0.25 * norm4_hat_4 ** 0.25) / norm2_sq
    
    # Use central differences for gradient
    eps = 1e-6
    grad = np.zeros(3)
    for i in range(3):
        c_plus = coeffs.copy()
        c_plus[i] += eps
        c_minus = coeffs.copy()
        c_minus[i] -= eps
        
        n2_p, n4_p = _compute_polynomial_moments(c_plus[0], c_plus[1], c_plus[2])
        nh_p = _compute_fourier_moments(c_plus[0], c_plus[1], c_plus[2])
        if n2_p > 1e-10 and n4_p > 1e-10 and nh_p > 1e-10:
            R_plus = (n4_p ** 0.25 * nh_p ** 0.25) / n2_p
        else:
            R_plus = 1e10
        
        n2_m, n4_m = _compute_polynomial_moments(c_minus[0], c_minus[1], c_minus[2])
        nh_m = _compute_fourier_moments(c_minus[0], c_minus[1], c_minus[2])
        if n2_m > 1e-10 and n4_m > 1e-10 and nh_m > 1e-10:
            R_minus = (n4_m ** 0.25 * nh_m ** 0.25) / n2_m
        else:
            R_minus = 1e10
        
        grad[i] = (R_plus - R_minus) / (2 * eps)
    
    return R, grad

def _objective(coeffs):
    val, _ = _bound_and_grad(coeffs)
    return val

def _objective_grad(coeffs):
    return _bound_and_grad(coeffs)

def construct():
    start_time = time.time()
    deadline = start_time + 200
    
    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.0003])
    
    best_val = _objective(x0)
    best_coeffs = x0.copy()
    
    # Try L-BFGS-B with analytic gradient
    try:
        res = minimize(_objective, x0, jac=_objective_grad, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_coeffs
    
    # Try multiple random restarts with L-BFGS-B
    rng = np.random.RandomState(42)
    for i in range(20):
        if time.time() > deadline - 30:
            break
        # Generate random starting points with varying scales
        scale = 0.1 * (1 + i * 0.5)
        noise = rng.normal(0, scale, 3)
        x_start = x0 + noise
        
        try:
            res_r = minimize(_objective, x_start, jac=_objective_grad, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if res_r.fun < best_val:
                best_val = res_r.fun
                best_coeffs = res_r.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 10:
        try:
            res_final = minimize(_objective, best_coeffs, jac=_objective_grad, method='L-BFGS-B', 
                               options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass
    
    return best_coeffs
