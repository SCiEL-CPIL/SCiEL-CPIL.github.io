import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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

# Precompute Hermite polynomial coefficients (physicists' convention)
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13824x^2 + 4032
# H_12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 5996736x^6 + 15975168x^4 - 17694720x^2 + 10321920

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
    
    norm2_sq = 0.0
    for i in range(13):
        if p_coeffs[i] == 0:
            continue
        for j in range(13):
            if p_coeffs[j] == 0:
                continue
            total_power = i + j
            if total_power % 2 == 0:
                n = total_power // 2
                if n < len(_I_TABLE):
                    norm2_sq += p_coeffs[i] * p_coeffs[j] * _I_TABLE[n]
    
    J_TABLE = _I_TABLE / np.sqrt(2.0)
    
    norm4_4 = 0.0
    for i in range(13):
        if p_coeffs[i] == 0:
            continue
        for j in range(13):
            if p_coeffs[j] == 0:
                continue
            for k in range(13):
                if p_coeffs[k] == 0:
                    continue
                for l in range(13):
                    if p_coeffs[l] == 0:
                        continue
                    total_power = i + j + k + l
                    if total_power % 2 == 0:
                        n = total_power // 2
                        if n < len(J_TABLE):
                            norm4_4 += p_coeffs[i] * p_coeffs[j] * p_coeffs[k] * p_coeffs[l] * J_TABLE[n]
    
    return norm2_sq, norm4_4

def _compute_fourier_moments(c0, c4, c8):
    c12 = _get_c12(c0, c4, c8)
    
    q_coeffs = np.zeros(13)
    q_coeffs += c0 * _H0_coeffs
    q_coeffs += c4 * _H4_coeffs
    q_coeffs += c8 * _H8_coeffs
    q_coeffs += (-c12) * _H12_coeffs
    
    K_TABLE = _I_TABLE / 2.0
    
    norm4_hat_4 = 0.0
    for i in range(13):
        if q_coeffs[i] == 0:
            continue
        for j in range(13):
            if q_coeffs[j] == 0:
                continue
            for k in range(13):
                if q_coeffs[k] == 0:
                    continue
                for l in range(13):
                    if q_coeffs[l] == 0:
                        continue
                    total_power = i + j + k + l
                    if total_power % 2 == 0:
                        n = total_power // 2
                        if n < len(K_TABLE):
                            norm4_hat_4 += q_coeffs[i] * q_coeffs[j] * q_coeffs[k] * q_coeffs[l] * K_TABLE[n]
    
    norm4_hat_4 *= (2 * np.pi) ** 2
    
    return norm4_hat_4

def _bound_and_grad(coeffs):
    c0, c4, c8 = coeffs
    
    norm2_sq, norm4_4 = _compute_polynomial_moments(c0, c4, c8)
    norm4_hat_4 = _compute_fourier_moments(c0, c4, c8)
    
    if norm2_sq < 1e-10 or norm4_4 < 1e-10 or norm4_hat_4 < 1e-10:
        return 1e10, np.zeros(3)
    
    R = (norm4_4 ** 0.25 * norm4_hat_4 ** 0.25) / norm2_sq
    
    eps = 1e-7
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
    
    # Try SLSQP with analytic gradient
    try:
        res = minimize(_objective, x0, jac=_objective_grad, method='SLSQP', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'maxls': 50})
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_coeffs
    
    # Try a few random restarts
    for i in range(8):
        if time.time() > deadline - 30:
            break
        np.random.seed(42 + i)
        noise = np.random.normal(0, 0.5, 3)
        x_start = x0 + noise
        try:
            res_r = minimize(_objective, x_start, jac=_objective_grad, method='SLSQP', 
                           options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 30})
            if res_r.fun < best_val:
                best_val = res_r.fun
                best_coeffs = res_r.x
        except Exception:
            pass
    
    return best_coeffs
