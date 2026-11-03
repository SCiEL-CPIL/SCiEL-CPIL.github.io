import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma

_START = time.time()
_DEADLINE = _START + 170.0  # safety margin under 210s

def _time_left():
    return _DEADLINE - time.time()

# Precompute gamma values for _poly4_integral
# d has length 13 (indices 0 to 12)
_GAMMA_VALS = gamma(np.arange(13) + 0.5) / (2.0 ** (np.arange(13) + 0.5))

# Hermite polynomials (physicists') even ones, coefficients for x^0..x^12
# H0 = 1
# H4 = 16x^4 - 48x^2 + 12
# H8 = 256x^8 - 1280x^6 + 2688x^4 - 1344x^2 + 112
# H12 = 4096x^12 - 39936x^10 + 153600x^8 - 282240x^6 + 253440x^4 - 96768x^2 + 9216
# Coefficients (index = power)
_H0 = {0:1}
_H4 = {2:-48, 4:16, 0:12}
_H8 = {2:-1344, 4:2688, 6:-1280, 8:256}
_H12 = {0:9216, 2:-96768, 4:253440, 6:-282240, 8:153600, 10:-39936, 12:4096}

def _b_coeffs(c0, c4, c8):
    c12 = -(c0 + 12*c4 + 112*c8) / 9216.0
    b = np.zeros(7)  # b[0]=b0 (x^0), b[1]=b2 (x^2), ..., b[6]=b12 (x^12)
    # x^0
    b[0] = c0*1 + c4*12 + c8*112 + c12*9216  # = 0
    # x^2
    b[1] = c4*(-48) + c8*(-1344) + c12*(-96768)
    # x^4
    b[2] = c4*16 + c8*2688 + c12*253440
    # x^6
    b[3] = c8*(-1280) + c12*(-282240)
    # x^8
    b[4] = c8*256 + c12*153600
    # x^10
    b[5] = c12*(-39936)
    # x^12
    b[6] = c12*4096
    return b

def _poly4_integral(coeffs_u):
    # coeffs_u[m] = coeff of u^m, u=x^2. Compute ∫ (Σ coeffs_u[m] u^m)^4 e^{-2x^2} dx
    # = Σ d[n] Γ(n+1/2)/2^{n+1/2}, d = coeffs_u^4 (convolution)
    d = np.convolve(np.convolve(coeffs_u, coeffs_u), np.convolve(coeffs_u, coeffs_u))
    # d has length 13
    return float(np.dot(d, _GAMMA_VALS))

# Precompute P_k polynomials (in ξ), P_k[i] = coeff of ξ^i
def _build_P():
    P = [[1.0]]  # P_0
    for k in range(12):
        Pk = P[k]
        # deriv
        dpk = [i*Pk[i] for i in range(1, len(Pk))]
        # x*Pk
        xPk = [0.0] + Pk
        # P_{k+1} = dpk - xPk
        L = max(len(dpk), len(xPk))
        dpk = dpk + [0.0]*(L-len(dpk))
        xPk = xPk + [0.0]*(L-len(xPk))
        Pk1 = [dpk[i] - xPk[i] for i in range(L)]
        P.append(Pk1)
    return P

_P = _build_P()

def _R_coeffs_u(b):
    # R(ξ) = Σ_{j=0}^{6} b[2j index] ... b is indexed b[m] = coeff of x^{2m}
    # R(ξ) = Σ_{j=0}^{6} b[j] * (-1)^j * P[2j](ξ)
    R = np.zeros(13)  # up to ξ^12
    for j in range(7):
        sign = (-1.0)**j
        Pk = np.array(_P[2*j])
        R[:len(Pk)] += b[j]*sign*Pk
    # R is even; extract even coefficients -> u form
    R_u = R[0::2]  # R[0],R[2],...,R[12]
    return R_u

def _raw_value(c0, c4, c8):
    b = _b_coeffs(c0, c4, c8)
    A_g = _poly4_integral(b)
    R_u = _R_coeffs_u(b)
    A_gf = _poly4_integral(R_u)
    return A_g * A_gf

_SEED = np.array([-113/100, 1/25, 1/3240])
_RAW_SEED = _raw_value(*_SEED)

def _value(c0, c4, c8):
    return _raw_value(c0, c4, c8) / _RAW_SEED

def _objective(x):
    return _value(x[0], x[1], x[2])

def construct():
    global _START, _DEADLINE
    best_x = _SEED.copy()
    best_v = _value(*_SEED)  # = 1.0
    
    # Tighter bounds based on seed magnitude
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    # Start with the seed
    starts = [best_x]
    
    # Add some random perturbations around the seed
    rng = np.random.default_rng(42)
    for _ in range(20):
        perturbation = rng.normal(0, 0.1, 3)
        starts.append(best_x + perturbation)
        
    # Run local optimization from each start
    for x0 in starts:
        if _time_left() < 10:
            break
        try:
            # L-BFGS-B is good for smooth functions with bounds
            res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.fun < best_v:
                best_v = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish with Powell (derivative-free, robust)
    if _time_left() > 10:
        try:
            res = minimize(_objective, best_x, method='Powell', options={'maxiter': 2000, 'xtol': 1e-15, 'ftol': 1e-15})
            if res.fun < best_v:
                best_v = res.fun
                best_x = res.x
        except Exception:
            pass
            
    return np.array([best_x[0], best_x[1], best_x[2]])
