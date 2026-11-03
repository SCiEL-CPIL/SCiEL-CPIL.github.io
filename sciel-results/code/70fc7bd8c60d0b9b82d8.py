# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0  # seconds to spare before the 210s kill

def _remaining():
    return _START_TIME + 210.0 - _DEADLINE_MARGIN - time.time()

# Pre-compute Gauss-Hermite nodes and weights for high accuracy
# n=200 is usually sufficient for double precision with smooth polynomials
_N_QUAD = 200
_nodes, _weights = np.polynomial.hermite.hermgauss(_N_QUAD)

# Pre-compute Hermite polynomial evaluation using numpy.polynomial.hermite
# We need to evaluate H_0, H_4, H_8, H_12 at the nodes.
# H_n(x) coefficients in physicist's convention are handled by numpy.polynomial.hermite.hermvander
# But it's easier to just use the recurrence or explicit coefficients if needed.
# Actually, numpy.polynomial.hermite.hermvander returns the values of Hermite polynomials.
# The default is physicist's Hermite? No, numpy uses physicist's?
# Let's check: numpy.polynomial.hermite.hermvander(x, deg)
# The documentation says "The Hermite series is ... sum_{n=0}^{deg} c_n He_n(x)" where He_n are probabilists' Hermite?
# No, numpy.polynomial.hermite uses Physicist's Hermite polynomials H_n(x).
# Let's verify: H_0=1, H_1=2x, H_2=4x^2-2.
# numpy.polynomial.hermite.hermval(0, [1, 0, 0]) -> 1.
# numpy.polynomial.hermite.hermval(1, [0, 1, 0]) -> 2.
# So yes, it's physicist's.

# We want P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
# We can precompute the matrix of H_k(x_i) for k in [0, 4, 8, 12]
# M[i, j] = H_{deg_j}(x_i)
_degs = [0, 4, 8, 12]
_M = np.zeros((_N_QUAD, 4))
for j, deg in enumerate(_degs):
    _M[:, j] = np.polynomial.hermite.hermval(_nodes, [0]*deg + [1]) # This is wrong, hermval expects coefficients.
    
# Correct way to get H_n(x):
# H_n(x) is the n-th Hermite polynomial.
# We can use the recurrence or just evaluate the specific polynomials.
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 4480x^6 + 26880x^4 - 53760x^2 + 1680
# H_12 = 4096x^12 - 147456x^10 + 2419200x^8 - 16329600x^6 + 43224000x^4 - 40828800x^2 + 665280

# Let's construct the coefficient vectors for these specific degrees to use hermval efficiently or just manual evaluation.
# Manual evaluation with Horner's rule is very fast.
def _eval_hermite(x, n):
    if n == 0: return np.ones_like(x)
    if n == 4: return 16*x**4 - 48*x**2 + 12
    if n == 8: return 256*x**8 - 4480*x**6 + 26880*x**4 - 53760*x**2 + 1680
    if n == 12: return 4096*x**12 - 147456*x**10 + 2419200*x**8 - 16329600*x**6 + 43224000*x**4 - 40828800*x**2 + 665280
    raise ValueError("Unsupported degree")

_H0_nodes = _eval_hermite(_nodes, 0)
_H4_nodes = _eval_hermite(_nodes, 4)
_H8_nodes = _eval_hermite(_nodes, 8)
_H12_nodes = _eval_hermite(_nodes, 12)

# Values at x=0 for c12 calculation
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

def _compute_upper_bound(c0, c4, c8):
    c12 = -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0
    
    # Evaluate P at nodes
    P_nodes = c0 * _H0_nodes + c4 * _H4_nodes + c8 * _H8_nodes + c12 * _H12_nodes
    
    # ||g||_2^2 = sum( P_nodes^2 * weights )
    norm_sq = np.sum(P_nodes**2 * _weights)
    
    # ||F(g)||_2^2 = 2*pi * sum( |P(i*nodes)|^2 * weights )
    # Note: The quadrature is for e^{-x^2}.
    # We need integral of |P(i xi)|^2 e^{-xi^2} dxi.
    # Let y = xi. We evaluate P(i*y) at y = nodes.
    # P(i*y) = c0*H0(iy) + c4*H4(iy) + ...
    # H_n(iy) = i^n H_n(y) for even n?
    # H_n(x) is even for even n. H_n(ix) = i^n H_n(x) if n is even?
    # H_0(ix) = 1.
    # H_4(ix) = 16(iy)^4 - 48(iy)^2 + 12 = 16y^4 + 48y^2 + 12.
    # H_4(y) = 16y^4 - 48y^2 + 12.
    # So H_4(iy) != H_4(y).
    # We must evaluate the polynomials at imaginary arguments.
    # P(iy) = c0*1 + c4*(16y^4 + 48y^2 + 12) + c8*(256y^8 - 4480y^6 + 26880y^4 + 53760y^2 + 1680) + ...
    # It is safer to just evaluate the polynomial coefficients at 1j*_nodes.
    
    # Let's precompute the complex values of the basis functions at 1j*nodes
    _i_nodes = 1j * _nodes
    _H0_i = _eval_hermite(_i_nodes, 0)
    _H4_i = _eval_hermite(_i_nodes, 4)
    _H8_i = _eval_hermite(_i_nodes, 8)
    _H12_i = _eval_hermite(_i_nodes, 12)
    
    P_i_nodes = c0 * _H0_i + c4 * _H4_i + c8 * _H8_i + c12 * _H12_i
    abs2_P_i = np.abs(P_i_nodes)**2
    
    norm_f_sq = 2 * np.pi * np.sum(abs2_P_i * _weights)
    
    return norm_sq * norm_f_sq

def construct():
    t_start = time.time()
    
    # Seed value
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = seed.copy()
    best_val = _compute_upper_bound(best[0], best[1], best[2])
    
    def objective(params):
        if time.time() > t_start + 210.0 - _DEADLINE_MARGIN:
            return 1e30
        return _compute_upper_bound(params[0], params[1], params[2])
    
    start_points = [
        seed,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-1.5, 0.08, 0.001]),
        np.array([-0.8, 0.03, 0.0002]),
    ]
    
    for sp in start_points:
        if time.time() > t_start + 210.0 - _DEADLINE_MARGIN:
            break
        
        try:
            result = minimize(
                objective,
                sp,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best = result.x.copy()
        except Exception:
            continue
    
    # Final refinement
    if time.time() < t_start + 210.0 - _DEADLINE_MARGIN:
        try:
            result = minimize(
                objective,
                best,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 100}
            )
            if result.fun < best_val:
                best_val = result.fun
                best = result.x.copy()
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
