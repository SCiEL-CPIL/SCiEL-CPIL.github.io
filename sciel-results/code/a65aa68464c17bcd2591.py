"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Pre-compute grid for numerical integration
# The function g(x) decays as exp(-x^2), so [-10, 10] is sufficient for high precision
_x_grid = np.linspace(-10.0, 10.0, 10001)
_dx = _x_grid[1] - _x_grid[0]

# Pre-compute Hermite polynomials on the grid
# H_0, H_4, H_8, H_12
# Using physicists' Hermite polynomials: H_n(x)
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 4480x^6 + 20160x^4 - 26880x^2 + 1680
# H_12 = 4096x^12 - 147456x^10 + 2419200x^8 - 15482880x^6 + 46489600x^4 - 58720224x^2 + 277200

_x2 = _x_grid**2
_x4 = _x2**2
_x6 = _x4 * _x2
_x8 = _x4**2
_x10 = _x8 * _x2
_x12 = _x8 * _x4

_H0 = np.ones_like(_x_grid)
_H4 = 16 * _x4 - 48 * _x2 + 12
_H8 = 256 * _x8 - 4480 * _x6 + 20160 * _x4 - 26880 * _x2 + 1680
_H12 = 4096 * _x12 - 147456 * _x10 + 2419200 * _x8 - 15482880 * _x6 + 46489600 * _x4 - 58720224 * _x2 + 277200

# Gaussian envelope exp(-x^2/2)
_gauss = np.exp(-0.5 * _x2)

# Values at x=0 for the constraint g(0)=0
# H_0(0) = 1
# H_4(0) = 12
# H_8(0) = 1680
# H_12(0) = 277200
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 277200.0


def _compute_raw_value(c0, c4, c8):
    """
    Compute the raw uncertainty product upper bound related to C4.
    The bound is proportional to ||g||_4^2 / ||g||_2.
    Since g(0)=0, c12 is determined by c0, c4, c8.
    """
    # Determine c12 such that g(0) = 0
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    c12 = -(c0 * _H0_0 + c4 * _H4_0 + c8 * _H8_0) / _H12_0
    
    # Construct g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    poly = c0 * _H0 + c4 * _H4 + c8 * _H8 + c12 * _H12
    g = poly * _gauss
    
    # Compute norms
    # ||g||_2^2 = integral |g|^2 dx
    norm2_sq = np.sum(g**2) * _dx
    
    if norm2_sq <= 0:
        return 1e10
    
    # ||g||_4^4 = integral |g|^4 dx
    # We want a quantity that scales like A(f)A(F(f)).
    # In the context of Lp uncertainty, the constant C4 is often defined via
    # ||f||_4 ||\hat{f}||_4 / ||f||_2.
    # Due to Plancherel and scaling, ||\hat{f}||_4 is proportional to ||f||_4.
    # Specifically, if we use the unitary Fourier transform, ||\hat{f}||_4 = ||f||_4.
    # So the product is ||f||_4^2 / ||f||_2 * (constant factors).
    # The "raw value" to minimize is likely ||g||_4^2 / ||g||_2.
    
    norm4_4 = np.sum(g**4) * _dx
    norm4 = norm4_4 ** 0.25
    norm4_sq = norm4 ** 2 # which is norm4_4 ** 0.5
    
    norm2 = np.sqrt(norm2_sq)
    
    # The score is the upper bound. Lower is better.
    # The seed program returns a value that results in score 1.0.
    # We return the raw bound value.
    return norm4_sq / norm2


def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin for 180s limit
    
    # Seed values from Goncalves et al. 2017
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    # Evaluate seed
    try:
        seed_val = _compute_raw_value(*seed)
    except Exception:
        return seed
    
    best_x = seed.copy()
    best_val = seed_val
    
    # Define objective for optimizer
    def objective(x):
        return _compute_raw_value(x[0], x[1], x[2])
    
    # 1. Nelder-Mead from seed
    try:
        res = minimize(objective, best_x, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        return best_x
    
    # 2. BFGS polish
    try:
        res = minimize(objective, best_x, method='BFGS',
                       options={'maxiter': 1000, 'gtol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # 3. Try a few random restarts if time permits
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb around seed or best_x
            perturbation = rng.standard_normal(3) * 0.1
            x0 = best_x + perturbation
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                continue
    
    return best_x
# EVOLVE-BLOCK-END
