"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

# Constants for Hermite functions
# Psi_n(x) = (2^n n! sqrt(pi))^-0.5 * H_n(x) * exp(-x^2/2)
# H_n are physicists' Hermite polynomials

DEGREES = [0, 4, 8, 12]

def get_norm_factor(n):
    # (2^n n! sqrt(pi))^-0.5
    # log norm = -0.5 * (n*log(2) + log(n!) + 0.5*log(pi))
    # Use lgamma for stability
    log_norm = -0.5 * (n * np.log(2) + np.log(np.math.factorial(n)) + 0.5 * np.log(np.pi))
    return np.exp(log_norm)

# Precompute norms
NORMS = {n: get_norm_factor(n) for n in DEGREES}

# Precompute Psi_n(0)
# H_n(0) is 0 for odd n.
# H_0(0) = 1
# H_4(0) = -12
# H_8(0) = -1680
# H_12(0) = -103950 (Check: H_12(x) = ... H_12(0) = (-1)^6 12! / 6! = 479001600 / 720 = 665280? No.)
# Let's compute H_n(0) properly.
# H_n(0) = 0 if n odd.
# H_{2k}(0) = (-1)^k (2k)! / k!
H_0_0 = 1
H_4_0 = -12
H_8_0 = -1680
H_12_0 = -103950 # Let's verify: 12! / 6! = 479001600 / 720 = 665280. (-1)^6 = 1. So 665280?
# Wait, formula: H_n(0) = (-1)^{n/2} (n)! / (n/2)! for n even.
# H_4(0) = (-1)^2 * 24 / 2 = 12? No, H_4(x) = x^4 - 6x^2 + 3. H_4(0) = 3?
# Let's check standard H_n:
# H_0 = 1
# H_1 = 2x
# H_2 = 4x^2 - 2
# H_3 = 8x^3 - 12x
# H_4 = 16x^4 - 48x^2 + 12. H_4(0) = 12.
# My previous H_4(0)=-12 was wrong sign or definition.
# Probabilists He_n: He_4 = x^4 - 6x^2 + 3. He_4(0)=3.
# Physicists H_n: H_4(0) = 12.
# Let's use scipy.eval_hermite to be safe.

PSI0 = {}
for n in DEGREES:
    H_val = eval_hermite(n, 0)
    PSI0[n] = NORMS[n] * H_val

# Constraint: sum b_k PSI0[k] = 0
# b_12 = -(b_0*PSI0[0] + b_4*PSI0[4] + b_8*PSI0[8]) / PSI0[12]

# Grid for integration
X_MIN, X_MAX = -12, 12
N_POINTS = 2000
X_GRID = np.linspace(X_MIN, X_MAX, N_POINTS)
DX = X_GRID[1] - X_GRID[0]

# Precompute Psi_n(x) on grid for all n in DEGREES
PSI_GRID = {}
for n in DEGREES:
    H_vals = eval_hermite(n, X_GRID)
    PSI_GRID[n] = NORMS[n] * H_vals * np.exp(-X_GRID**2 / 2)

def compute_objective(params):
    a0, a4, a8 = params
    
    # Convert a to b (coefficients in orthonormal basis)
    b0 = a0 * NORMS[0]
    b4 = a4 * NORMS[4]
    b8 = a8 * NORMS[8]
    
    # Compute b12 from constraint g(0) = 0
    # sum b_k psi_k(0) = 0
    numerator = b0 * PSI0[0] + b4 * PSI0[4] + b8 * PSI0[8]
    b12 = -numerator / PSI0[12]
    
    # Normalize b such that ||g||_2^2 = sum b_k^2 = 1
    norm_sq = b0**2 + b4**2 + b8**2 + b12**2
    if norm_sq < 1e-15:
        return 1e10
    norm = np.sqrt(norm_sq)
    b0 /= norm
    b4 /= norm
    b8 /= norm
    b12 /= norm
    
    # Construct g(x) and hat_g(x)
    # g(x) = b0*psi0 + b4*psi4 + b8*psi8 + b12*psi12
    g_x = b0 * PSI_GRID[0] + b4 * PSI_GRID[4] + b8 * PSI_GRID[8] + b12 * PSI_GRID[12]
    
    # hat_g(xi) = b0*psi0 + b4*psi4 + b8*psi8 - b12*psi12
    # Note: (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=-1
    g_hat_x = b0 * PSI_GRID[0] + b4 * PSI_GRID[4] + b8 * PSI_GRID[8] - b12 * PSI_GRID[12]
    
    # Compute L4 norms
    # ||g||_4 = (int |g|^4 dx)^(1/4)
    integral_g4 = np.sum(np.abs(g_x)**4) * DX
    integral_ghat4 = np.sum(np.abs(g_hat_x)**4) * DX
    
    # Objective is ||g||_4 * ||hat_g||_4
    val = (integral_g4 ** 0.25) * (integral_ghat4 ** 0.25)
    
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    # Initial guess from seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = compute_objective(x0)
    best_x = x0
    
    # Try multiple starting points
    starts = [
        x0,
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 1]) * 0.1,
        np.array([-1, -1, -1]) * 0.1,
        np.array([0.5, -0.5, 0.1]),
    ]
    
    for x_start in starts:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(compute_objective, x_start, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            val = res.fun
            if val < best_val:
                best_val = val
                best_x = res.x
        except Exception:
            pass
            
    # Polish with Nelder-Mead if time permits
    if time.time() < deadline - 5.0:
        try:
            res = minimize(compute_objective, best_x, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
