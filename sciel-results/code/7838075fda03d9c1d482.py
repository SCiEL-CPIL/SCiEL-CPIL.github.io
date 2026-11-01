"""
Optimized uncertainty bound minimization using Quantum Harmonic Oscillator basis.
Treats the Hermite-Gaussian combination as a state in the subspace spanned by |0>, |4>, |8>, |12>.
"""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 185

# Precompute the HO basis indices
_N_LEVELS = np.array([0, 4, 8, 12])
_N = 4

# Precompute X2 and P2 matrices
# X2[i,j] = <n_i | x^2 | n_j>
# P2[i,j] = <n_i | p^2 | n_j> = 2(n_i + 0.5) delta_ij - X2[i,j]
_X2 = np.zeros((_N, _N))
_P2 = np.zeros((_N, _N))

for i in range(_N):
    for j in range(_N):
        ni = _N_LEVELS[i]
        nj = _N_LEVELS[j]
        if i == j:
            _X2[i, j] = ni + 0.5
        elif abs(ni - nj) == 2:
            # Ensure we pick the correct formula based on which is larger
            n_high = max(ni, nj)
            n_low = min(ni, nj)
            # <n_low | x^2 | n_high> = sqrt((n_high+1)(n_high+2)/4)
            # <n_high | x^2 | n_low> = sqrt((n_high)(n_high-1)/4) -- Wait.
            # x = (a + a^dagger) / sqrt(2)
            # x^2 |n> = (a^2 + a^dagger a + a a^dagger + a^dagger 2) / 2 |n>
            # a^2 |n> = sqrt(n(n-1)) |n-2>
            # a^dagger 2 |n> = sqrt((n+1)(n+2)) |n+2>
            # a a^dagger |n> = (n+1) |n>
            # a^dagger a |n> = n |n>
            # So x^2 |n> = 1/2 [ sqrt(n(n-1)) |n-2> + (2n+1) |n> + sqrt((n+1)(n+2)) |n+2> ]
            # <m| x^2 |n> is non-zero if m=n, n+2, n-2.
            # If ni = nj + 2 (i is higher), then <ni| x^2 |nj> = 1/2 * sqrt((nj+1)(nj+2))
            # If nj = ni + 2 (j is higher), then <ni| x^2 |nj> = 1/2 * sqrt(ni(ni-1))
            
            if ni > nj:
                # i is higher, j is lower. <ni| x^2 |nj>
                # ni = nj + 2.
                # Term from a^dagger 2 acting on |nj> gives |nj+2> = |ni>.
                # Coeff is sqrt((nj+1)(nj+2))/2
                _X2[i, j] = 0.5 * np.sqrt((nj + 1) * (nj + 2))
            else:
                # j is higher, i is lower. <ni| x^2 |nj>
                # ni = nj - 2.
                # Term from a^2 acting on |nj> gives |nj-2> = |ni>.
                # Coeff is sqrt(nj(nj-1))/2
                _X2[i, j] = 0.5 * np.sqrt(nj * (nj - 1))
        else:
            _X2[i, j] = 0.0

        _P2[i, j] = 2.0 * (ni + 0.5) * (1.0 if i == j else 0.0) - _X2[i, j]

# Constraint: c0 + 3c4 + 105c8 + 945c12 = 0
# c0 = -3c4 - 105c8 - 945c12
# Let y = [c4, c8, c12]
# c = [-3y0 - 105y1 - 945y2, y0, y1, y2]

_CONST_MATRIX = np.array([
    [-3.0, -105.0, -945.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

def _objective(y):
    c = _CONST_MATRIX @ y
    norm_sq = np.dot(c, c)
    if norm_sq < 1e-30:
        return 1e18
    
    ex2 = np.dot(c, _X2 @ c) / norm_sq
    ep2 = np.dot(c, _P2 @ c) / norm_sq
    
    if ex2 < 0 or ep2 < 0:
        return 1e18
        
    return np.sqrt(ex2 * ep2)

def _objective_grad(y):
    # Finite difference gradient is sufficient and robust for 3 variables
    eps = 1e-8
    grad = np.zeros(3)
    f0 = _objective(y)
    for i in range(3):
        y_plus = y.copy()
        y_plus[i] += eps
        grad[i] = (_objective(y_plus) - f0) / eps
    return grad

def construct():
    best_val = np.inf
    best_params = None
    
    # Seed and diverse starts
    # The seed program returned coefficients for H0, H4, H8.
    # In our parameterization, y = [c4, c8, c12].
    # We need to map the seed's output to y if possible, or just use generic starts.
    # The seed output was a0, a4, a8. a12 is derived.
    # Our y are the coefficients c4, c8, c12 in the HO basis expansion.
    # Note: The "coefficients" in the task description are the weights of H_n.
    # g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) exp(-x^2/2).
    # This is exactly c0|0> + c4|4> + c8|8> + c12|12> if we identify a_n with c_n.
    # So y = [a4, a8, a12].
    
    # Let's use the seed's approximate solution to generate a start point.
    # Seed: a0=-1.13, a4=0.04, a8=0.000308642
    # a12 = -(a0 + 3a4 + 105a8)/945
    a0_seed = -1.13
    a4_seed = 0.04
    a8_seed = 0.000308642
    a12_seed = -(a0_seed + 3.0*a4_seed + 105.0*a8_seed) / 945.0
    
    y_seed = np.array([a4_seed, a8_seed, a12_seed])
    
    starts = [
        y_seed,
        np.array([0.04, 0.0003, -0.000001]),
        np.array([0.05, 0.0005, -0.000002]),
        np.array([0.03, 0.0002, -0.000001]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.001, -0.00001]),
        np.array([-0.1, -0.001, 0.00001]),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(20):
        starts.append(rng.normal(y_seed, [0.05, 0.001, 0.00001]))
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Polish with BFGS if NM found a good spot
        if best_params is not None and time.time() < _DEADLINE - 10:
             try:
                 res2 = minimize(_objective, best_params, method='BFGS',
                               options={'maxiter': 1000, 'gtol': 1e-12})
                 if res2.fun < best_val and np.isfinite(res2.fun):
                     best_val = res2.fun
                     best_params = res2.x.copy()
             except Exception:
                 pass

    if best_params is None:
        # Fallback to seed
        return np.array([a0_seed, a4_seed, a8_seed])
    
    # Convert y back to a0, a4, a8
    a4, a8, a12 = best_params
    a0 = -3.0*a4 - 105.0*a8 - 945.0*a12
    
    return np.array([a0, a4, a8])

# EVOLVE-BLOCK-END
