"""Spherical optimization for Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Precompute moments of Gaussian e^{-2*pi*x^2}
def _get_moments(max_power=26):
    moments = {}
    a = 2 * np.pi
    sqrt_pi_over_a = np.sqrt(np.pi / a)
    for m in range(max_power // 2 + 1):
        k = 2 * m
        if m == 0:
            val = sqrt_pi_over_a
        else:
            double_fact = 1
            for i in range(1, m + 1):
                double_fact *= (2 * i - 1)
            val = double_fact / ((2 * a) ** m) * sqrt_pi_over_a
        moments[k] = val
    return moments

MOMENTS = _get_moments(26)

# Coefficients for Hermite polynomials in monomial basis (even powers only)
H_COEFFS = {
    0: np.array([1.0]),
    4: np.array([12.0, -48.0, 16.0]),
    8: np.array([3024.0, -20160.0, 20160.0, -4608.0, 256.0]),
    12: np.array([665280.0, -7983360.0, 8316000.0, -7096320.0, 59400.0, -135168.0, 4096.0])
}

H_VALS_0 = {0: 1.0, 4: 12.0, 8: 3024.0, 12: 665280.0}

def _poly_to_monomial(coeffs_dict):
    max_deg = max(coeffs_dict.keys())
    mono_coeffs = np.zeros(max_deg // 2 + 1)
    for deg, c in coeffs_dict.items():
        h_coeffs = H_COEFFS[deg]
        for i, hc in enumerate(h_coeffs):
            mono_coeffs[i] += c * hc
    return mono_coeffs

def _calc_variance(mono_coeffs):
    L = len(mono_coeffs)
    square_coeffs = np.zeros(2 * L - 1)
    for j in range(2 * L - 1):
        s = 0.0
        for i in range(max(0, j - L + 1), min(L, j + 1)):
            k = j - i
            s += mono_coeffs[i] * mono_coeffs[k]
        square_coeffs[j] = s
    norm = 0.0
    second_moment = 0.0
    for j, c_sq in enumerate(square_coeffs):
        if c_sq == 0.0:
            continue
        power = 2 * j
        if power in MOMENTS:
            norm += c_sq * MOMENTS[power]
            power2 = 2 * j + 2
            if power2 in MOMENTS:
                second_moment += c_sq * MOMENTS[power2]
    return norm, second_moment

def _compute_score(c0, c4, c8):
    c12 = -(c0 * H_VALS_0[0] + c4 * H_VALS_0[4] + c8 * H_VALS_0[8]) / H_VALS_0[12]
    coeffs_pos = {0: c0, 4: c4, 8: c8, 12: c12}
    mono_pos = _poly_to_monomial(coeffs_pos)
    coeffs_fourier = {0: c0, 4: -c4, 8: c8, 12: -c12}
    mono_fourier = _poly_to_monomial(coeffs_fourier)
    norm_x, second_moment_x = _calc_variance(mono_pos)
    norm_xi, second_moment_xi = _calc_variance(mono_fourier)
    if norm_x <= 0 or norm_xi <= 0:
        return 1e10
    var_x = second_moment_x / norm_x
    var_xi = second_moment_xi / norm_xi
    return np.sqrt(var_x * var_xi)

def _spherical_fibonacci(n):
    """Generate n points on the unit sphere using Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1) if n > 1 else 0.0
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points

def _sphere_to_cartesian(sphere_point, center, radius):
    """Map a point on unit sphere to a box centered at 'center' with half-size 'radius'."""
    return center + radius * sphere_point

def _local_optimize(x0, deadline=None):
    """Local optimization using L-BFGS-B with tight tolerances."""
    if deadline is not None and time.time() > deadline:
        return x0, _compute_score(*x0)
    
    try:
        res = opt.minimize(_compute_score, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
        if res.fun < _compute_score(*x0):
            return res.x, res.fun
    except Exception:
        pass
    
    # Try Powell as backup
    try:
        res = opt.minimize(_compute_score, x0, method='Powell',
                          options={'maxiter': 1000, 'ftol': 1e-16, 'xtol': 1e-14})
        if res.fun < _compute_score(*x0):
            return res.x, res.fun
    except Exception:
        pass
    
    return x0, _compute_score(*x0)

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin for 180s limit
    
    # Initial guess from seed
    x0_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = _compute_score(*x0_seed)
    best_x = x0_seed.copy()
    
    # Phase 1: Spherical Fibonacci lattice search
    # The problem is scale-invariant, so we search on a sphere in coefficient space
    # Use multiple radii to cover different scales
    
    # Define search centers and radii
    search_configs = [
        (np.array([-1.13, 0.04, 0.0003]), np.array([0.5, 0.1, 0.001])),
        (np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.2, 0.005])),
        (np.array([-0.8, 0.05, 0.001]), np.array([0.8, 0.15, 0.003])),
    ]
    
    n_sphere_points = 2000  # Number of points per sphere
    
    for center, radius in search_configs:
        if time.time() > deadline - 30.0:
            break
        
        sphere_points = _spherical_fibonacci(n_sphere_points)
        
        # Sample the sphere and evaluate
        candidates = []
        for sp in sphere_points:
            x_candidate = _sphere_to_cartesian(sp, center, radius)
            score = _compute_score(*x_candidate)
            if score < 1e9:  # Filter out invalid points
                candidates.append((score, x_candidate))
        
        # Sort by score and keep top candidates for local optimization
        candidates.sort(key=lambda x: x[0])
        
        # Optimize top 50 candidates
        for score, x_candidate in candidates[:50]:
            if time.time() > deadline - 20.0:
                break
            
            x_opt, score_opt = _local_optimize(x_candidate, deadline - 5.0)
            if score_opt < best_score:
                best_score = score_opt
                best_x = x_opt.copy()
    
    # Phase 2: Final polish on the best solution
    if time.time() < deadline - 5.0 and best_score < 1e9:
        x_final, score_final = _local_optimize(best_x, deadline - 2.0)
        if score_final < best_score:
            best_score = score_final
            best_x = x_final.copy()
    
    return best_x

# EVOLVE-BLOCK-END
