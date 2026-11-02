"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Precompute moments of Gaussian e^{-2*pi*x^2}
# Integral of x^k * exp(-2*pi*x^2) from -inf to inf
# I_0 = 1/sqrt(2)
# I_2 = 1/(4*pi*sqrt(2))
# I_4 = 3/(16*pi^2*sqrt(2))
# ...
def _get_moments(max_power=26):
    moments = {}
    a = 2 * np.pi
    sqrt_pi_over_a = np.sqrt(np.pi / a)
    # I_{2m} = (2m-1)!! / (2a)^m * sqrt(pi/a)
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

# Coefficients for Hermite polynomials H_0, H_4, H_8, H_12 in monomial basis
# H_0: [1]
# H_4: [12, -48, 16] -> c0 + c2 x^2 + c4 x^4
# H_8: [3024, -20160, 20160, -4608, 256]
# H_12: [665280, -7983360, 831600, -7096320, 5940, -135168, 4096]

H_COEFFS = {
    0: np.array([1.0]),
    4: np.array([12.0, -48.0, 16.0]),
    8: np.array([3024.0, -20160.0, 20160.0, -4608.0, 256.0]),
    12: np.array([665280.0, -7983360.0, 831600.0, -7096320.0, 5940.0, -135168.0, 4096.0])
}

# Values at zero
H_VALS_0 = {
    0: 1.0,
    4: 12.0,
    8: 3024.0,
    12: 665280.0
}

def _poly_to_monomial(coeffs_dict):
    """
    coeffs_dict: {degree: coefficient}
    Returns a list of monomial coefficients [c0, c2, c4, ...] up to max degree.
    """
    max_deg = max(coeffs_dict.keys())
    # We only have even powers
    max_monomial_power = max_deg
    # Initialize array of size max_monomial_power//2 + 1
    mono_coeffs = np.zeros(max_monomial_power // 2 + 1)
    
    for deg, c in coeffs_dict.items():
        h_coeffs = H_COEFFS[deg]
        # h_coeffs[i] is coeff of x^{2i}
        for i, hc in enumerate(h_coeffs):
            mono_coeffs[i] += c * hc
            
    return mono_coeffs

def _calc_variance(mono_coeffs):
    """
    Calculate variance of x given P(x)^2 * exp(-2*pi*x^2).
    Returns (norm, second_moment).
    """
    L = len(mono_coeffs)
    # Square the polynomial coefficients
    # mono_coeffs[i] is coeff of x^{2i}
    # Square coeff of x^{2j} is sum_{i+k=j} mono_coeffs[i]*mono_coeffs[k]
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
        power = 2 * j # x^{2j}
        # Integral of x^{2j} e^{-2pi x^2}
        if power in MOMENTS:
            norm += c_sq * MOMENTS[power]
            # Second moment: \int x^2 * x^{2j} e^{-...} = \int x^{2j+2} e^{-...}
            power2 = 2 * j + 2
            if power2 in MOMENTS:
                second_moment += c_sq * MOMENTS[power2]
                
    return norm, second_moment

def _compute_score(c0, c4, c8):
    # Calculate c12
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    c12 = -(c0 * H_VALS_0[0] + c4 * H_VALS_0[4] + c8 * H_VALS_0[8]) / H_VALS_0[12]
    
    coeffs_pos = {0: c0, 4: c4, 8: c8, 12: c12}
    mono_pos = _poly_to_monomial(coeffs_pos)
    
    # Fourier side: H_n -> (-1)^{n/2} H_n
    # n=0: +, n=4: -, n=8: +, n=12: -
    coeffs_fourier = {0: c0, 4: -c4, 8: c8, 12: -c12}
    mono_fourier = _poly_to_monomial(coeffs_fourier)
    
    norm_x, second_moment_x = _calc_variance(mono_pos)
    norm_xi, second_moment_xi = _calc_variance(mono_fourier)
    
    if norm_x <= 0 or norm_xi <= 0:
        return 1e10
        
    var_x = second_moment_x / norm_x
    var_xi = second_moment_xi / norm_xi
    
    # A(f) is likely ||f||_2 or related. The task says "upper bound on A(f)*A(Fourier(f))".
    # The seed score is 1.0. Let's assume the score is sqrt(var_x * var_xi).
    # Or maybe var_x * var_xi?
    # Let's stick to the previous definition which seemed consistent with the seed.
    return np.sqrt(var_x * var_xi)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0 # Safety margin for 180s limit
    
    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_score = _compute_score(*x0)
    best_x = x0.copy()
    
    # 1. Local optimization using L-BFGS-B with tight tolerances
    try:
        res = opt.minimize(_compute_score, x0, method='L-BFGS-B', 
                           options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 200})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # 2. Try multiple local optimization methods from different starting points
    # Generate diverse starting points
    start_points = [
        x0.copy(),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.2, 0.05, 0.0]),
        np.array([-1.1, 0.03, 0.001]),
        np.array([-0.9, 0.02, 0.0005]),
        np.array([-1.3, 0.04, 0.0002]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30.0:
            break
        for method in ['L-BFGS-B', 'Powell', 'Nelder-Mead']:
            if time.time() > deadline - 5.0:
                break
            try:
                if method == 'L-BFGS-B':
                    res = opt.minimize(_compute_score, sp, method=method,
                                       options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15})
                elif method == 'Powell':
                    res = opt.minimize(_compute_score, sp, method=method,
                                       options={'maxiter': 1000, 'ftol': 1e-18, 'xtol': 1e-15})
                else:  # Nelder-Mead
                    res = opt.minimize(_compute_score, sp, method=method,
                                       options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-18})
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # 3. Differential Evolution with tighter bounds and more iterations
    if time.time() < deadline - 15.0:
        # Based on the seed values, use tighter bounds
        bounds = [(-2.0, 0.0), (-0.1, 0.1), (-0.001, 0.001)]
        
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 5.0:
                break
            try:
                res_de = opt.differential_evolution(_compute_score, bounds, 
                                                    maxiter=50, popsize=20, 
                                                    seed=seed, polish=True,
                                                    tol=1e-12, mutation=(0.3, 0.9),
                                                    recombination=0.7)
                if res_de.fun < best_score:
                    best_score = res_de.fun
                    best_x = res_de.x.copy()
            except Exception:
                pass
    
    # 4. Final high-precision polish with multiple methods
    if time.time() < deadline - 2.0 and best_score < 1e10:
        # Try Powell first (derivative-free, robust)
        try:
            res_final = opt.minimize(_compute_score, best_x, method='Powell', 
                                     options={'maxiter': 5000, 'ftol': 1e-20, 'xtol': 1e-18})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass
        
        # Then Nelder-Mead for final polish
        if time.time() < deadline - 1.0:
            try:
                res_nm = opt.minimize(_compute_score, best_x, method='Nelder-Mead', 
                                      options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-20,
                                               'initial_simplex': None, 'adaptive': True})
                if res_nm.fun < best_score:
                    best_score = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass
    
    return best_x

# EVOLVE-BLOCK-END
