import numpy as np
from scipy.optimize import minimize
from itertools import product

# Hermite Polynomials coefficients for H0, H4, H8, H12
# H_n(x) = sum_{k=0}^{n} c_{n,k} x^k
# Using standard probabilists' or physicists'?
# The task mentions "Hermite-polynomial times Gaussian".
# Standard physicists' Hermite: H_n(x) e^{-x^2/2}.
# H_0 = 1
# H_4 = x^4 - 6x^2 + 3
# H_8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12 = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 226800x^2 + 10395

def get_hermite_coeffs(n):
    # Returns array of coefficients for x^0 to x^n
    # Using recurrence or explicit formulas
    # For n=0, 4, 8, 12, we can hardcode or generate
    if n == 0:
        return np.array([1.0])
    if n == 4:
        return np.array([3.0, 0.0, -6.0, 0.0, 1.0])
    if n == 8:
        return np.array([105.0, 0.0, -420.0, 0.0, 210.0, 0.0, -28.0, 0.0, 1.0])
    if n == 12:
        return np.array([10395.0, 0.0, -226800.0, 0.0, 151200.0, 0.0, -30240.0, 0.0, 2520.0, 0.0, -84.0, 0.0, 1.0])
    raise ValueError("n not supported")

H0 = get_hermite_coeffs(0)
H4 = get_hermite_coeffs(4)
H8 = get_hermite_coeffs(8)
H12 = get_hermite_coeffs(12)

# Values at 0
H0_0 = H0[0]
H4_0 = H4[0]
H8_0 = H8[0]
H12_0 = H12[0]

# Weights for L2 norm: ||sum c_n H_n||_2^2 = sqrt(pi) * sum c_n^2 * 2^n * n!
W0 = np.sqrt(np.pi) * 1
W4 = np.sqrt(np.pi) * (2**4 * 24)
W8 = np.sqrt(np.pi) * (2**8 * 40320)
W12 = np.sqrt(np.pi) * (2**12 * 479001600)

def poly_mul(p, q):
    return np.convolve(p, q)

def poly_pow4(p):
    return poly_mul(poly_mul(p, p), poly_mul(p, p))

def integrate_monomial(coeff, power, a=2.0):
    # Integral of x^power * exp(-a x^2) from -inf to inf
    # Only even powers contribute
    if power % 2 != 0:
        return 0.0
    k = power // 2
    # (2k-1)!! / (2a)^(k+0.5) * sqrt(pi/a)
    # (2k-1)!! = (2k)! / (2^k k!)
    if k == 0:
        double_fact = 1.0
    else:
        double_fact = np.math.factorial(2*k) / (2**k * np.math.factorial(k))
    val = double_fact * np.sqrt(np.pi / a) / ((2*a) ** (k + 0.5))
    return coeff * val

def compute_norm_sq(c0, c4, c8, c12):
    return c0**2 * W0 + c4**2 * W4 + c8**2 * W8 + c12**2 * W12

def compute_L4_norm_sq(c0, c4, c8, c12, is_fourier=False):
    # Construct P(x)
    # c_n * H_n(x)
    p = np.zeros(13)
    p[0:1] += c0 * H0
    p[0:5] += c4 * H4
    p[0:9] += c8 * H8
    p[0:13] += c12 * H12
    
    if is_fourier:
        # Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}
        # So we multiply coefficients by (-i)^n
        # But we are working with real polynomials?
        # The L4 norm is integral of |g|^4.
        # g(x) is real. g(x)^4 is real.
        # The Fourier transform \hat{g}(x) is real?
        # H_n is even. e^{-x^2/2} is even. g is even.
        # \hat{g} is even.
        # The coefficients of the polynomial part of \hat{g} are c_n (-i)^n.
        # Since n is even, (-i)^n is real (+1 or -1).
        # n=0: 1
        # n=4: 1
        # n=8: 1
        # n=12: 1
        # So the polynomial is the same!
        # Wait, (-i)^4 = 1. (-i)^8 = 1. (-i)^12 = 1.
        # So P(x) for Fourier is the same as P(x) for g.
        pass

    p4 = poly_pow4(p)
    integral = 0.0
    for i, coeff in enumerate(p4):
        if coeff != 0:
            integral += integrate_monomial(coeff, i, a=2.0)
    return integral

def objective(params):
    c0, c4, c8 = params
    # c12 from g(0)=0
    # c0*1 + c4*3 + c8*105 + c12*10395 = 0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    norm_sq = compute_norm_sq(c0, c4, c8, c12)
    if norm_sq < 1e-12:
        return 1e100
        
    # Normalize coefficients to unit L2 norm
    scale = 1.0 / np.sqrt(norm_sq)
    nc0, nc4, nc8, nc12 = c0*scale, c4*scale, c8*scale, c12*scale
    
    l4_g = compute_L4_norm_sq(nc0, nc4, nc8, nc12, is_fourier=False)
    l4_f = compute_L4_norm_sq(nc0, nc4, nc8, nc12, is_fourier=True)
    
    # Product
    return l4_g * l4_f

def construct():
    # Use differential evolution for global optimization
    # Bounds for c0, c4, c8
    # The coefficients are roughly of order 1, 0.1, 0.01
    bounds = [(-5, 5), (-5, 5), (-1, 1)]
    
    # Run optimization
    res = minimize(objective, [0, 0, 0], method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    # Try multiple starts if Nelder-Mead fails or to ensure global
    best_val = res.fun
    best_x = res.x
    
    # A few random starts
    np.random.seed(42)
    for _ in range(20):
        start = np.random.uniform(-2, 2, 3)
        r = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if r.fun < best_val:
            best_val = r.fun
            best_x = r.x
            
    c0, c4, c8 = best_x
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    norm_sq = compute_norm_sq(c0, c4, c8, c12)
    scale = 1.0 / np.sqrt(norm_sq)
    
    return np.array([c0*scale, c4*scale, c8*scale])

