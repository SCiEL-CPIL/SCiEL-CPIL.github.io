import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import time

# Precompute Hermite polynomials coefficients or use scipy's hermite
# H_n(x) are physicists' Hermite polynomials.
# We need H_0, H_4, H_8, H_12.
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 84x^10 + 2100x^8 - 25200x^6 + 151200x^4 - 302400x^2 + 10395

def get_hermite_coeffs(deg):
    # Returns coefficients of x^k for H_deg(x)
    # We only need even degrees.
    # Using scipy.special.hermite gives a poly1d object.
    h = hermite(deg)
    # h.coef() returns [x^deg, x^(deg-1), ..., x^0]
    return h.coef()

# Coefficients for H_0, H_4, H_8, H_12
# H_0: [1]
# H_4: [1, 0, -6, 0, 3]
# H_8: [1, 0, -28, 0, 210, 0, -420, 0, 105]
# H_12: [1, 0, -84, 0, 2100, 0, -25200, 0, 151200, 0, -302400, 0, 10395]

H0 = np.array([1.0])
H4 = np.array([1.0, 0.0, -6.0, 0.0, 3.0])
H8 = np.array([1.0, 0.0, -28.0, 0.0, 210.0, 0.0, -420.0, 0.0, 105.0])
H12 = np.array([1.0, 0.0, -84.0, 0.0, 2100.0, 0.0, -25200.0, 0.0, 151200.0, 0.0, -302400.0, 0.0, 10395.0])

# We need to compute integrals of x^k e^{-x^2} and derivatives.
# Instead of symbolic, we can use numerical integration or known moments.
# The function is g(x) = e^{-x^2} P(x) where P(x) is a polynomial of degree 12.
# g(x) is a linear combination of Hermite functions? No, Hermite functions are H_n(x) e^{-x^2/2}.
# Here we have e^{-x^2}. This is a Gaussian with variance 1/2.
# Let's use numerical integration for robustness and speed.
# Range of integration: [-10, 10] is sufficient for e^{-x^2}.

x = np.linspace(-10, 10, 10000)
dx = x[1] - x[0]
w = np.exp(-x**2)

# Precompute powers of x weighted by w
# We need:
# 1. ||g||^2 = int |g|^2 dx
# 2. ||x g||^2 = int x^2 |g|^2 dx
# 3. ||g'||^2 = int |g'|^2 dx

# g(x) = w(x) * (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x))
# Let P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
# g(x) = w(x) P(x)
# g'(x) = w'(x) P(x) + w(x) P'(x) = -2x w(x) P(x) + w(x) P'(x) = w(x) (P'(x) - 2x P(x))

# To compute these efficiently, we can precompute the basis vectors for the integrals.
# However, P(x) depends on c.
# Let's define the basis functions b_i(x) = H_{4i}(x) for i=0,1,2,3.
# P(x) = sum c_i b_i(x).
# P'(x) = sum c_i b_i'(x).

# We need to compute:
# I_00 = int w^2 b_0^2, I_01 = int w^2 b_0 b_1, etc.
# J_00 = int x^2 w^2 b_0^2, etc.
# K_00 = int w^2 (b_0' - 2x b_0)^2, etc.

# Actually, it's easier to just evaluate P(x) and P'(x) for the given c in the objective function.
# With 10000 points, this is fast enough for optimization.

def compute_integrals(c):
    c0, c4, c8 = c
    c12 = (c0 - 12*c4 + 168*c8) / 2880.0
    
    # Evaluate P(x) and P'(x)
    # H0(x) = 1, H0'(x) = 0
    # H4(x) = x^4 - 6x^2 + 3, H4'(x) = 4x^3 - 12x
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H8'(x) = 8x^7 - 168x^5 + 840x^3 - 840x
    # H12(x) = x^12 - 84x^10 + 2100x^8 - 25200x^6 + 151200x^4 - 302400x^2 + 10395
    # H12'(x) = 12x^11 - 840x^9 + 16800x^7 - 151200x^5 + 604800x^3 - 604800x

    x2 = x**2
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3
    x7 = x6 * x
    x8 = x4 * x4
    x9 = x8 * x
    x10 = x5 * x5
    x11 = x10 * x
    x12 = x6 * x6

    H0_val = np.ones_like(x)
    H4_val = x4 - 6*x2 + 3
    H8_val = x8 - 28*x6 + 210*x4 - 420*x2 + 105
    H12_val = x12 - 84*x10 + 2100*x8 - 25200*x6 + 151200*x4 - 302400*x2 + 10395

    H0_dval = np.zeros_like(x)
    H4_dval = 4*x3 - 12*x
    H8_dval = 8*x7 - 168*x5 + 840*x3 - 840*x
    H12_dval = 12*x11 - 840*x9 + 16800*x7 - 151200*x5 + 604800*x3 - 604800*x

    P_val = c0*H0_val + c4*H4_val + c8*H8_val + c12*H12_val
    P_dval = c0*H0_dval + c4*H4_dval + c8*H8_dval + c12*H12_dval

    g_val = w * P_val
    g_dval = w * (P_dval - 2*x * P_val)

    norm_sq = np.sum(g_val**2) * dx
    x_norm_sq = np.sum((x * g_val)**2) * dx
    d_norm_sq = np.sum(g_dval**2) * dx

    if norm_sq < 1e-12:
        return np.inf

    sigma_x = np.sqrt(x_norm_sq / norm_sq)
    sigma_xi = np.sqrt(d_norm_sq / norm_sq)
    
    return sigma_x * sigma_xi

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin
    
    # Seed solution
    seed_c = np.array([-113/100, 1/25, 1/3240])
    best_c = seed_c
    best_score = compute_integrals(seed_c)
    
    # Try a few random starts and the seed
    starts = [seed_c]
    # Add some heuristic starts
    starts.append(np.array([1.0, -0.1, 0.01]))
    starts.append(np.array([-1.0, 0.1, -0.01]))
    starts.append(np.array([0.0, 1.0, -0.1]))
    
    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        starts.append(rng.normal(0, 1, 3))
    
    for c0 in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(compute_integrals, c0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass

    return np.array(best_c)
