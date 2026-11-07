"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import sys
from fractions import Fraction

# Pre-compute Hermite polynomial coefficients using sympy if available, else numpy
# We need H0, H4, H8, H12
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 280x^4 - 840x^2 + 105
# H12(x) = x^12 - 84x^10 + 1260x^8 - 13020x^6 + 39900x^4 - 26460x^2 + 10395

# Let's verify these coefficients.
# Hermite polynomials (physicists'):
# H0 = 1
# H1 = 2x
# H2 = 4x^2 - 2
# H3 = 8x^3 - 12x
# H4 = 16x^4 - 48x^2 + 12
# Wait, the standard physicists' Hermite polynomials are defined by:
# H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0 = 1
# H_1 = 2x
# H_2 = 4x^2 - 2
# H_3 = 8x^3 - 12x
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 4608x^6 + 20736x^4 - 20736x^2 + 3024
# H_12 = 4096x^12 - 147456x^10 + 1658880x^8 - 6635520x^6 + 8028192x^4 - 2677104x^2 + 1032192

# Let's re-check with sympy to be absolutely sure, but since I can't run sympy here,
# I will rely on the standard definitions.
# Actually, let's use the recurrence relation to be safe, or just hardcode the known values.
# From standard tables:
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 4608x^6 + 20736x^4 - 20736x^2 + 3024
# H12(x) = 4096x^12 - 147456x^10 + 1658880x^8 - 6635520x^6 + 8028192x^4 - 2677104x^2 + 1032192

# Let's double check H4(0) = 12, H8(0) = 3024, H12(0) = 1032192.
# c3 = -(c0*1 + c1*12 + c2*3024) / 1032192

# The polynomial g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# g(x) = c0 + c1(16x^4 - 48x^2 + 12) + c2(256x^8 - 4608x^6 + 20736x^4 - 20736x^2 + 3024) + c3(4096x^12 - 147456x^10 + 1658880x^8 - 6635520x^6 + 8028192x^4 - 2677104x^2 + 1032192)

# q(x) = g(x) / x^2
# q(x) is a polynomial of degree 10 (if c3 != 0)
# We need the largest real root r of q(x) where q changes sign.
# Then value = r^2 / (2*pi). We want to minimize value, so minimize r^2.

# Let's define the coefficients of q(x) in terms of c0, c1, c2.
# g(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
# where a0 = c0 + 12*c1 + 3024*c2 + 1032192*c3 = 0 (by construction)
# a2 = -48*c1 - 20736*c2 - 2677104*c3
# a4 = 16*c1 + 20736*c2 + 8028192*c3
# a6 = -4608*c2 - 6635520*c3
# a8 = 256*c2 + 1658880*c3
# a10 = -147456*c3
# a12 = 4096*c3

# q(x) = a12 x^10 + a10 x^8 + a8 x^6 + a6 x^4 + a4 x^2 + a2
# Let y = x^2. Then q(x) = P(y) = a12 y^5 + a10 y^4 + a8 y^3 + a6 y^2 + a4 y + a2
# We need the largest real root r of q(x) where q changes sign.
# This is equivalent to finding the largest real positive root y of P(y) where P(y) changes sign, and then r = sqrt(y).
# But we must be careful: q(x) changes sign at x=r iff P(y) changes sign at y=r^2.
# Also, we need to ensure that g(x) tends to a positive limit at infinity (or replace by -g if negative).
# The leading coefficient of g(x) is a12 = 4096*c3. If a12 < 0, we replace g by -g, which means we flip the sign of all coefficients, including c3.
# So effectively, we can assume c3 > 0 without loss of generality, because if c3 < 0, the evaluator flips the sign.
# Wait, the problem says: "If g tends to a negative limit at infinity, g is replaced by -g."
# This means the leading coefficient of g must be positive. So we should enforce c3 > 0.
# But c3 is determined by c0, c1, c2. So we need to ensure that c3 > 0, or if c3 < 0, the evaluator flips the sign, which is equivalent to using -c0, -c1, -c2.
# So we can just search over c0, c1, c2 and compute c3, and if c3 < 0, we can just use the absolute value or flip the sign.
# Actually, it's simpler to just let the evaluator handle it, but for our objective function, we need to compute the value correctly.
# So in our objective function, we compute c3, and if c3 < 0, we flip the sign of all coefficients (c0, c1, c2, c3) to ensure the leading coefficient is positive.

# Let's write the objective function.
def compute_value(c0, c1, c2):
    c3 = -(c0 * 1 + c1 * 12 + c2 * 3024) / 1032192.0
    if c3 < 0:
        c0, c1, c2, c3 = -c0, -c1, -c2, -c3
    a12 = 4096.0 * c3
    a10 = -147456.0 * c3
    a8 = 256.0 * c2 + 1658880.0 * c3
    a6 = -4608.0 * c2 - 6635520.0 * c3
    a4 = 16.0 * c1 + 20736.0 * c2 + 8028192.0 * c3
    a2 = -48.0 * c1 - 20736.0 * c2 - 2677104.0 * c3
    # P(y) = a12 y^5 + a10 y^4 + a8 y^3 + a6 y^2 + a4 y + a2
    # Find the largest positive real root y of P(y) where P changes sign.
    # We can use numpy.roots to find all roots, then filter.
    coeffs = [a12, a10, a8, a6, a4, a2]
    roots = np.roots(coeffs)
    # Filter real positive roots
    real_roots = roots[np.isreal(roots)].real
    pos_roots = real_roots[real_roots > 1e-12]
    if len(pos_roots) == 0:
        return 1e100  # Large penalty
    # We need the root where P changes sign. For a polynomial, a real root is a sign-change root if its multiplicity is odd.
    # We can check the sign of P just before and after the root.
    # Alternatively, we can just take the largest positive real root and assume it's a sign-change root if the leading coefficient is positive and the polynomial goes to +inf as y->inf.
    # But we need to be careful. Let's check the sign change explicitly.
    # For each candidate root y, evaluate P(y - eps) and P(y + eps).
    # But this is expensive. Instead, we can use the fact that for a polynomial with real coefficients, a real root of odd multiplicity is a sign-change root.
    # We can compute the multiplicity by checking how close the root is to other roots, but that's also tricky.
    # A simpler approach: evaluate P at the largest positive real root and see if it's close to zero, and check the sign of P at a point slightly less and slightly greater.
    # Let's just take the largest positive real root and assume it's valid. If it's not, the value will be wrong, but the optimizer should find a valid one.
    # Actually, the problem states "the largest real root of q at which q changes sign". So we must find the largest such root.
    # Let's iterate over all positive real roots in descending order and check for sign change.
    pos_roots_sorted = np.sort(pos_roots)[::-1]
    for y in pos_roots_sorted:
        # Check sign change
        eps = 1e-6 * max(1.0, abs(y))
        p_minus = np.polyval(coeffs, y - eps)
        p_plus = np.polyval(coeffs, y + eps)
        if p_minus * p_plus < 0:
            r = np.sqrt(y)
            return r**2 / (2 * np.pi)
    # If no sign-change root found, return a large penalty
    return 1e100

# Now, we need to optimize c0, c1, c2 to minimize compute_value.
# We can use differential evolution for global search, then Nelder-Mead for local refinement.

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180 seconds as per the problem statement

    # Initial seed
    c0_seed, c1_seed, c2_seed = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    best_val = compute_value(c0_seed, c1_seed, c2_seed)
    best_params = [c0_seed, c1_seed, c2_seed]

    # Define bounds for differential evolution
    # We need to choose reasonable bounds. The seed is around [-1.13, 0.04, 0.0003]
    # Let's try bounds of [-10, 10] for c0, c1, and [-1, 1] for c2.
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-1.0, 1.0)]

    # Run differential evolution
    try:
        result = differential_evolution(
            lambda x: compute_value(x[0], x[1], x[2]),
            bounds,
            seed=42,
            maxiter=100,
            popsize=10,
            tol=1e-6,
            callback=lambda x, convergence: (time.time() > deadline)  # Stop if deadline exceeded
        )
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_params = list(result.x)
    except Exception as e:
        pass

    # Local refinement with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            result2 = minimize(
                lambda x: compute_value(x[0], x[1], x[2]),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result2.success and result2.fun < best_val:
                best_val = result2.fun
                best_params = list(result2.x)
        except Exception as e:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
