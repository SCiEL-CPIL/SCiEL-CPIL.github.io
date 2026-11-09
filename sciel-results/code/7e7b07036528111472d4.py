"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
import sympy as sp

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 630s wall-clock, use 590s safety margin
    t0 = time.time()
    DEADLINE = 590.0

    # Strategy: use zeros of J_0 (Bessel function of the first kind, order 0)
    # The asymptotic zeros of generalized Laguerre polynomials L_n^{alpha}(x)
    # are related to the zeros of Bessel functions. For large n, the zeros
    # of L_n^{alpha}(x) are approximately (4n + 2alpha + 2) * (j_{alpha,k}/(2*pi))^2
    # where j_{alpha,k} is the k-th zero of J_alpha.
    #
    # Here m=12 (number of double roots), n=1, alpha=-1/2.
    # The relevant parameter for the Bessel connection: the "effective" order
    # relates to alpha. For L_n^{alpha}, the zeros are asymptotically at
    # x_k ≈ 4n * (j_{alpha+1/2, k} / (2π))^2 for the largest n involved.
    #
    # Since we have degrees up to 50 (2*25), the largest polynomial degree is 50.
    # The zeros of L_50^{-1/2}(x) should be near our double roots to minimize
    # the largest root of q.
    #
    # Actually, let's think more carefully. We need 12 double roots z_1..z_12
    # such that g(z_i)=g'(z_i)=0. The polynomial g has degree at most 50.
    # The quotient q = g / (x * prod (x-z_i)^2) has degree at most 50 - 1 - 24 = 25.
    # We want to minimize the largest real root of q.
    #
    # Key insight: if the z_i are chosen to be near the zeros of L_50^{-1/2}(x),
    # then g can be constructed to have small coefficients, and q will have
    # roots that are controlled.
    #
    # For alpha = -1/2, the Laguerre polynomials L_n^{-1/2}(x) are related to
    # Chebyshev polynomials. Specifically, L_n^{-1/2}(x) = c_n * T_n(sqrt(x)) * x^{-n/2}
    # or similar. The zeros of L_n^{-1/2}(x) are at x_k = 4n * sin^2(k*pi/(2n+1)) approximately?
    # No, let me use the Bessel connection properly.
    #
    # The zeros of L_n^{alpha}(x) for large n are approximately:
    # x_k ≈ 4n * (j_{alpha, k} / (2π))^2  for k = 1, ..., n
    # where j_{alpha,k} is the k-th positive zero of J_alpha.
    #
    # For alpha = -1/2, J_{-1/2}(x) = sqrt(2/(pi*x)) * sin(x)
    # So the zeros of J_{-1/2} are at x = k*pi for k = 1, 2, 3, ...
    #
    # Therefore, for L_n^{-1/2}(x), the zeros are approximately:
    # x_k ≈ 4n * (k*pi / (2π))^2 = 4n * k^2/4 = n * k^2
    #
    # So for n=50, the zeros are approximately at 50*k^2 for k=1,...,50.
    # The 12 smallest would be at 50, 200, 450, 800, 1250, 1800, 2450, 3200, 4050, 5000, 6050, 7200.
    #
    # But we want to minimize the LARGEST root of q, not the largest z_i.
    # Let me think about what determines the roots of q.
    #
    # Alternative approach: use the exact zeros of L_50^{-1/2}(x) computed numerically,
    # take the 12 that give the best result.
    #
    # Let me compute the zeros of L_50^{-1/2}(x) using numpy/scipy.

    def get_laguerre_zeros(n, alpha, num_zeros=None):
        """Get zeros of generalized Laguerre polynomial L_n^alpha(x)."""
        # Use sympy for exact computation, then evaluate
        x = sp.Symbol('x')
        L = sp.laguerre(n, alpha, x)
        # Find roots numerically
        roots = sp.nroots(L, n=50)
        real_roots = sorted([complex(r).real for r in roots if abs(complex(r).imag) < 1e-10])
        if num_zeros is not None:
            return real_roots[:num_zeros]
        return real_roots

    # Try different n values and select the 12 zeros that minimize B
    # The idea: pick 12 zeros from a Laguerre polynomial of some degree
    # that make the quotient q have small largest root.
    
    best_result = None
    best_B = float('inf')

    # Strategy 1: Use zeros of L_50^{-1/2}(x)
    # Since the highest degree in g is 50, the zeros of L_50^{-1/2} are natural candidates.
    try:
        x = sp.Symbol('x')
        L50 = sp.laguerre(50, sp.Rational(-1, 2), x)
        # Get numerical roots
        roots_50 = sp.nroots(L50, n=30)
        real_roots_50 = sorted([float(complex(r).real) for r in roots_50 if abs(complex(r).imag) < 1e-8 and complex(r).real > 0])
        
        if len(real_roots_50) >= 12:
            # Take the 12 smallest positive roots
            candidates = real_roots_50[:12]
            if all(c > 0 for c in candidates) and len(set(candidates)) == 12:
                result = evaluate_candidates(candidates, t0, DEADLINE)
                if result is not None:
                    z_list, B_val = result
                    if B_val < best_B:
                        best_B = B_val
                        best_result = z_list
    except Exception as e:
        pass

    # Strategy 2: Use asymptotic formula x_k = 50 * k^2 (from J_{-1/2} zeros)
    candidates2 = [50.0 * k**2 for k in range(1, 13)]
    result = evaluate_candidates(candidates2, t0, DEADLINE)
    if result is not None:
        z_list, B_val = result
        if B_val < best_B:
            best_B = B_val
            best_result = z_list

    # Strategy 3: Try different scalings of the asymptotic formula
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        candidates3 = [scale * 50.0 * k**2 for k in range(1, 13)]
        if time.time() - t0 > DEADLINE - 60:
            break
        result = evaluate_candidates(candidates3, t0, DEADLINE)
        if result is not None:
            z_list, B_val = result
            if B_val < best_B:
                best_B = B_val
                best_result = z_list

    # Strategy 4: Use zeros of lower-degree Laguerre polynomials
    for n_test in [20, 25, 30, 35, 40]:
        if time.time() - t0 > DEADLINE - 60:
            break
        try:
            x = sp.Symbol('x')
            Ln = sp.laguerre(n_test, sp.Rational(-1, 2), x)
            roots_n = sp.nroots(Ln, n=30)
            real_roots_n = sorted([float(complex(r).real) for r in roots_n if abs(complex(r).imag) < 1e-8 and complex(r).real > 0])
            if len(real_roots_n) >= 12:
                candidates4 = real_roots_n[:12]
                if all(c > 0 for c in candidates4) and len(set(candidates4)) == 12:
                    result = evaluate_candidates(candidates4, t0, DEADLINE)
                    if result is not None:
                        z_list, B_val = result
                        if B_val < best_B:
                            best_B = B_val
                            best_result = z_list
        except Exception:
            pass

    # If no valid result found, fall back to simple spacing
    if best_result is None:
        best_result = [4.0 * (i + 1) for i in range(12)]

    return best_result


def evaluate_candidates(z_candidates, t0, deadline):
    """Evaluate a set of candidate double roots. Returns (z_list, B) or None."""
    elapsed = time.time() - t0
    if elapsed > deadline - 30:
        return None

    z_candidates = sorted(z_candidates)
    
    # Check validity: finite, > 0, pairwise distinct
    if len(z_candidates) != 12:
        return None
    for z in z_candidates:
        if not np.isfinite(z) or z <= 0:
            return None
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_candidates[i] - z_candidates[j]) < 1e-10:
                return None

    # Convert to rationals
    z_rationals = [sp.Rational(float(z)) for z in z_candidates]
    z_rationals = sorted(z_rationals)

    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)

    # Build g(x) = sum_{k=0}^{25} c_k * L_{2k}^{-1/2}(x)
    # We need 26 coefficients c_0..c_25
    # Constraints: g(0)=0, g'(0)=1, and for each z_i: g(z_i)=0, g'(z_i)=0
    # That's 2 + 12*2 = 26 constraints, matching 26 unknowns.

    # Build the system of linear equations
    # For each constraint, we need the value of L_{2k}^{-1/2} at the point
    # and its derivative.

    # Precompute Laguerre polynomials and their derivatives at needed points
    points = [sp.Integer(0)] + list(z_rationals)
    
    # Build matrix A and vector b
    # Row 0: g(0) = 0 => sum c_k * L_{2k}(0) = 0
    # Row 1: g'(0) = 1 => sum c_k * L'_{2k}(0) = 1
    # Rows 2..13: g(z_i) = 0
    # Rows 14..25: g'(z_i) = 0

    num_constraints = 26
    num_unknowns = 26

    A = sp.zeros(num_constraints, num_unknowns)
    b = sp.zeros(num_constraints, 1)

    # Evaluate Laguerre polynomials at 0
    L_at_0 = []
    dL_at_0 = []
    for k in range(26):
        d = 2 * k
        Ld = sp.laguerre(d, alpha, x)
        L_at_0.append(Ld.subs(x, 0))
        dLd = sp.diff(Ld, x)
        dL_at_0.append(dLd.subs(x, 0))

    # Row 0: sum c_k * L_{2k}(0) = 0
    for k in range(26):
        A[0, k] = L_at_0[k]
    b[0, 0] = 0

    # Row 1: sum c_k * L'_{2k}(0) = 1
    for k in range(26):
        A[1, k] = dL_at_0[k]
    b[1, 0] = 1

    # For each z_i, compute L_{2k}(z_i) and L'_{2k}(z_i)
    for i, zi in enumerate(z_rationals):
        # g(zi) = 0
        row_g = 2 + i
        # g'(zi) = 0
        row_dg = 14 + i

        for k in range(26):
            d = 2 * k
            Ld = sp.laguerre(d, alpha, x)
            A[row_g, k] = Ld.subs(x, zi)
            dLd = sp.diff(Ld, x)
            A[row_dg, k] = dLd.subs(x, zi)
        b[row_g, 0] = 0
        b[row_dg, 0] = 0

    # Solve the system
    try:
        # Check if system is singular
        det = A.det()
        if det == 0:
            return None
        sol = sp.LUsolve(A, b)
    except Exception:
        return None

    # Build g(x)
    c_coeffs = [sol[k, 0] for k in range(26)]
    g = sum(c_coeffs[k] * sp.laguerre(2*k, alpha, x) for k in range(26))
    g = sp.expand(g)

    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    denom = sp.expand(denom)

    # Polynomial division: q(x) = g(x) / denom
    try:
        q, remainder = sp.div(g, denom, x)
        if remainder != 0:
            # Not divisible, which shouldn't happen if our constraints are correct
            # But let's still try to proceed
            pass
    except Exception:
        return None

    # Find real roots of q
    try:
        real_roots_q = sp.real_roots(q)
    except Exception:
        return None

    # Count roots with sign change
    B = 0
    for r in real_roots_q:
        # Check sign change at r - 1e-198 and r + 1e-198
        try:
            val_neg = q.subs(x, r - sp.Rational(1, 10)**198)
            val_pos = q.subs(x, r + sp.Rational(1, 10)**198)
            if val_neg * val_pos < 0:
                if r > B:
                    B = r
        except Exception:
            continue

    B_float = float(B)
    return (sorted([float(z) for z in z_rationals]), B_float)


# EVOLVE-BLOCK-END
