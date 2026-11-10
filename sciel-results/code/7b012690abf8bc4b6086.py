import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def compute_square(coeffs):
        """Compute exact square coefficients (ascending degree)."""
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(i, n):
                cj = coeffs[j]
                if cj == 0:
                    continue
                val = ci * cj
                if i == j:
                    sq[i + j] += val
                else:
                    sq[i + j] += 2 * val
        return sq

    def count_nz(sq):
        return sum(1 for x in sq if x != 0)

    def evaluate(coeffs):
        sq = compute_square(coeffs)
        return count_nz(sq)

    def validate(coeffs):
        if len(coeffs) > 41:
            return False
        if coeffs[-1] == 0:
            return False
        if coeffs[0] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    def make_poly(exponents, coefficients, max_deg=40):
        """Create coefficient list from exponents and coefficients."""
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            if 0 <= e <= max_deg:
                coeffs[e] = c
        return coeffs

    best_score = float('inf')
    best_poly = None

    def update_best(poly, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            best_poly = poly.copy()

    # ============================================================
    # STRATEGY 1: Factor-based construction
    # P = A * B, |A| * |B| = 12
    # P^2 = A^2 * B^2, sparsity(P^2) <= sparsity(A^2) * sparsity(B^2)
    # ============================================================

    def try_factor_pair(fa, ca, fb, cb, max_deg=40):
        """
        Build P = A*B from factor A (exponents fa, coeffs ca) and B (exponents fb, coeffs cb).
        Returns the polynomial and its square nonzero count, or None if invalid.
        """
        # Build A and B as dense coefficient lists
        max_a = max(fa) if fa else 0
        max_b = max(fb) if fb else 0
        A = [0] * (max_a + 1)
        B = [0] * (max_b + 1)
        for e, c in zip(fa, ca):
            A[e] = c
        for e, c in zip(fb, cb):
            B[e] = c

        # Multiply A * B
        P = np.convolve(A, B)
        P = [int(x) for x in P]

        # Trim to max_deg
        if len(P) > max_deg + 1:
            return None, None

        # Check: exactly 12 nonzero terms
        nz = sum(1 for c in P if c != 0)
        if nz != 12:
            return None, None

        # Check: last coefficient nonzero
        if P[-1] == 0:
            # Trim trailing zeros
            while len(P) > 1 and P[-1] == 0:
                P.pop()
            if P[-1] == 0 or len(P) > max_deg + 1:
                return None, None

        # Check: first coefficient nonzero
        if P[0] == 0:
            # This is allowed per problem statement? Let me re-read...
            # "nonzero final coefficient" - only the final one must be nonzero
            # But "exactly 12 nonzero terms" - first can be zero
            pass

        # Check coefficient bounds
        if not all(-64 <= c <= 64 for c in P):
            return None, None

        # Pad to length 41 if needed? No, just return as is (length <= 41)
        score = evaluate(P)
        return P, score

    # --- Factor pair (3, 4): A has 3 terms, B has 4 terms ---
    # A exponents: arithmetic progression a, a+d, a+2d
    # B exponents: arithmetic progression b, b+e, b+2e, b+3e
    # Coefficients from small set

    coeff_set = [-3, -2, -1, 1, 2, 3]
    coeff_set_small = [-2, -1, 1, 2]

    # Precompute: for given factor exponents, try coefficient assignments
    # A: 3 terms -> 3 coeffs from coeff_set (6^3 = 216)
    # B: 4 terms -> 4 coeffs from coeff_set (6^4 = 1296)
    # Total per (a,d,b,e) combo: 216 * 1296 = 280,000 -- too many per combo
    # Instead: fix exponents, do smart coefficient search

    def search_factor_3x4():
        nonlocal best_score, best_poly
        if time.time() > deadline - 30:
            return

        # A: 3 terms at a, a+d, a+2d
        # B: 4 terms at b, b+e, b+2e, b+3e
        # P = A*B has at most 3*4=12 terms (fewer if exponents collide)
        # We need exactly 12 nonzero terms in P

        # Key: P's exponents are {a+ib + e*j : i=0..2, j=0..3}
        # For exactly 12 terms, all 12 must be distinct (no collisions)
        # AND all products must be nonzero

        # Let's try: A at {0, d, 2d}, B at {0, e, 2e, 3e}
        # P exponents: {i*d + j*e : i=0,1,2; j=0,1,2,3}
        # For no collisions, need gcd(d,e) structure...
        # Actually just enumerate and check

        for d in range(1, 12):
            for e in range(1, 12):
                # Check that all 12 exponents are distinct
                exps = set()
                for i in range(3):
                    for j in range(4):
                        exps.add(i * d + j * e)
                if len(exps) != 12:
                    continue

                max_exp = max(exps)
                if max_exp > 40:
                    continue

                # Now try coefficient combinations
                # A coeffs: (ca0, ca1, ca2) from coeff_set
                # B coeffs: (cb0, cb1, cb2, cb3) from coeff_set
                # P[i*d+j*e] = ca_i * cb_j
                # All 12 must be nonzero (guaranteed if all ca, cb nonzero)
                # And within [-64, 64]

                for ca0, ca1, ca2 in product(coeff_set_small, repeat=3):
                    for cb0, cb1, cb2, cb3 in product(coeff_set_small, repeat=4):
                        # Build P
                        P = [0] * (max_exp + 1)
                        ca = [ca0, ca1, ca2]
                        cb = [cb0, cb1, cb2, cb3]
                        for i in range(3):
                            for j in range(4):
                                P[i * d + j * e] = ca[i] * cb[j]

                        # Check bounds
                        if any(abs(c) > 64 for c in P if c != 0):
                            continue
                        # Check exactly 12 nonzero
                        if sum(1 for c in P if c != 0) != 12:
                            continue
                        if P[-1] == 0:
                            continue

                        score = evaluate(P)
                        update_best(P, score)

        # Also try shifted versions: A at {s, s+d, s+2d}, B at {t, t+e, t+2e, t+3e}
        for s in range(0, 3):
            for t in range(0, 3):
                for d in range(1, 10):
                    for e in range(1, 10):
                        exps = set()
                        for i in range(3):
                            for j in range(4):
                                exps.add(s + i * d + t + j * e)
                        if len(exps) != 12:
                            continue
                        max_exp = max(exps)
                        if max_exp > 40:
                            continue

                        for ca0, ca1, ca2 in product(coeff_set_small, repeat=3):
                            for cb0, cb1, cb2, cb3 in product(coeff_set_small, repeat=4):
                                P = [0] * (max_exp + 1)
                                ca = [ca0, ca1, ca2]
                                cb = [cb0, cb1, cb2, cb3]
                                for i in range(3):
                                    for j in range(4):
                                        P[s + i * d + t + j * e] = ca[i] * cb[j]

                                if any(abs(c) > 64 for c in P if c != 0):
                                    continue
                                if sum(1 for c in P if c != 0) != 12:
                                    continue
                                if P[-1] == 0:
                                    continue

                                score = evaluate(P)
                                update_best(P, score)

    def search_factor_2x6():
        nonlocal best_score, best_poly
        if time.time() > deadline - 30:
            return

        # A: 2 terms at a, a+d
        # B: 6 terms at b, b+e, b+2e, b+3e, b+4e, b+5e
        # P exponents: {a+ib + e*j : i=0,1; j=0..5} = 12 terms if distinct

        for d in range(1, 15):
            for e in range(1, 10):
                exps = set()
                for i in range(2):
                    for j in range(6):
                        exps.add(i * d + j * e)
                if len(exps) != 12:
                    continue
                max_exp = max(exps)
                if max_exp > 40:
                    continue

                # A coeffs: (ca0, ca1) from coeff_set
                # B coeffs: (cb0..cb5) from coeff_set_small
                for ca0, ca1 in product(coeff_set_small, repeat=2):
                    for cb in product(coeff_set_small, repeat=6):
                        P = [0] * (max_exp + 1)
                        for i in range(2):
                            for j in range(6):
                                P[i * d + j * e] = (ca0 if i == 0 else ca1) * cb[j]

                        if any(abs(c) > 64 for c in P if c != 0):
                            continue
                        if sum(1 for c in P if c != 0) != 12:
                            continue
                        if P[-1] == 0:
                            continue

                        score = evaluate(P)
                        update_best(P, score)

    # --- Also try (4,3) and (6,2) which are the same as (3,4) and (2,6) with A,B swapped ---
    # Already covered above since multiplication is commutative

    # ============================================================
    # STRATEGY 2: Direct structured search (from current program)
    # ============================================================

    def direct_structured_search():
        nonlocal best_score, best_poly
        if time.time() > deadline - 25:
            return

        # Arithmetic progression exponents with various coefficient patterns
        for d in range(1, 15):
            for a in range(0, 5):
                exps = [a + i * d for i in range(12)]
                if max(exps) > 40:
                    continue

                # Try various coefficient patterns
                patterns = [
                    [1] * 12,
                    [1 if i % 2 == 0 else -1 for i in range(12)],
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                    [1, 2, 1, -1, -2, -1, 1, 2, 1, -1, -2, -1],
                    [1, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1],
                ]

                for pat in patterns:
                    P = [0] * 41
                    for i, e in enumerate(exps):
                        P[e] = pat[i]
                    if P[-1] == 0:
                        continue
                    score = evaluate(P)
                    update_best(P, score)

    # ============================================================
    # STRATEGY 3: Random + Hill Climbing (from current program, refined)
    # ============================================================

    def random_init():
        """Generate random 12-term polynomial."""
        # Choose 12 distinct exponents
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        P = [0] * 41
        for e, c in zip(exps, coeffs):
            P[e] = c
        # Ensure last is nonzero
        if P[-1] == 0:
            P[-1] = 1
        return P

    def hill_climb(poly, max_iters=30000):
        nonlocal best_score, best_poly
        curr = poly.copy()
        curr_score = evaluate(curr)
        update_best(curr, curr_score)

        no_improve = 0
        max_no_improve = 5000
        T = 3.0
        T_min = 0.1
        cooling = 0.9999

        while time.time() < deadline - 5 and no_improve < max_no_improve:
            # Pick a random nonzero position to modify
            nz_indices = [i for i, c in enumerate(curr) if c != 0]
            idx = random.choice(nz_indices)
            old_val = curr[idx]

            # Generate candidates
            candidates = set()
            for dv in [-3, -2, -1, 1, 2, 3]:
                nv = old_val + dv
                if -64 <= nv <= 64 and nv != 0:
                    candidates.add(nv)
            candidates.add(-old_val)
            candidates.add(random.choice([-5, -3, -2, -1, 1, 2, 3, 5]))

            improved = False
            for new_val in candidates:
                if new_val == old_val:
                    continue
                curr[idx] = new_val
                new_score = evaluate(curr)
                delta = new_score - curr_score

                if delta < 0 or random.random() < np.exp(-delta / max(T, T_min)):
                    curr_score = new_score
                    improved = True
                    update_best(curr, curr_score)
                    no_improve = 0
                    break
                else:
                    curr[idx] = old_val

            if not improved:
                no_improve += 1
            T *= cooling
            if T < T_min:
                T = T_min

    def targeted_zeroing(poly, max_iters=10000):
        """Try to zero out specific square coefficients."""
        nonlocal best_score, best_poly
        curr = poly.copy()
        curr_score = evaluate(curr)
        update_best(curr, curr_score)

        no_improve = 0
        max_no_improve = 3000

        while time.time() < deadline - 5 and no_improve < max_no_improve:
            sq = compute_square(curr)
            nonzero_ks = [k for k in range(len(sq)) if sq[k] != 0]

            if not nonzero_ks:
                break

            # Pick a random nonzero square coefficient to try to zero
            k = random.choice(nonzero_ks)

            # Try modifying each coefficient that affects sq[k]
            found = False
            for i in range(len(curr)):
                if curr[i] == 0:
                    continue
                j = k - i
                if j < 0 or j >= len(curr) or curr[j] == 0:
                    continue

                # sq[k] involves curr[i]*curr[j] (and curr[j]*curr[i] if i!=j)
                # Try setting curr[i] to zero it out
                # sq[k] = 2*curr[i]*curr[j] + (other terms not involving curr[i])
                # This is complex; try simple perturbation instead

                old_val = curr[i]
                for dv in [-2, -1, 1, 2]:
                    nv = old_val + dv
                    if -64 <= nv <= 64 and nv != 0:
                        curr[i] = nv
                        new_score = evaluate(curr)
                        if new_score < curr_score:
                            curr_score = new_score
                            found = True
                            update_best(curr, curr_score)
                            no_improve = 0
                            break
                        curr[i] = old_val
                if found:
                    break

            if not found:
                # Random perturbation
                idx = random.randint(0, len(curr) - 1)
                old_val = curr[idx]
                nv = old_val + random.choice([-3, -2, -1, 1, 2, 3])
                if -64 <= nv <= 64 and nv != 0:
                    curr[idx] = nv
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        update_best(curr, curr_score)
                        no_improve = 0
                    else:
                        curr[idx] = old_val
                        no_improve += 1
                else:
                    no_improve += 1

    # ============================================================
    # Run strategies in order of expected value
    # ============================================================

    random.seed(42)

    # 1. Factor-based search (most promising structural approach)
    search_factor_3x4()

    if time.time() < deadline - 25:
        search_factor_2x6()

    # 2. Direct structured search
    if time.time() < deadline - 20:
        direct_structured_search()

    # 3. Random initialization + hill climbing
    if time.time() < deadline - 15:
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            poly = random_init()
            score = evaluate(poly)
            update_best(poly, score)

    # 4. Hill climbing from best
    if best_poly is not None and time.time() < deadline - 10:
        hill_climb(best_poly, max_iters=20000)

    # 5. Targeted zeroing from best
    if best_poly is not None and time.time() < deadline - 5:
        targeted_zeroing(best_poly, max_iters=5000)

    # Final validation
    if best_poly is None:
        best_poly = [1] * 12 + [0] * 29  # 12 ones, padded to length 41
        best_poly = best_poly[:41]

    if not validate(best_poly):
        best_poly = [1] * 12 + [0] * 29
        best_poly = best_poly[:41]

    return best_poly
