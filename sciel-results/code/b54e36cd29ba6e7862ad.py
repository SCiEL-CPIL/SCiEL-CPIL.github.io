import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def count_nonzero(poly):
        return np.count_nonzero(poly)

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    best_score = float('inf')
    best_poly = None

    def update_best(c, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            best_poly = c.copy()

    # Phase 1: Systematic enumeration of sign patterns with ±1 coefficients on support {0..11}
    # 2^12 = 4096 patterns, very fast
    for signs in product([1, -1], repeat=12):
        if time.time() > deadline:
            break
        c = np.array(signs, dtype=np.int64)
        score = evaluate(c)
        if score < best_score:
            update_best(c, score)
        if best_score <= 12:
            break

    # Phase 2: For the best sign pattern found, try extending coefficients to {-2,-1,1,2}
    if best_poly is not None and time.time() < deadline:
        # Local search with coefficients in {-3,-2,-1,1,2,3}
        curr_c = best_poly.copy()
        curr_score = best_score
        no_improvement = 0
        max_no_improvement = 20000
        random.seed(42)

        while time.time() < deadline and no_improvement < max_no_improvement:
            idx = random.randint(0, 11)
            new_val = random.randint(-3, 3)
            if new_val == 0:
                new_val = 1
            if curr_c[idx] == new_val:
                continue

            old_val = curr_c[idx]
            curr_c[idx] = new_val
            new_score = evaluate(curr_c)

            if new_score < curr_score:
                curr_score = new_score
                no_improvement = 0
                update_best(curr_c, curr_score)
                if best_score <= 12:
                    break
            else:
                curr_c[idx] = old_val
                no_improvement += 1

    # Phase 3: Try different supports - shift the support or use structured supports
    if time.time() < deadline - 10:
        # Try supports that are arithmetic progressions or other structured sets
        structured_supports = [
            list(range(12)),           # 0..11
            list(range(0, 24, 2)),     # 0,2,4,...,22
            [0,1,2,3,4,5,6,7,8,9,10,20],  # mostly consecutive with one far term
            [0,1,2,3,4,5,6,7,8,9,10,11],
            [0,1,2,3,4,5,6,7,8,9,10,11],
        ]

        for support in structured_supports:
            if time.time() > deadline:
                break
            if len(support) != 12:
                continue
            max_deg = max(support)
            c = np.zeros(max_deg + 1, dtype=np.int64)
            for i, s in enumerate(support):
                c[s] = 1

            # Try sign patterns
            for signs in product([1, -1], repeat=12):
                if time.time() > deadline:
                    break
                c = np.zeros(max_deg + 1, dtype=np.int64)
                for i, s in enumerate(support):
                    c[s] = signs[i]
                score = evaluate(c)
                if score < best_score:
                    update_best(c, score)
                if best_score <= 12:
                    break

            if best_score <= 12:
                break

    # Phase 4: Final local search from best found
    if best_poly is not None and time.time() < deadline:
        curr_c = best_poly.copy()
        curr_score = best_score
        no_improvement = 0
        random.seed(123)

        while time.time() < deadline and no_improvement < 50000:
            # Try modifying a coefficient
            if len(curr_c) > 0:
                idx = random.randint(0, len(curr_c) - 1)
                new_val = random.randint(-5, 5)
                if new_val == 0:
                    new_val = 1
                if curr_c[idx] == new_val:
                    continue

                old_val = curr_c[idx]
                curr_c[idx] = new_val

                # Check constraints
                if len(curr_c) > 41 or curr_c[-1] == 0 or not np.all((curr_c >= -64) & (curr_c <= 64)):
                    curr_c[idx] = old_val
                    no_improvement += 1
                    continue

                nz = count_nonzero(curr_c)
                if nz != 12:
                    curr_c[idx] = old_val
                    no_improvement += 1
                    continue

                new_score = evaluate(curr_c)

                if new_score < curr_score:
                    curr_score = new_score
                    no_improvement = 0
                    update_best(curr_c, curr_score)
                    if best_score <= 12:
                        break
                else:
                    curr_c[idx] = old_val
                    no_improvement += 1

    # Final validation and cleanup
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    # Ensure final coefficient is nonzero
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    # Ensure exactly 12 non-zero terms
    if count_nonzero(best_poly) != 12:
        # Fallback
        best_poly = np.array([1]*12, dtype=np.int64)

    return best_poly.tolist()
