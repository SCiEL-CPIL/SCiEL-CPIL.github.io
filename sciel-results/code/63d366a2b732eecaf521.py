import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def count_nonzero(poly):
        return np.count_nonzero(poly)

    def poly_square(coeffs):
        # coeffs is a numpy array
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def validate(coeffs):
        # Check constraints
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if count_nonzero(coeffs) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    random.seed(42)

    def try_poly(poly_list):
        nonlocal best_score, best_poly
        if not validate(np.array(poly_list)):
            return
        score = evaluate(np.array(poly_list))
        if score < best_score:
            best_score = score
            best_poly = np.array(poly_list, dtype=np.int64).copy()

    # Phase 1: Try structured candidates with various supports
    # Contiguous support
    for _ in range(50):
        c = [random.choice([-1, 1]) for _ in range(12)]
        try_poly(c)

    # Try patterns based on known sparse square constructions
    # (1+x)^2 = 1+2x+x^2
    # We want P with 12 terms where P^2 is very sparse
    # Try: P = sum of terms that pair up to cancel in the square
    
    # Strategy: Try polynomials where coefficients come in pairs that might cancel
    # e.g., a_i = a_j for some i,j, creating symmetric structure
    
    # Try all-ones with sign flips at strategic positions
    base = [1]*12
    for flip_count in range(1, 5):
        for _ in range(200):
            c = base[:]
            flips = random.sample(range(12), flip_count)
            for f in flips:
                c[f] = -c[f]
            try_poly(c)
    
    # Try with magnitude variations
    for _ in range(200):
        c = [random.choice([-2,-1,1,2]) for _ in range(12)]
        try_poly(c)

    # Phase 2: Local search on best candidate
    if best_poly is not None:
        curr = best_poly.copy()
        curr_score = best_score
        no_improve = 0
        max_no_improve = 5000
        temp = 1.0

        while time.time() < deadline - 5 and no_improve < max_no_improve:
            idx = random.randint(0, len(curr)-1)
            old_val = curr[idx]
            new_val = random.randint(-8, 8)
            if new_val == 0:
                new_val = 1 if random.random() < 0.5 else -1
            if new_val == old_val:
                continue

            curr[idx] = new_val
            new_score = evaluate(curr)

            if new_score < curr_score:
                curr_score = new_score
                no_improve = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr.copy()
                    if best_score <= 7:
                        break
            else:
                delta = new_score - curr_score
                if random.random() < np.exp(-delta / max(temp, 0.1)):
                    curr_score = new_score
                    no_improve = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                else:
                    curr[idx] = old_val
                    no_improve += 1

            temp *= 0.9995

    # Phase 3: Try non-contiguous supports
    if time.time() < deadline - 10:
        for _ in range(500):
            # Choose 12 distinct exponents from 0 to 40
            exponents = sorted(random.sample(range(41), 12))
            c = [0]*41
            for i, e in enumerate(exponents):
                c[e] = random.choice([-2,-1,1,2])
            # Ensure last coefficient is nonzero
            if c[-1] == 0:
                c[exponents[-1]] = 1
            try_poly(c)

        # Local search on best non-contiguous candidate
        if best_poly is not None and len(best_poly) > 12:
            curr = best_poly.copy()
            curr_score = best_score
            no_improve = 0
            max_no_improve = 3000
            temp = 1.0

            while time.time() < deadline - 3 and no_improve < max_no_improve:
                idx = random.randint(0, len(curr)-1)
                old_val = curr[idx]
                new_val = random.randint(-8, 8)
                if new_val == 0:
                    new_val = 1 if random.random() < 0.5 else -1
                if new_val == old_val:
                    continue

                curr[idx] = new_val
                new_score = evaluate(curr)

                if new_score < curr_score:
                    curr_score = new_score
                    no_improve = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                        if best_score <= 7:
                            break
                else:
                    delta = new_score - curr_score
                    if random.random() < np.exp(-delta / max(temp, 0.1)):
                        curr_score = new_score
                        no_improve = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                    else:
                        curr[idx] = old_val
                        no_improve += 1

                temp *= 0.9995

    # Ensure final coefficient is nonzero
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
