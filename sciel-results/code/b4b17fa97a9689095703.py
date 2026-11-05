import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 52.0

    best_count = float('inf')
    best_coeffs = None

    def eval_square(coeffs_list):
        # coeffs_list: list of (exp, coef) with len 12, coef != 0
        max_e = max(e for e, c in coeffs_list)
        # Use numpy convolution for speed
        # Create dense vector
        vec = np.zeros(max_e + 1, dtype=np.int64)
        for e, c in coeffs_list:
            vec[e] = c
        sq = np.convolve(vec, vec)
        count = np.count_nonzero(sq)
        return count, sq, max_e

    def to_full(coeffs_list):
        max_e = max(e for e, c in coeffs_list)
        full = [0] * (max_e + 1)
        for e, c in coeffs_list:
            full[e] = c
        if len(full) > 41:
            return None
        if full[-1] == 0:
            return None
        nz = sum(1 for x in full if x != 0)
        if nz != 12:
            return None
        if any(abs(x) > 64 for x in full):
            return None
        return full

    def try_update(count, coeffs_list):
        nonlocal best_count, best_coeffs
        if count < best_count:
            full = to_full(coeffs_list)
            if full is not None:
                best_count = count
                best_coeffs = full
                return True
        return False

    # Phase 1: Brute force signs on 12 consecutive exponents 0..11, coeffs in {-1,1}
    for bits in range(1 << 12):
        if time.time() > deadline:
            break
        coeffs_list = [(i, (1 if (bits >> i) & 1 else -1)) for i in range(12)]
        count, _, _ = eval_square(coeffs_list)
        try_update(count, coeffs_list)
        if best_count == 12: # Theoretical min for 12 terms? No, min is 1?
            # Actually min non-zero terms in square of 12-term poly is at least 1.
            # But practically high.
            pass

    # Phase 2: Random search with NumPy
    random.seed(12345)
    vals = [-2, -1, 1, 2]
    iters = 0
    while time.time() < deadline:
        iters += 1
        # Random exponents
        exps = sorted(random.sample(range(0, 41), 12))
        cl = [(exps[i], random.choice(vals)) for i in range(12)]
        count, _, _ = eval_square(cl)
        try_update(count, cl)
        if best_count <= 6:
            break
        if iters % 10000 == 0 and time.time() > deadline:
            break

    # Phase 3: Hill climbing
    def hill_climb(start_cl, time_budget=1.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = list(start_cl)
        count, _, _ = eval_square(cl)
        try_update(count, cl)
        while time.time() - t0 < time_budget:
            improved = False
            idx = random.randrange(12)
            e, c = cl[idx]
            # Try changing coefficient
            new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_c != c:
                cl2 = list(cl)
                cl2[idx] = (e, new_c)
                c2, _, _ = eval_square(cl2)
                if c2 < count:
                    cl = cl2
                    count = c2
                    try_update(count, cl)
                    improved = True
            else:
                # Try shifting exponent
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_e = e + shift
                if 0 <= new_e <= 40:
                    exps = [cl[i][0] for i in range(12)]
                    exps.remove(e)
                    if new_e not in exps:
                        cl2 = list(cl)
                        cl2[idx] = (new_e, c)
                        c2, _, _ = eval_square(cl2)
                        if c2 < count:
                            cl = cl2
                            count = c2
                            try_update(count, cl)
                            improved = True
            if not improved and random.random() < 0.1:
                break

    if best_coeffs is not None:
        bcl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        hill_climb(bcl, time_budget=5.0)
    
    # More random starts
    while time.time() < deadline - 5.0:
        exps = sorted(random.sample(range(0, 41), 12))
        cl = [(exps[i], random.choice(vals)) for i in range(12)]
        hill_climb(cl, time_budget=0.5)
        if best_count <= 6:
            break

    if best_coeffs is None:
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1] + [0]*29

    return best_coeffs
