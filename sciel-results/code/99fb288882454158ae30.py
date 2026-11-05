import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 52.0  # safety margin before 60s kill

    best_count = float('inf')
    best_coeffs = None

    def eval_square(coeffs_list):
        # coeffs_list: list of (exp, coef) with len 12, coef != 0
        max_e = max(e for e, c in coeffs_list)
        square = [0] * (2 * max_e + 1)
        for i, (ei, ci) in enumerate(coeffs_list):
            for j in range(i, len(coeffs_list)):
                ej, cj = coeffs_list[j]
                v = ci * cj
                if i == j:
                    square[ei + ej] += v
                else:
                    square[ei + ej] += 2 * v
        count = sum(1 for x in square if x != 0)
        return count, square, max_e

    def to_full(coeffs_list):
        max_e = max(e for e, c in coeffs_list)
        full = [0] * (max_e + 1)
        for e, c in coeffs_list:
            full[e] = c
        if len(full) > 41:
            return None
        if full[-1] == 0:
            return None
        # verify exactly 12 nonzero, entries in range
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
        if best_count == 12:
            return best_coeffs
    
    # Phase 1b: Arithmetic progression exponents with structured coefficients
    # For step k, exponents are 0, k, 2k, ..., 11k. Coeffs alternate signs.
    for k in range(1, 4):  # steps 1, 2, 3
        if time.time() > deadline:
            break
        max_exp = 11 * k
        if max_exp > 40:
            continue
        for sign_pattern in range(1 << 12):
            if time.time() > deadline:
                break
            coeffs_list = [(i * k, (1 if (sign_pattern >> i) & 1 else -1)) for i in range(12)]
            count, _, _ = eval_square(coeffs_list)
            try_update(count, coeffs_list)
            if best_count == 12:
                return best_coeffs
        # Also try with magnitude scaling
        for scale in [1, 2, 3]:
            if time.time() > deadline:
                break
            for sign_pattern in range(1 << 12):
                if time.time() > deadline:
                    break
                coeffs_list = [(i * k, scale * (1 if (sign_pattern >> i) & 1 else -1)) for i in range(12)]
                count, _, _ = eval_square(coeffs_list)
                try_update(count, coeffs_list)
                if best_count == 12:
                    return best_coeffs

    # Phase 2: Brute force magnitudes {-2,-1,1,2} on consecutive 0..11 (4^12 too big; use random + sign patterns)
    random.seed(12345)
    vals = [-2, -1, 1, 2]
    # Smart: fix one coefficient to 1 (global scale/sign invariance), random the rest
    iters = 0
    while time.time() < deadline:
        iters += 1
        cl = [(0, 1)]
        for i in range(1, 12):
            cl.append((i, random.choice(vals)))
        count, _, _ = eval_square(cl)
        try_update(count, cl)
        if best_count == 12:
            return best_coeffs
        if iters % 200000 == 0 and time.time() > deadline:
            break
    
    # Phase 2b: Structured coefficient patterns on arithmetic progressions
    # Try patterns like c_i = (-1)^i * i, c_i = (-1)^i * (i+1), etc.
    for k in range(1, 4):
        if time.time() > deadline:
            break
        max_exp = 11 * k
        if max_exp > 40:
            continue
        # Pattern 1: alternating signs, magnitude = i+1
        coeffs_list = [(i * k, (i + 1) * (1 if i % 2 == 0 else -1)) for i in range(12)]
        if all(abs(c) <= 64 for _, c in coeffs_list):
            count, _, _ = eval_square(coeffs_list)
            try_update(count, coeffs_list)
            if best_count == 12:
                return best_coeffs
        # Pattern 2: alternating signs, magnitude = 1
        coeffs_list = [(i * k, (1 if i % 2 == 0 else -1)) for i in range(12)]
        count, _, _ = eval_square(coeffs_list)
        try_update(count, coeffs_list)
        if best_count == 12:
            return best_coeffs
        # Pattern 3: all same sign
        coeffs_list = [(i * k, 1) for i in range(12)]
        count, _, _ = eval_square(coeffs_list)
        try_update(count, coeffs_list)
        if best_count == 12:
            return best_coeffs
        # Pattern 4: alternating, magnitude = 2^i (if fits)
        coeffs_list = [(i * k, (1 << i) * (1 if i % 2 == 0 else -1)) for i in range(12)]
        if all(abs(c) <= 64 for _, c in coeffs_list):
            count, _, _ = eval_square(coeffs_list)
            try_update(count, coeffs_list)
            if best_count == 12:
                return best_coeffs

    # Phase 3: Local hill-climbing from best found (and from random starts)
    def hill_climb(start_cl, time_budget=2.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        cl = list(start_cl)
        count, _, _ = eval_square(cl)
        try_update(count, cl)
        no_improve_count = 0
        while time.time() - t0 < time_budget:
            improved = False
            # Try multiple moves per iteration for better exploration
            for _ in range(3):
                idx = random.randrange(12)
                e, c = cl[idx]
                # try changing coefficient
                new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_c != c and abs(new_c) <= 64:
                    cl2 = list(cl)
                    cl2[idx] = (e, new_c)
                    c2, _, _ = eval_square(cl2)
                    if c2 < count:
                        cl = cl2
                        count = c2
                        try_update(count, cl)
                        improved = True
                        no_improve_count = 0
                        break
                else:
                    # try shifting exponent within a window
                    shift = random.choice([-3, -2, -1, 1, 2, 3])
                    new_e = e + shift
                    if new_e >= 0 and new_e <= 40:
                        # ensure no exponent collision
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
                                no_improve_count = 0
                                break
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50 and random.random() < 0.1:
                    # Perturb: change 2 random coefficients
                    for _ in range(2):
                        idx = random.randrange(12)
                        e, c = cl[idx]
                        new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                        if new_c != c and abs(new_c) <= 64:
                            cl[idx] = (e, new_c)
                            no_improve_count = 0
                            break
            if best_count == 12:
                break
        return cl, count

    # Hill climb from several random starts
    start_cl = [(0, 1)] + [(i, random.choice(vals)) for i in range(1, 12)]
    if best_coeffs is not None:
        # reconstruct start from best
        bcl = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        start_cl = bcl
    while time.time() < deadline:
        hill_climb(start_cl, time_budget=1.5)
        if best_count == 12:
            return best_coeffs
        # new random start
        start_cl = [(0, 1)] + [(i, random.choice(vals)) for i in range(1, 12)]

    # Phase 4: Try non-consecutive exponents with small coefficients
    while time.time() < deadline:
        exps = sorted(random.sample(range(25), 12))
        cl = [(exps[i], random.choice(vals)) for i in range(12)]
        cl[0] = (exps[0], 1)
        count, _, _ = eval_square(cl)
        try_update(count, cl)
        if best_count == 12:
            return best_coeffs
        hill_climb(cl, time_budget=0.5)
        if best_count == 12:
            return best_coeffs
    
    # Phase 5: Focused search on arithmetic progression structures
    # The key insight: if exponents form AP with step k, and coefficients are chosen
    # to maximize cancellation, we can get very sparse squares.
    # For step k, the square has support at multiples of 2k only if coefficients
    # satisfy certain orthogonality conditions.
    for k in range(1, 5):
        if time.time() > deadline:
            break
        max_exp = 11 * k
        if max_exp > 40:
            continue
        # Systematically try different coefficient magnitudes on AP
        for mag_pattern in range(1 << 4):  # 4 magnitude bits: 1,2,3,4
            if time.time() > deadline:
                break
            mags = []
            for i in range(12):
                bit = (mag_pattern >> (i % 4)) & 1
                base_mag = 1 + bit  # 1 or 2
                mags.append(base_mag)
            for sign_bits in range(1 << 12):
                if time.time() > deadline:
                    break
                coeffs_list = [(i * k, mags[i] * (1 if (sign_bits >> i) & 1 else -1)) for i in range(12)]
                count, _, _ = eval_square(coeffs_list)
                try_update(count, coeffs_list)
                if best_count == 12:
                    return best_coeffs
                if count < best_count:
                    # Hill climb from this promising start
                    hill_climb(coeffs_list, time_budget=0.3)
                    if best_count == 12:
                        return best_coeffs

    if best_coeffs is None:
        # Fallback: use the best known structure
        best_coeffs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1] + [0]*29
    return best_coeffs
