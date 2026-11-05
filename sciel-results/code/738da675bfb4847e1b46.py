import random
import time

def construct():
    start_time = time.time()
    deadline = 55.0

    def count_nonzero_square(coeffs):
        n = len(coeffs)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(n):
                cj = coeffs[j]
                if cj == 0:
                    continue
                sq[i + j] += ci * cj
        return sum(1 for x in sq if x != 0)

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if coeffs[-1] == 0:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    best_count = 23
    best_coeffs = [1] * 12

    # Try structured patterns
    test_patterns = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, 2, 3, 4, 5, 6, -6, -5, -4, -3, -2, -1],
        [1, 1, 2, 2, 3, 3, -3, -3, -2, -2, -1, -1],
        [1, 3, 5, 7, 9, 11, -11, -9, -7, -5, -3, -1],
    ]

    for p in test_patterns:
        if is_valid(p):
            c = count_nonzero_square(p)
            if c < best_count:
                best_count = c
                best_coeffs = list(p)

    random.seed(12345)

    def local_optimize(coeffs):
        nonlocal best_count, best_coeffs
        coeffs = list(coeffs)
        for _ in range(200):
            if time.time() - start_time > deadline - 2:
                break
            improved = False
            idx = random.randint(0, 11)
            old = coeffs[idx]
            # Try various new values
            candidates = []
            for d in range(-5, 6):
                nv = old + d
                if -64 <= nv <= 64 and nv != old:
                    candidates.append(nv)
            random.shuffle(candidates)
            for nv in candidates[:10]:
                trial = list(coeffs)
                trial[idx] = nv
                if sum(1 for x in trial if x != 0) != 12:
                    continue
                if trial[-1] == 0:
                    continue
                c = count_nonzero_square(trial)
                if c < best_count:
                    best_count = c
                    best_coeffs = trial
                    coeffs = trial
                    improved = True
                    break
            if not improved:
                break
        return coeffs

    # Random search
    while time.time() - start_time < deadline:
        # Generate random 12-term polynomial
        coeffs = []
        for i in range(12):
            if random.random() < 0.4:
                coeffs.append(random.choice([-1, 1]))
            elif random.random() < 0.6:
                coeffs.append(random.choice([-2, 2, -3, 3, -4, 4]))
            else:
                coeffs.append(random.randint(-20, 20))

        # Ensure exactly 12 nonzero
        for i in range(12):
            if coeffs[i] == 0:
                coeffs[i] = random.choice([-1, 1, 2, -2])
        if coeffs[-1] == 0:
            coeffs[-1] = random.choice([-1, 1, 2, -2])

        c = count_nonzero_square(coeffs)
        if c < best_count:
            best_count = c
            best_coeffs = list(coeffs)

        local_optimize(coeffs)

    if not is_valid(best_coeffs):
        best_coeffs = [1] * 12

    return best_coeffs
