import itertools
import math
import time
import numpy as np

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    U_arr = np.array(U)
    plus = np.add.outer(U_arr, U_arr)
    minus = np.subtract.outer(U_arr, U_arr)
    n_plus = len(np.unique(plus))
    n_minus = len(np.unique(minus))
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    best_score = 0.0
    best_set = [0, 1, 3]

    def try_set(U):
        nonlocal best_score, best_set
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = list(U)

    # Phase 1: Try structured sets
    structured_sets = []
    # Powers of 2
    for n in range(2, 10):
        structured_sets.append([0] + [2**i for i in range(n)])
    # Small arithmetic progressions with gaps
    for a in range(1, 8):
        for d in range(1, 8):
            structured_sets.append([0] + [a * i * d for i in range(1, 7)])
    # Mixed: small + large
    for small in range(1, 6):
        for large in range(10, 50, 3):
            structured_sets.append([0, small, large])
            structured_sets.append([0, small, large, 2*large])

    for U in structured_sets:
        if time.time() > deadline:
            return best_set
        try_set(U)

    # Phase 2: Exhaustive search over small ranges
    max_val = 30
    candidates = list(range(1, max_val + 1))

    for size in range(2, 8):
        if time.time() > deadline:
            return best_set
        for combo in itertools.combinations(candidates, size - 1):
            if time.time() > deadline:
                return best_set
            U = [0] + list(combo)
            try_set(U)

    # Phase 3: Local search / hill climbing from best set
    if time.time() < deadline - 10:
        current = list(best_set)
        current_score = best_score
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try adding a new element
            for x in range(1, 60):
                if x not in current:
                    new_U = sorted(current + [x])
                    s = _score(new_U)
                    if s > current_score:
                        current = new_U
                        current_score = s
                        improved = True
                        break
            # Try removing an element (keep 0)
            for i in range(1, len(current)):
                new_U = current[:i] + current[i+1:]
                s = _score(new_U)
                if s > current_score:
                    current = new_U
                    current_score = s
                    improved = True
                    break
            # Try swapping an element
            for i in range(1, len(current)):
                for x in range(1, 60):
                    if x not in current:
                        new_U = current[:i] + [x] + current[i+1:]
                        s = _score(new_U)
                        if s > current_score:
                            current = new_U
                            current_score = s
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        try_set(current)

    return best_set
