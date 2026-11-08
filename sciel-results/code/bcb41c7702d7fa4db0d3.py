import time
import math

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Theoretical best found: {0, 1, 3, 9}
    best_U = [0, 1, 3, 9]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline
    
    # Try structured candidates with geometric growth
    for base in [2, 3, 4, 5]:
        for n in range(3, 8):
            if time.time() > deadline:
                break
            U = [0]
            x = 1
            for _ in range(n):
                U.append(x)
                x *= base
            try_set(U)
    
    # Try {0, 1, a, a*b} type sets
    for a in range(2, 30):
        if time.time() > deadline:
            break
        for b in range(2, 20):
            if time.time() > deadline:
                break
            c = a * b
            if c > 500:
                break
            try_set([0, 1, a, c])
    
    # Try {0, 1, a, b, c} with geometric-like spacing
    for a in range(2, 15):
        if time.time() > deadline:
            break
        for b in range(a + 1, 30):
            if time.time() > deadline:
                break
            for c in range(b + 1, 80):
                if time.time() > deadline:
                    break
                try_set([0, 1, a, b, c])
    
    # Exhaustive search over 4-element sets {0, a, b, c} with 0 < a < b < c <= 60
    for a in range(1, 25):
        if time.time() > deadline:
            break
        for b in range(a + 1, 40):
            if time.time() > deadline:
                break
            for c in range(b + 1, 61):
                if not try_set([0, a, b, c]):
                    break
    
    # Exhaustive search over 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 35
    for a in range(1, 8):
        if time.time() > deadline:
            break
        for b in range(a + 1, 15):
            if time.time() > deadline:
                break
            for c in range(b + 1, 25):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 36):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # Try 6-element sets with geometric growth
    for a in range(1, 6):
        if time.time() > deadline:
            break
        for b in range(a + 1, 10):
            if time.time() > deadline:
                break
            for c in range(b + 1, 15):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 25):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 40):
                        if not try_set([0, a, b, c, d, e]):
                            break
    
    # Try sets with one large element and small dense part
    for small_max in range(5, 20):
        if time.time() > deadline:
            break
        for large in range(small_max + 5, small_max * 5 + 1):
            if time.time() > deadline:
                break
            # Small dense part: {0, 1, 2, ..., small_max} plus large
            U = list(range(small_max + 1)) + [large]
            try_set(U)
    
    # Try {0, 1, 2, ..., k} + {m} for various k, m
    for k in range(3, 15):
        if time.time() > deadline:
            break
        for m in range(k + 2, k * 5 + 1):
            if time.time() > deadline:
                break
            U = list(range(k + 1)) + [m]
            try_set(U)
    
    # Try arithmetic progressions
    for a in range(1, 10):
        if time.time() > deadline:
            break
        for d in range(1, 15):
            if time.time() > deadline:
                break
            for n in range(4, 8):
                U = [a * i * d for i in range(n)]
                U[0] = 0
                try_set(U)
    
    return best_U
