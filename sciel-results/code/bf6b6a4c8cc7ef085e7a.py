import time
import math
import random
import numpy as np

def _score_of(U):
    n = len(U)
    if n < 2:
        return 0.0
    a = np.asarray(U, dtype=np.int64)
    s = a[:, None] + a[None, :]
    d = a[:, None] - a[None, :]
    ns = len(np.unique(s))
    nd = len(np.unique(d))
    if ns <= 0:
        return 0.0
    ratio = nd / ns
    if ratio <= 0:
        return 0.0
    maxU = a.max()
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    DEADLINE = 145
    
    def now_ok():
        return (time.time() - start_time) < DEADLINE

    # Initialize with a strong candidate
    best_U = [0, 4, 16, 64]
    best_score = _score_of(best_U)

    def try_candidate(U):
        nonlocal best_U, best_score
        if not now_ok():
            return False
        U = sorted(set(int(x) for x in U))
        if 0 not in U or len(U) < 2:
            return now_ok()
        s = _score_of(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return now_ok()

    # 1. Search Geometric Progressions {0, b, b^2, ..., b^{k-1}}
    # We found that b=4, k=4 is very good.
    # Search b in [2, 30] and k in [2, 10]
    for k in range(2, 11):
        if not now_ok():
            break
        for b in range(2, 31):
            if not now_ok():
                break
            U = []
            val = 1
            for i in range(k):
                U.append(val)
                val *= b
            # Ensure 0 is in U (it is, U[0]=1? No, U[0]=1. We need 0.)
            # The set is {0, b, b^2...}. My loop generated {1, b, b^2...}.
            # Correction:
            U = [0]
            val = 1
            for i in range(1, k):
                val *= b
                U.append(val)
            try_candidate(U)

    # 2. Refine by adding one extra element to the best GP found so far
    # The best_U is likely a GP. Let's try adding x to it.
    base_U = best_U[:]
    if len(base_U) >= 2:
        max_base = max(base_U)
        # x should be > max_base or < min(base_U) (but min is 0, so x>0)
        # Or between elements.
        # Scan x in a range
        x_lo = 1
        x_hi = max_base * 5 + 10
        step = max(1, (x_hi - x_lo) // 1000)
        
        for x in range(x_lo, x_hi, step):
            if not now_ok():
                break
            U = base_U + [x]
            try_candidate(U)
        
        # Fine scan around the best if we found something better
        if best_U != base_U:
            # Identify the added element
            added = [x for x in best_U if x not in base_U]
            if added:
                x_best = added[0]
                lo = max(1, x_best - 50)
                hi = min(x_hi, x_best + 50)
                for x in range(lo, hi):
                    if not now_ok():
                        break
                    U = base_U + [x]
                    try_candidate(U)

    # 3. Hill Climbing
    rng = random.Random(42)
    current_U = best_U[:]
    
    def perturb(U):
        U = sorted(set(U))
        op = rng.random()
        if op < 0.2 and len(U) > 2:
            idx = rng.randrange(1, len(U))
            U = U[:idx] + U[idx+1:]
        elif op < 0.6:
            mx = max(U) if U else 0
            if rng.random() < 0.5:
                x = rng.randint(1, max(2, mx // 2))
            else:
                x = mx + rng.randint(1, max(2, mx // 3))
            U.append(x)
        else:
            idx = rng.randrange(1, len(U))
            old = U[idx]
            delta = rng.randint(1, max(1, old // 3))
            if rng.random() < 0.5:
                new = old - delta
            else:
                new = old + delta
            if new <= 0:
                new = old + delta
            U[idx] = new
            U = sorted(set(U))
        return U

    for _ in range(50000):
        if not now_ok():
            break
        U = perturb(current_U)
        if try_candidate(U):
            if _score_of(U) > best_score:
                current_U = U
        else:
            break

    return best_U

if __name__ == "__main__":
    U = construct()
    print(U)
    print(_score_of(U))
