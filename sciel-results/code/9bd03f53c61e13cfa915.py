import time
import random

def _compute_scores(A):
    """Compute |A+A| and |A*A|, return max."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _try_improve(A, deadline):
    """Local search: try replacing one element with a better one."""
    n = len(A)
    current_score = _compute_scores(A)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for idx in range(n):
            if time.time() > deadline:
                break
            old = A[idx]
            # Try candidates: smaller numbers, nearby numbers
            candidates = set()
            for c in range(1, 10001):
                if c != old:
                    candidates.add(c)
            # To be efficient, try a subset first
            # Try all values 1..10000 but that's 10000 * 20 = 200k evaluations per pass
            # Each evaluation is O(n^2) = 400 ops, so 80M ops per pass - too slow
            # Instead, try a smart subset
            
            # Try: values near old, small values, and some random
            trial_set = set()
            for c in range(1, min(100, old + 1)):
                if c != old:
                    trial_set.add(c)
            for c in range(max(1, old - 50), min(10001, old + 51)):
                if c != old:
                    trial_set.add(c)
            # Also try some structured values
            for p in [2, 3, 5, 7, 11, 13, 17, 19]:
                power = 1
                while power <= 10000:
                    if power != old:
                        trial_set.add(power)
                    power *= p
            
            for c in sorted(trial_set):
                if time.time() > deadline:
                    break
                A[idx] = c
                if len(set(A)) != n:
                    A[idx] = old
                    continue
                new_score = _compute_scores(A)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                A[idx] = old
    return A, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    random.seed(42)
    
    # Initial solution: powers of 2
    A = [2**i for i in range(20)]
    best_A = list(A)
    best_score = _compute_scores(A)
    
    # Also try other initializations
    inits = [
        list(range(1, 21)),  # seed
        [2**i for i in range(20)],  # powers of 2
        [3**i for i in range(20) if 3**i <= 10000] + list(range(1, 20 - len([3**i for i in range(20) if 3**i <= 10000]))),  # powers of 3
    ]
    
    # Fix inits to be exactly 20 distinct values <= 10000
    inits = []
    # Seed
    inits.append(list(range(1, 21)))
    # Powers of 2
    inits.append([2**i for i in range(20)])
    # Powers of 3 (up to 10000): 1,3,9,27,81,243,729,2187,6561 -> 9 values, need 11 more
    p3 = [3**i for i in range(20) if 3**i <= 10000]
    if len(p3) < 20:
        remaining = sorted(set(range(1, 10001)) - set(p3))
        p3 = p3 + remaining[:20 - len(p3)]
    inits.append(p3)
    # Geometric with ratio 2 but offset
    inits.append([2 * 2**i for i in range(20)])
    
    for init in inits:
        if time.time() > deadline:
            break
        if len(set(init)) != 20 or any(x > 10000 for x in init):
            continue
        score = _compute_scores(init)
        if score < best_score:
            best_score = score
            best_A = list(init)
    
    # Local search from best
    A = list(best_A)
    A, score = _try_improve(A, deadline)
    if score < best_score:
        best_score = score
        best_A = list(A)
    
    # More local search passes
    for _ in range(3):
        if time.time() > deadline:
            break
        A = list(best_A)
        A, score = _try_improve(A, deadline)
        if score < best_score:
            best_score = score
            best_A = list(A)
    
    return best_A
