import math
import itertools
import time
import random

def score_set(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a+b)
            diffs.add(a-b)
    size_sum = len(sums)
    size_diff = len(diffs)
    if size_sum == 0:
        return 0
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
    return 1 + math.log(ratio) / denom

def local_search(U, max_val, time_limit):
    """Try to improve U by replacing elements one at a time."""
    U = sorted(set(U))
    if 0 not in U:
        return U
    best = U
    best_score = score_set(best)
    start = time.time()
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(1, len(best)):  # Don't replace 0
            for new_val in range(1, max_val + 1):
                if new_val in best:
                    continue
                trial = best[:i] + [new_val] + best[i+1:]
                trial.sort()
                s = score_set(trial)
                if s > best_score + 1e-12:
                    best = trial
                    best_score = s
                    improved = True
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin from 150s
    
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Phase 1: Exhaustive search for small sets
    # Sizes 3-5 with reasonable max values
    configs = [
        (3, 50),
        (4, 40),
        (5, 30),
        (6, 20),
        (7, 15),
        (8, 12),
    ]
    
    for n, lim in configs:
        if time.time() > deadline:
            break
        count = 0
        for comb in itertools.combinations(range(1, lim+1), n-1):
            if time.time() > deadline:
                break
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
            count += 1
    
    # Phase 2: Local search on the best found set
    if time.time() < deadline - 5:
        max_val = max(best_set) + 15
        improved = local_search(best_set, max_val, deadline - time.time() - 3)
        s = score_set(improved)
        if s > best_score:
            best_score = s
            best_set = improved
    
    # Phase 3: Try some random initial candidates and local search
    if time.time() < deadline - 10:
        random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            n = random.randint(4, 8)
            lim = random.randint(10, 30)
            U = [0] + sorted(random.sample(range(1, lim+1), min(n-1, lim)))
            s = score_set(U)
            if s > best_score - 0.01:  # Only polish if reasonably close
                max_val = max(U) + 10
                improved = local_search(U, max_val, min(2.0, deadline - time.time() - 3))
                s2 = score_set(improved)
                if s2 > best_score:
                    best_score = s2
                    best_set = improved
    
    return best_set
