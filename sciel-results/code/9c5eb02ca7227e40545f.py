import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin
    n_max = 4096

    def build_sums(S):
        sums = set()
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(S[i] + S[j] + S[l])
        return sums

    def can_add(x, S, sums):
        # Check 3x
        if 3 * x in sums:
            return False
        # Check 2x + s
        for s in S:
            if 2 * x + s in sums:
                return False
        # Check x + s_i + s_j
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                if x + S[i] + S[j] in sums:
                    return False
        return True

    def add_x(x, S, sums):
        S.append(x)
        sums.add(3 * x)
        for s in S[:-1]:
            sums.add(2 * x + s)
        k = len(S)
        for i in range(k - 1):
            for j in range(i, k - 1):
                sums.add(x + S[i] + S[j])

    def greedy_random(start_list, deadline, seed):
        random.seed(seed)
        S = list(start_list)
        sums = build_sums(S)
        existing = set(S)
        
        # Candidates: all numbers not in S
        candidates = [x for x in range(1, n_max + 1) if x not in existing]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, S, sums):
                add_x(x, S, sums)
                existing.add(x)
        
        return S

    # Strong seed: Powers of 3
    pow3 = []
    v = 1
    while v <= n_max:
        pow3.append(v)
        v *= 3

    best = []
    best_len = 0

    def consider(S):
        nonlocal best, best_len
        if len(S) > best_len:
            best = sorted(S)
            best_len = len(S)

    # Run multiple randomized greedy passes
    # We can also try starting from empty, but pow3 is a strong base.
    # Let's try a few seeds.
    
    # Pass 1-10: Randomized greedy from pow3
    for i in range(20):
        if time.time() > deadline:
            break
        res = greedy_random(pow3, deadline, seed=1000 + i)
        consider(res)

    # Pass 11-20: Randomized greedy from empty (might find a different structure)
    for i in range(10):
        if time.time() > deadline:
            break
        res = greedy_random([], deadline, seed=2000 + i)
        consider(res)

    # Local search improvement: Try to swap elements to increase density
    # If we have a set S, can we remove y and add x (where x < y or just x not in S) to maintain validity?
    # Actually, if we remove y, we free up sums. Then we can add x.
    # This is complex to implement efficiently in the remaining time.
    # The randomized greedy from a good seed is usually quite effective.

    return best[:80]
