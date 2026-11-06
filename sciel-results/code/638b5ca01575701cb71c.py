import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def is_sidon(S):
        n = len(S)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_size(S):
        n = len(S)
        for k in range(n, 0, -1):
            for subset in combinations(range(n), k):
                if is_sidon([S[i] for i in subset]):
                    return k
        return 0
    
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    import random
    random.seed(42)
    
    # Exhaustive: all consecutive blocks
    for base in range(0, 117):
        if time.time() > deadline:
            break
        S = list(range(base, base + 12))
        sz = max_sidon_size(S)
        if sz < best_size:
            best_size = sz
            best_set = S
    
    # Random search in various ranges
    for _ in range(50000):
        if time.time() > deadline:
            break
        width = random.randint(12, 40)
        base = random.randint(0, max(0, 128 - width))
        S = sorted(random.sample(range(base, min(129, base + width + 1)), 12))
        sz = max_sidon_size(S)
        if sz < best_size:
            best_size = sz
            best_set = S
    
    # Structured: pairs that create collisions
    for a in range(0, 60):
        for d in range(1, 15):
            if time.time() > deadline:
                break
            S = []
            for i in range(12):
                S.append(a + i * d)
            if max(S) <= 128:
                S = sorted(S)
                sz = max_sidon_size(S)
                if sz < best_size:
                    best_size = sz
                    best_set = S
    
    return best_set
