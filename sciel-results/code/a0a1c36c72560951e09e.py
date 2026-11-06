import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    N = 512
    
    def score(lst):
        return sum(1.0/(N - a) for a in lst)
    
    def is_coprime_with_all(a, lst):
        return all(gcd(a, b) == 1 for b in lst)
    
    # Multiple initializations
    candidates = []
    
    # Init 1: Greedy from largest
    chosen1 = []
    for a in range(N-1, 0, -1):
        if is_coprime_with_all(a, chosen1):
            chosen1.append(a)
    candidates.append(chosen1)
    
    # Init 2: Greedy from largest, but skip composites with many small factors?
    # Actually, let's try: pick primes first (largest primes), then composites
    primes = [a for a in range(N-1, 1, -1) if all(a % p != 0 for p in range(2, int(a**0.5)+1))]
    chosen2 = []
    for a in primes:
        if is_coprime_with_all(a, chosen2):
            chosen2.append(a)
    # Then add composites
    for a in range(N-1, 0, -1):
        if a in chosen2:
            continue
        if is_coprime_with_all(a, chosen2):
            chosen2.append(a)
    candidates.append(chosen2)
    
    # Init 3: Random restarts (if time permits)
    import random
    random.seed(42)
    for _ in range(5):
        random.shuffle(primes)
        chosen3 = []
        for a in primes:
            if is_coprime_with_all(a, chosen3):
                chosen3.append(a)
        for a in range(N-1, 0, -1):
            if a in chosen3:
                continue
            if is_coprime_with_all(a, chosen3):
                chosen3.append(a)
        candidates.append(chosen3)
    
    # Pick best candidate
    best = max(candidates, key=score)
    best_score = score(best)
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            a_i = best[i]
            remaining = [best[j] for j in range(len(best)) if j != i]
            best_b = None
            best_b_val = 0
            for b in range(N-1, 0, -1):
                if b in best:
                    continue
                if is_coprime_with_all(b, remaining):
                    val = 1.0/(N - b)
                    if val > best_b_val:
                        best_b_val = val
                        best_b = b
                if 1.0/(N - b) <= 1.0/(N - a_i):
                    break
            if best_b is not None and best_b_val > 1.0/(N - a_i):
                best[i] = best_b
                new_score = score(best)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
    
    return best
