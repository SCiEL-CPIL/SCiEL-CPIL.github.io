import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 40
    numbers = list(range(2, N + 1))
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    d = len(primes)
    
    vectors = []
    for n in numbers:
        v = [0] * d
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vectors.append(tuple(v))
        
    num_to_idx = {n: i for i, n in enumerate(numbers)}
    
    def is_valid_full(indices):
        k = len(indices)
        if k == 0: return True
        vecs = [vectors[i] for i in indices]
        sums = set()
        for mask in range(1 << k):
            s = [0] * d
            for i in range(k):
                if mask & (1 << i):
                    vi = vecs[i]
                    for j in range(d):
                        s[j] += vi[j]
            sums.add(tuple(s))
        return len(sums) == (1 << k)

    def can_add(current_sums, v):
        # Check if any s+v is in current_sums
        for s in current_sums:
            ns = tuple(s[j] + v[j] for j in range(d))
            if ns in current_sums:
                return False
        return True

    def update_sums(current_sums, v):
        new_sums = set()
        for s in current_sums:
            ns = tuple(s[j] + v[j] for j in range(d))
            new_sums.add(ns)
        return current_sums.union(new_sums)

    best_set = [num_to_idx[p] for p in primes]
    best_size = len(best_set)
    best_sums = None
    # Compute initial sums
    if best_set:
        vecs = [vectors[i] for i in best_set]
        sums = {tuple([0]*d)}
        for v in vecs:
            new_sums = set()
            for s in sums:
                ns = tuple(s[j] + v[j] for j in range(d))
                new_sums.add(ns)
            sums = sums.union(new_sums)
        best_sums = sums

    random.seed(42)
    current_set = list(best_set)
    current_sums = best_sums
    
    while time.time() < deadline:
        # Try to add
        if len(current_set) < N - 1:
            candidates = [i for i in range(N - 1) if i not in current_set]
            if candidates:
                idx = random.choice(candidates)
                v = vectors[idx]
                if can_add(current_sums, v):
                    current_set.append(idx)
                    current_sums = update_sums(current_sums, v)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                        best_sums = current_sums
        
        # Try to swap (full recheck)
        if len(current_set) > 0 and random.random() < 0.5:
            i_in = random.choice(current_set)
            candidates_out = [i for i in range(N - 1) if i not in current_set]
            if candidates_out:
                j_out = random.choice(candidates_out)
                new_set = [i for i in current_set if i != i_in] + [j_out]
                if is_valid_full(new_set):
                    current_set = new_set
                    # Recompute sums
                    vecs = [vectors[i] for i in current_set]
                    sums = {tuple([0]*d)}
                    for v in vecs:
                        new_sums = set()
                        for s in sums:
                            ns = tuple(s[j] + v[j] for j in range(d))
                            new_sums.add(ns)
                        sums = sums.union(new_sums)
                    current_sums = sums
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                        best_sums = current_sums
                        
    if not is_valid_full(best_set):
        best_set = [num_to_idx[p] for p in primes]
        
    return [int(numbers[i]) for i in best_set]
