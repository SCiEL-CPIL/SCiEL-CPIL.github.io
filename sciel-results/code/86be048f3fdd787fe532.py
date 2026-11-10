import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    candidates = list(range(2, 41))
    
    vectors = {}
    for n in candidates:
        v = [0] * 12
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vectors[n] = tuple(v)
        
    base_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    remaining = [n for n in candidates if n not in base_set]
    
    # Calculate initial sums for base_set
    current_sums = {tuple([0]*12)}
    for n in base_set:
        vec = vectors[n]
        new_sums = set()
        for s in current_sums:
            ns = tuple(a + b for a, b in zip(s, vec))
            new_sums.add(ns)
        current_sums = new_sums
        
    best_set = base_set[:]
    best_size = len(best_set)
    
    def is_valid_addition(current_sums, vec):
        new_sums = set()
        for s in current_sums:
            ns = tuple(a + b for a, b in zip(s, vec))
            if ns in current_sums:
                return None
            new_sums.add(ns)
        return new_sums

    def dfs(idx, current_set, current_sums):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if len(current_set) + (len(remaining) - idx) <= best_size:
            return
            
        if idx == len(remaining):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
            
        # Exclude
        dfs(idx + 1, current_set, current_sums)
        
        # Include
        n = remaining[idx]
        vec = vectors[n]
        new_sums = is_valid_addition(current_sums, vec)
        if new_sums is not None:
            current_set.append(n)
            dfs(idx + 1, current_set, new_sums)
            current_set.pop()
            
    dfs(0, best_set[:], current_sums)
    
    return best_set

# EVOLVE-BLOCK-END
