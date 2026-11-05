import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    # 150s limit, use 110s to be safe
    deadline = start_time + 110.0
    
    total_words = q ** n
    
    # Precompute neighbors
    # For q=3, n=8, total_words=6561
    # This is fast enough
    neighbors = []
    neighbor_sets = []
    
    # If the problem is the standard one, we can hardcode some optimizations
    # But let's keep it generic for the precomp
    for idx in range(total_words):
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        
        nb = [idx]
        for i in range(n):
            orig = w[i]
            for val in range(q):
                if val == orig:
                    continue
                new_w = w.copy()
                new_w[i] = val
                new_idx = 0
                for j in range(n):
                    new_idx = new_idx * q + new_w[j]
                nb.append(new_idx)
        neighbors.append(nb)
        neighbor_sets.append(set(nb))

    # Initialization
    code_set = set()
    
    if q == 3 and n == 8 and radius == 1:
        # Direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for t in tetracode:
            for tail in itertools.product(range(3), repeat=4):
                w = t + list(tail)
                idx = 0
                for val in w:
                    idx = idx * 3 + val
                code_set.add(idx)
    else:
        # Fallback: Random init
        # Estimate size needed: total_words / (1 + n*(q-1))
        est_size = int(total_words / (1 + n * (q - 1)) * 2)
        code_set = set(random.sample(range(total_words), est_size))
        
    code_list = list(code_set)
    
    # Initialize counts
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code_set:
        for v in neighbors[c]:
            counts[v] += 1
            
    # Check validity
    # If min(counts) < 1, we need to add more
    # For the specific case, direct sum is valid.
    # For fallback, greedy add is better.
    if np.min(counts) < 1:
        # Greedy add
        while np.min(counts) < 1:
            # Find c that covers most uncovered
            # Simplified: pick random until covered
            # Better:
            best_c = -1
            max_new = 0
            # This loop is expensive if done naively.
            # For fallback, just add randoms until valid
            while np.min(counts) < 1:
                c = random.randint(0, total_words - 1)
                if c not in code_set:
                    code_set.add(c)
                    code_list.append(c)
                    for v in neighbors[c]:
                        counts[v] += 1
                    break

    best_size = len(code_set)
    best_code = set(code_set)

    # Pruning function
    def prune(cs, counts):
        changed = True
        while changed:
            changed = False
            for c in list(cs):
                # Check if redundant
                # min count in neighbors >= 2
                min_c = min(counts[v] for v in neighbors[c])
                if min_c >= 2:
                    cs.remove(c)
                    for v in neighbors[c]:
                        counts[v] -= 1
                    changed = True
        return cs

    # Initial Prune
    code_set = prune(code_set, counts)
    code_list = list(code_set)
    if len(code_set) < best_size:
        best_size = len(code_set)
        best_code = set(code_set)

    # Local Search
    while time.time() < deadline:
        if not code_list:
            break
            
        # Pick random codeword
        i = random.randint(0, len(code_list) - 1)
        c = code_list[i]
        
        # Compute uniquely covered words
        U_c = [v for v in neighbors[c] if counts[v] == 1]
        
        if not U_c:
            # Redundant, remove
            code_set.remove(c)
            code_list[i] = code_list[-1]
            code_list.pop()
            for v in neighbors[c]:
                counts[v] -= 1
            continue
            
        # Try to swap
        found_swap = False
        # Try a few random candidates
        for _ in range(15):
            c_prime = random.randint(0, total_words - 1)
            if c_prime in code_set:
                continue
                
            # Check if U_c is subset of neighbors[c_prime]
            if all(v in neighbor_sets[c_prime] for v in U_c):
                # Perform swap
                code_set.remove(c)
                code_set.add(c_prime)
                
                code_list[i] = c_prime
                
                for v in neighbors[c]:
                    counts[v] -= 1
                for v in neighbors[c_prime]:
                    counts[v] += 1
                    
                found_swap = True
                break
                
        if not found_swap:
            # Occasionally kick
            if random.random() < 0.05:
                c_new = random.randint(0, total_words - 1)
                if c_new not in code_set:
                    code_set.add(c_new)
                    code_list.append(c_new)
                    for v in neighbors[c_new]:
                        counts[v] += 1
                    
                    # Prune
                    code_set = prune(code_set, counts)
                    code_list = list(code_set)
                    
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)

    # Convert best_code to list of lists
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
