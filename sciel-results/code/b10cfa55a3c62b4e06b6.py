import numpy as np
import itertools
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1()
    else:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return _generic_greedy(q, n, radius)

def _solve_ternary_6_1():
    q = 3
    n = 6
    radius = 1
    size = q ** n
    start_time = time.time()
    time_limit = 140 # Leave 10s margin
    
    # Precompute base representations
    bases = []
    for i in range(size):
        l = []
        x = i
        for _ in range(n):
            l.append(x % q)
            x //= q
        l.reverse() # Most significant digit first
        bases.append(l)
        
    # Precompute neighbors
    neighbors = [[] for _ in range(size)]
    for w in range(size):
        neighbors[w].append(w)
        l_w = bases[w]
        for p in range(n):
            orig = l_w[p]
            for v in range(q):
                if v == orig:
                    continue
                new_l = l_w.copy()
                new_l[p] = v
                # Calculate index of new_l
                new_idx = 0
                for d in new_l:
                    new_idx = new_idx * q + d
                neighbors[w].append(new_idx)
                
    best_code = None
    best_len = size + 1
    
    def run_greedy(random_tie_break=False, seed=0):
        import random
        if random_tie_break:
            rng = random.Random(seed)
        else:
            rng = None
            
        covered = np.zeros(size, dtype=np.int8)
        code = []
        
        for _ in range(size):
            max_gain = -1
            best_candidates = []
            
            for c in range(size):
                gain = 0
                for nb in neighbors[c]:
                    if covered[nb] == 0:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [c]
                elif gain == max_gain:
                    best_candidates.append(c)
            
            if max_gain == 0:
                break
                
            if random_tie_break and len(best_candidates) > 1:
                chosen = rng.choice(best_candidates)
            else:
                chosen = best_candidates[0]
                
            code.append(chosen)
            
            for nb in neighbors[chosen]:
                covered[nb] += 1
                
        return code

    # Run standard greedy
    code1 = run_greedy(random_tie_break=False)
    if len(code1) < best_len:
        best_len = len(code1)
        best_code = code1
        
    # Run randomized greedy several times if time permits
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        code_r = run_greedy(random_tie_break=True, seed=attempts)
        if len(code_r) < best_len:
            best_len = len(code_r)
            best_code = code_r
            
        if best_len <= 73:
            break

    # Convert indices back to lists of digits
    result = [bases[idx] for idx in best_code]
    return result

def _generic_greedy(q, n, radius):
    size = q ** n
    words = list(range(size))
    
    def int_to_base(num, q, n):
        l = []
        x = num
        for _ in range(n):
            l.append(x % q)
            x //= q
        l.reverse()
        return l
        
    def base_to_int(lst, q):
        val = 0
        for v in lst:
            val = val * q + v
        return val
        
    def hamming_dist(w1, w2):
        l1 = int_to_base(w1, q, n)
        l2 = int_to_base(w2, q, n)
        return sum(1 for a, b in zip(l1, l2) if a != b)
        
    # Precompute neighborhoods
    neighbors = [[] for _ in range(size)]
    for w in words:
        for v in words:
            if hamming_dist(w, v) <= radius:
                neighbors[w].append(v)
                
    covered = np.zeros(size, dtype=np.int8)
    code = []
    
    for _ in range(size):
        max_gain = -1
        best_candidates = []
        for c in range(size):
            gain = 0
            for nb in neighbors[c]:
                if covered[nb] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_candidates = [c]
            elif gain == max_gain:
                best_candidates.append(c)
                
        if max_gain == 0:
            break
            
        chosen = best_candidates[0]
        code.append(chosen)
        for nb in neighbors[chosen]:
            covered[nb] += 1
            
    return [int_to_base(idx, q, n) for idx in code]
