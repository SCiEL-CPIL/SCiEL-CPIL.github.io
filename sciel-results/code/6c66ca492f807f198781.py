import time
import random
import itertools

# Global cache for neighbors
_NEIGHBORS_CACHE = {}

def _get_neighbors(n, q):
    key = (n, q)
    if key in _NEIGHBORS_CACHE:
        return _NEIGHBORS_CACHE[key]
        
    neighbors = []
    total = q ** n
    for w in range(total):
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        
        nbrs = [w]
        for i in range(n):
            original = digits[i]
            for val in range(q):
                if val == original:
                    continue
                new_digits = digits[:]
                new_digits[i] = val
                new_w = 0
                for d in new_digits:
                    new_w = new_w * q + d
                nbrs.append(new_w)
        neighbors.append(nbrs)
        
    _NEIGHBORS_CACHE[key] = neighbors
    return neighbors

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return [list(h) for h in tetracode]
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: Return all words (valid but large)
        return [list(itertools.product(range(q), repeat=n))]
        
    deadline = time.time() + 110
    neighbors = _get_neighbors(n, q)
    total_words = q ** n
    all_words = list(range(total_words))
    
    best_code = None
    best_size = float('inf')
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        uncovered = [1] * total_words
        covered_count = 0
        code = []
        
        while covered_count < total_words:
            max_covered = -1
            best_w = -1
            for w in all_words:
                cnt = 0
                for nb in neighbors[w]:
                    if uncovered[nb]:
                        cnt += 1
                # Add noise for random tie-breaking
                score = cnt + rng.random() * 0.001
                if score > max_covered:
                    max_covered = score
                    best_w = w
            
            code.append(best_w)
            for nb in neighbors[best_w]:
                if uncovered[nb]:
                    uncovered[nb] = 0
                    covered_count += 1
        
        return code

    def greedy_reduce(code, seed):
        rng = random.Random(seed)
        counts = [0] * total_words
        for c in code:
            for nb in neighbors[c]:
                counts[nb] += 1
        
        code_list = list(code)
        rng.shuffle(code_list)
        
        removed_set = set()
        for c in code_list:
            for nb in neighbors[c]:
                counts[nb] -= 1
            
            if 0 not in counts:
                removed_set.add(c)
            else:
                for nb in neighbors[c]:
                    counts[nb] += 1
        
        return [c for c in code_list if c not in removed_set]

    seed = 0
    while time.time() < deadline:
        # Construct
        code = greedy_construct(seed)
        # Reduce
        code = greedy_reduce(code, seed)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # Local search: Swap
        if best_code and len(best_code) > 156:
            rng = random.Random(seed + 5000)
            if len(best_code) > 0:
                idx = rng.randint(0, len(best_code) - 1)
                c_remove = best_code[idx]
                w_add = rng.randint(0, total_words - 1)
                
                if w_add != c_remove:
                    new_code_set = set(best_code)
                    new_code_set.remove(c_remove)
                    new_code_set.add(w_add)
                    new_code = list(new_code_set)
                    
                    new_code = greedy_reduce(new_code, seed + 6000)
                    
                    if len(new_code) < best_size:
                        best_size = len(new_code)
                        best_code = new_code
        
        seed += 1
        # If we hit 186, we might want to slow down construction and focus on local search?
        # But construction is cheap.
        
    if best_code is None:
        code = greedy_construct(0)
        code = greedy_reduce(code, 0)
        best_code = code

    result = []
    for w in best_code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        result.append(digits)
        
    return result
