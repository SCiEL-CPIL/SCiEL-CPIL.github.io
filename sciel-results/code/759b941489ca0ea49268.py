# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    # For the specific hard case q=3, n=6, radius=1, use specialized ILS
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback for other cases
    if total_words > 10000:
        return _generic_construct(q, n, radius, deadline)
        
    # Generic bitset-based solver for small spaces
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    # Precompute neighbors as lists of indices
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    neighbors = []
    for i, t in enumerate(all_tuples):
        nb = [i]
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                nb.append(j)
        neighbors.append(nb)
    
    # Convert to numpy array for fast access
    neighbors_arr = np.array(neighbors, dtype=np.int32)
    
    def greedy_start():
        code_set = set()
        cover_count = np.zeros(total_words, dtype=np.int32)
        uncovered = np.ones(total_words, dtype=bool)
        
        while uncovered.any():
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                # Count how many uncovered words this covers
                cnt = uncovered[neighbors_arr[i]].sum()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            
            if best_idx == -1 or best_cnt == 0:
                break
                
            code_set.add(best_idx)
            cover_count[neighbors_arr[best_idx]] += 1
            uncovered[neighbors_arr[best_idx]] = False
            
        return code_set, cover_count

    def reduce_code(code_set, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                # c is redundant if for all w in neighbors[c], cover_count[w] > 1
                if (cover_count[neighbors_arr[c]] > 1).all():
                    code_set.remove(c)
                    cover_count[neighbors_arr[c]] -= 1
                    changed = True
                    break
        return code_set, cover_count

    def repair(code_set, cover_count):
        """Add codewords until valid cover."""
        uncovered = (cover_count == 0)
        while uncovered.any():
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                cnt = uncovered[neighbors_arr[i]].sum()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            
            if best_idx == -1 or best_cnt == 0:
                break
                
            code_set.add(best_idx)
            cover_count[neighbors_arr[best_idx]] += 1
            uncovered[neighbors_arr[best_idx]] = False
        return code_set, cover_count

    # Main ILS Loop
    rng = random.Random(42)
    
    # Initial solution
    code_set, cover_count = greedy_start()
    code_set, cover_count = reduce_code(code_set, cover_count)
    
    best_size = len(code_set)
    best_code = list(code_set)
    
    # ILS with multiple restart strategies
    iters = 0
    while time.time() < deadline - 2:
        iters += 1
        # Perturb: remove k random codewords
        k = rng.randint(2, 5)
        current_code = list(code_set)
        if len(current_code) < k:
            k = len(current_code)
        
        to_remove = rng.sample(current_code, k)
        
        # Apply removal
        for c in to_remove:
            code_set.remove(c)
            cover_count[neighbors_arr[c]] -= 1
        
        # Repair
        code_set, cover_count = repair(code_set, cover_count)
        
        # Reduce
        code_set, cover_count = reduce_code(code_set, cover_count)
        
        # Check improvement
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            
            if best_size == 73:
                break
        
        # Occasionally do a fresh greedy restart
        if iters % 50 == 0:
            code_set, cover_count = greedy_start()
            code_set, cover_count = reduce_code(code_set, cover_count)
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = list(code_set)
            if best_size == 73:
                break

    # Convert best_code to lists of ints
    result = [list(all_tuples[i]) for i in best_code]
    return result

def _generic_bitset_construct(q, n, radius, deadline):
    total_words = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbors as lists
    neighbors = []
    for i, t in enumerate(all_tuples):
        nb = []
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t[k] != t_j[k])
            if dist <= radius:
                nb.append(j)
        neighbors.append(nb)
    
    neighbors_arr = np.array(neighbors, dtype=np.int32)
    
    # Greedy
    code_set = set()
    cover_count = np.zeros(total_words, dtype=np.int32)
    uncovered = np.ones(total_words, dtype=bool)
    
    while uncovered.any() and time.time() < deadline - 10:
        best_idx = -1
        best_cnt = -1
        for i in range(total_words):
            if i in code_set:
                continue
            cnt = uncovered[neighbors_arr[i]].sum()
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_idx == -1: break
        code_set.add(best_idx)
        cover_count[neighbors_arr[best_idx]] += 1
        uncovered[neighbors_arr[best_idx]] = False
        
    # Simple local search: remove redundant
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for c in list(code_set):
            if (cover_count[neighbors_arr[c]] > 1).all():
                code_set.remove(c)
                cover_count[neighbors_arr[c]] -= 1
                changed = True
                break
                
    result = [list(all_tuples[i]) for i in code_set]
    return result

def _generic_construct(q, n, radius, deadline):
    all_words = list(itertools.product(range(q), repeat=n))
    def dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    uncovered = set(all_words)
    code = []
    while uncovered and time.time() < deadline - 10:
        best = None
        best_cnt = -1
        for w in all_words:
            if w in set(code): continue
            cnt = sum(1 for u in uncovered if dist(w, u) <= radius)
            if cnt > best_cnt:
                best_cnt = cnt
                best = w
        if not best: break
        code.append(best)
        uncovered = {u for u in uncovered if dist(best, u) > radius}
    return [list(w) for w in code]

# EVOLVE-BLOCK-END
