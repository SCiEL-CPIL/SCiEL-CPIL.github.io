import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 150s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum or random
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays of int32 for fast operations
    neighbors_list = []
    neighbors_set = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbors_set.append(set(nb))

    # ---------- Greedy Set-Cover Construction with Randomization ----------
    def greedy_construct_random(top_k=5, seed=None):
        """
        Greedy set cover with random tie-breaking among top_k candidates.
        """
        if seed is not None:
            random.seed(seed)
        
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        # First pick: 0 (symmetry)
        code.append(0)
        code_set.add(0)
        coverage[neighbors_list[0]] += 1
        uncovered_mask[neighbors_list[0]] = False
        
        while uncovered_mask.any():
            best_candidates = []
            max_gain = 0
            
            for c in range(N):
                if c in code_set:
                    continue
                cnt = np.sum(uncovered_mask[neighbors_list[c]])
                if cnt > max_gain:
                    max_gain = cnt
                    best_candidates = [c]
                elif cnt == max_gain and max_gain > 0:
                    best_candidates.append(c)
            
            if max_gain == 0:
                break
            
            if len(best_candidates) > top_k:
                chosen = random.choice(best_candidates)
            else:
                chosen = random.choice(best_candidates)
                
            code.append(chosen)
            code_set.add(chosen)
            coverage[neighbors_list[chosen]] += 1
            uncovered_mask[neighbors_list[chosen]] = False
            
        return code, code_set, coverage, uncovered_mask

    # Run multiple greedy constructions to find a good start
    best_initial_code = None
    best_initial_size = float('inf')
    best_initial_set = None
    best_initial_cov = None
    best_initial_um = None
    
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 30:  # Keep time for local search
            break
        c, cs, cov, um = greedy_construct_random(top_k=5, seed=i)
        if len(c) < best_initial_size:
            best_initial_size = len(c)
            best_initial_code = c
            best_initial_set = cs
            best_initial_cov = cov
            best_initial_um = um
            
    if best_initial_code is None:
        c, cs, cov, um = greedy_construct_random(top_k=1, seed=0)
        best_initial_code = c
        best_initial_set = cs
        best_initial_cov = cov
        best_initial_um = um

    code = best_initial_code
    code_set = best_initial_set
    coverage = best_initial_cov
    uncovered_mask = best_initial_um

    # ---------- Local Search: Removal and Kick ----------
    
    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            # Build candidate pool from neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])
            
            # Find best candidate
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = len(neighbors_set[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    if best_count == len(temp_uncovered):
                        break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            temp_uncovered.difference_update(neighbors_set[best_c])
        
        return added

    def apply_code_change(removed_list, added_list):
        """
        Apply the change to global state.
        """
        nonlocal code, code_set, coverage
        for c in removed_list:
            code.remove(c)
            code_set.remove(c)
            coverage[neighbors_list[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1

    def try_remove():
        nonlocal code, code_set, coverage
        removed = False
        for c in list(code):
            if is_redundant(c):
                apply_code_change([c], [])
                removed = True
        return removed

    def kick():
        nonlocal code, code_set, coverage
        if not code:
            return False
            
        old_size = len(code)
        
        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]
        
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            apply_code_change([c_remove], [])
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)
        
        new_size = old_size - 1 + len(added)
        
        if new_size < old_size:
            apply_code_change([c_remove], added)
            return True
        else:
            return False

    def kick_2():
        nonlocal code, code_set, coverage
        if len(code) < 2:
            return False
            
        old_size = len(code)
        
        i1 = random.randint(0, len(code) - 1)
        i2 = random.randint(0, len(code) - 1)
        if i1 == i2:
            i2 = (i2 + 1) % len(code)
        
        c1 = code[i1]
        c2 = code[i2]
        
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c1]] -= 1
        temp_coverage[neighbors_list[c2]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) > 0:
            uncovered_set = set(uncovered.tolist())
            added = greedy_repair(uncovered_set)
            
            new_size = old_size - 2 + len(added)
            
            if new_size < old_size:
                apply_code_change([c1, c2], added)
                return True
        else:
            # Perfect removal
            apply_code_change([c1, c2], [])
            return True
            
        return False

    # Main Loop
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = try_remove()
        
        # 2. Kick
        if time.time() > deadline:
            break
        
        kicks_tried = 0
        success = False
        max_kicks = 20
        while kicks_tried < max_kicks:
            if kick():
                success = True
                kicks_tried = 0
            else:
                kicks_tried += 1
            if time.time() > deadline:
                break
        
        if not success and len(code) > 2:
            # 2-removal kick
            kick_2()

    result = [idx_to_tuple(c) for c in code]
    return result
