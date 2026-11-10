# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin before 630s kill
    
    m = 205
    
    # Compute forbidden differences F = {x^2 mod 205 : x = 1..204} \ {0}
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    
    # Precompute conflict bitmasks for each vertex
    # conflict_mask[i] has bit j set if (i-j) mod m in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        for j in range(m):
            if i != j:
                d = (i - j) % m
                if d in forbidden:
                    conflict_mask[i] |= (1 << j)
    
    # All vertices bitmask
    all_vertices = (1 << m) - 1
    
    def is_valid_set(S):
        """Check if set S is valid (no forbidden differences)."""
        S_list = list(S)
        for idx in range(len(S_list)):
            for jdx in range(idx + 1, len(S_list)):
                d = (S_list[idx] - S_list[jdx]) % m
                if d in forbidden:
                    return False
        return True
    
    def greedy_build(order):
        """Build a valid set by greedily adding elements in given order."""
        S = set()
        for elem in order:
            ok = True
            for s in S:
                if (elem - s) % m in forbidden:
                    ok = False
                    break
            if ok:
                S.add(elem)
        return S
    
    def local_search(S, max_iters=10000):
        """Try to improve set S via add/swap/remove operations."""
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try to add an element
            added = False
            for elem in range(m):
                if elem in S:
                    continue
                ok = True
                for s in S:
                    if (elem - s) % m in forbidden:
                        ok = False
                        break
                if ok:
                    S.add(elem)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # Try swap: remove one element, add another
            S_list = list(S)
            random.shuffle(S_list)
            for remove_elem in S_list:
                if time.time() >= deadline:
                    return S
                S_temp = set(S)
                S_temp.discard(remove_elem)
                candidates = [e for e in range(m) if e not in S_temp]
                random.shuffle(candidates)
                for add_elem in candidates:
                    ok = True
                    for s in S_temp:
                        if (add_elem - s) % m in forbidden:
                            ok = False
                            break
                    if ok:
                        S = S_temp
                        S.add(add_elem)
                        improved = True
                        break
                if improved:
                    break
        
        return S
    
    # First, get a good initial solution using local search
    best_set = set()
    
    # Multiple randomized greedy passes with local search
    num_trials = 100
    for trial in range(num_trials):
        if time.time() >= deadline - 400:  # Leave time for branch and bound
            break
        
        order = list(range(m))
        random.shuffle(order)
        S = greedy_build(order)
        S = local_search(S, max_iters=3000)
        
        if len(S) > len(best_set):
            best_set = S
    
    # Convert best_set to bitmask
    best_mask = 0
    for s in best_set:
        best_mask |= (1 << s)
    best_size = len(best_set)
    
    # Branch and bound for maximum independent set
    # Use iterative deepening with time limit
    
    def bnb_search(available_mask, current_size, current_mask, best_state, depth_limit=0):
        """
        Branch and bound for maximum independent set.
        best_state: [best_size, best_mask]
        """
        if time.time() >= deadline - 10:
            return
        
        if available_mask == 0:
            if current_size > best_state[0]:
                best_state[0] = current_size
                best_state[1] = current_mask
            return
        
        # Upper bound: current_size + popcount(available_mask)
        if current_size + bin(available_mask).count('1') <= best_state[0]:
            return
        
        # Pick vertex with maximum degree in available subgraph (for pruning)
        best_vertex = -1
        best_degree = -1
        temp = available_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = bin(conflict_mask[v] & available_mask).count('1')
            if deg > best_degree:
                best_degree = deg
                best_vertex = v
        
        if best_vertex == -1:
            return
        
        # Branch 1: Include best_vertex
        new_avail = available_mask & ~(1 << best_vertex) & ~conflict_mask[best_vertex]
        bnb_search(new_avail, current_size + 1, current_mask | (1 << best_vertex), best_state, depth_limit + 1)
        
        # Branch 2: Exclude best_vertex
        new_avail = available_mask & ~(1 << best_vertex)
        bnb_search(new_avail, current_size, current_mask, best_state, depth_limit + 1)
    
    # Run branch and bound from scratch with time limit
    bnb_start = time.time()
    best_state = [best_size, best_mask]
    bnb_search(all_vertices, 0, 0, best_state)
    
    # If we found something better, update best_set
    if best_state[0] > len(best_set):
        new_best = set()
        mask = best_state[1]
        for i in range(m):
            if mask & (1 << i):
                new_best.add(i)
        best_set = new_best
        best_size = best_state[0]
    
    # Final local search polish on the best set
    if time.time() < deadline - 5:
        best_set = local_search(best_set, max_iters=10000)
    
    # Final verification
    if not is_valid_set(best_set):
        S_list = sorted(best_set)
        new_S = []
        for e in S_list:
            ok = True
            for f in new_S:
                if (e - f) % m in forbidden:
                    ok = False
                    break
            if ok:
                new_S.append(e)
        best_set = set(new_S)
    
    if len(best_set) == 0:
        best_set = {0}
    
    return np.array(sorted(best_set), dtype=np.int64)

# EVOLVE-BLOCK-END
