# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses randomized greedy construction with multiple restarts and local search refinement.
    """
    m = 205
    deadline = time.time() + 590  # 10s safety margin before 600s kill
    
    # Precompute forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    
    # Precompute conflict matrix: conflict[i][j] = True if (i-j) mod m in forbidden
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            if d1 in forbidden:
                conflict[i, j] = True
    
    best_set = set()
    best_size = 0
    
    def greedy_construct(order):
        """Build a set using greedy method with given order of candidates."""
        s = set()
        for candidate in order:
            ok = True
            for existing in s:
                if conflict[candidate, existing]:
                    ok = False
                    break
            if ok:
                s.add(candidate)
        return s
    
    def can_add(x, s):
        """Check if x can be added to set s without conflict."""
        for y in s:
            if conflict[x, y]:
                return False
        return True
    
    def local_search(s, max_iters=2000):
        """
        Try to improve set s via local search: add/remove/swap operations.
        Uses a more efficient approach with bitmasks for conflict checking.
        """
        s = set(s)
        current_size = len(s)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a new element
            candidates_to_add = [x for x in range(m) if x not in s]
            random.shuffle(candidates_to_add)
            for x in candidates_to_add:
                if can_add(x, s):
                    s.add(x)
                    current_size += 1
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one element, add another
                s_list = list(s)
                random.shuffle(s_list)
                for y in s_list:
                    s_without_y = s - {y}
                    candidates_to_add = [x for x in range(m) if x not in s_without_y and x != y]
                    random.shuffle(candidates_to_add)
                    for x in candidates_to_add:
                        if can_add(x, s_without_y):
                            s = s_without_y
                            s.add(x)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try remove one and add two (net gain of 1)
                s_list = list(s)
                random.shuffle(s_list)
                for y in s_list:
                    s_without_y = s - {y}
                    candidates = [x for x in range(m) if x not in s_without_y and x != y]
                    random.shuffle(candidates)
                    found_pair = False
                    for i, x1 in enumerate(candidates):
                        if found_pair:
                            break
                        if not can_add(x1, s_without_y):
                            continue
                        for x2 in candidates[i+1:]:
                            if found_pair:
                                break
                            if not can_add(x2, s_without_y):
                                continue
                            if conflict[x1, x2]:
                                continue
                            s = s_without_y
                            s.add(x1)
                            s.add(x2)
                            found_pair = True
                            break
                    if found_pair:
                        improved = True
                        break
            
            if not improved:
                break
        
        return s
    
    def two_opt_local_search(s, max_iters=500):
        """
        More aggressive local search: try all pairs of swaps.
        """
        s = set(s)
        current_size = len(s)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            s_list = list(s)
            random.shuffle(s_list)
            
            for i, y1 in enumerate(s_list):
                if time.time() > deadline:
                    break
                for j, y2 in enumerate(s_list):
                    if i == j:
                        continue
                    s_without = s - {y1, y2}
                    candidates = [x for x in range(m) if x not in s_without and x not in (y1, y2)]
                    random.shuffle(candidates)
                    
                    # Try to find two elements to add
                    found_pair = False
                    for k, x1 in enumerate(candidates):
                        if found_pair:
                            break
                        if not can_add(x1, s_without):
                            continue
                        for x2 in candidates[k+1:]:
                            if found_pair:
                                break
                            if not can_add(x2, s_without):
                                continue
                            if conflict[x1, x2]:
                                continue
                            s = s_without
                            s.add(x1)
                            s.add(x2)
                            found_pair = True
                            break
                    if found_pair:
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return s
    
    # Run multiple randomized greedy constructions
    num_restarts = 80
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 12345)
        
        # Create a random order
        order = list(range(m))
        random.shuffle(order)
        
        # Greedy construction
        s = greedy_construct(order)
        
        # Local search refinement
        s = local_search(s)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # Additional local search passes on the best set
    for _ in range(10):
        if time.time() > deadline:
            break
        s = local_search(best_set)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
        
        # Try more aggressive search
        s = two_opt_local_search(best_set)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
