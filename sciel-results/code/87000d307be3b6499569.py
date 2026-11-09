"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            g = gcd(a, b)
            gcd_mat[a][b] = g
            gcd_mat[b][a] = g
    
    # For each x, precompute the set of pairs (a,b) such that {x,a,b} is a bad triple
    # i.e., gcd(x,a) == gcd(x,b) == gcd(a,b)
    conflict_pairs = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcd_mat[x][a] == gcd_mat[x][b] == gcd_mat[a][b]:
                    conflict_pairs[x].add((a, b))
    
    # Conflict count for ordering
    conflict_counts = [0] * (N + 1)
    for x in range(1, N + 1):
        conflict_counts[x] = len(conflict_pairs[x])
    
    def can_add(x, selected_set):
        for a, b in conflict_pairs[x]:
            if a in selected_set and b in selected_set:
                return False
        return True
    
    def greedy(order, selected_set):
        for x in order:
            if x not in selected_set and can_add(x, selected_set):
                selected_set.add(x)
    
    def local_search(selected_set, max_iters=100):
        """Try to improve by removing one element and adding a different one."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            s_list = list(selected_set)
            current_size = len(s_list)
            
            for i in range(len(s_list)):
                removed = s_list[i]
                selected_set.discard(removed)
                
                best_add = None
                for x in numbers:
                    if x in selected_set:
                        continue
                    if can_add(x, selected_set):
                        if best_add is None:
                            best_add = x
                
                if best_add is not None:
                    selected_set.add(best_add)
                    if len(selected_set) > current_size:
                        improved = True
                    else:
                        # Revert
                        selected_set.discard(best_add)
                        selected_set.add(removed)
                else:
                    # Revert
                    selected_set.add(removed)
        
        return selected_set
    
    def local_search2(selected_set, max_iters=100):
        """Remove one, try to add two (net +1)."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            s_list = list(selected_set)
            current_size = len(s_list)
            
            for i in range(len(s_list)):
                removed = s_list[i]
                selected_set.discard(removed)
                
                # Try to add two elements
                candidates = [x for x in numbers if x not in selected_set and can_add(x, selected_set)]
                if len(candidates) < 2:
                    selected_set.add(removed)
                    continue
                
                best_pair = None
                best_total = -1
                for j in range(len(candidates)):
                    for k in range(j + 1, len(candidates)):
                        x, y = candidates[j], candidates[k]
                        # Check if x and y can both be added
                        if can_add(y, selected_set | {x}):
                            if best_pair is None:
                                best_pair = (x, y)
                
                if best_pair is not None:
                    x, y = best_pair
                    selected_set.add(x)
                    selected_set.add(y)
                    if len(selected_set) > current_size:
                        improved = True
                    else:
                        selected_set.discard(x)
                        selected_set.discard(y)
                        selected_set.add(removed)
                else:
                    selected_set.add(removed)
        
        return selected_set
    
    best_set = set()
    best_size = 0
    
    rng = random.Random(12345)
    
    # Multiple restarts
    while time.time() < deadline:
        # Create ordering
        r = rng.random()
        if r < 0.33:
            order = sorted(numbers, key=lambda x: conflict_counts[x])
        elif r < 0.66:
            order = sorted(numbers, key=lambda x: -conflict_counts[x])
        else:
            order = numbers[:]
        
        rng.shuffle(order)
        
        selected_set = set()
        greedy(order, selected_set)
        
        if len(selected_set) > best_size:
            best_size = len(selected_set)
            best_set = set(selected_set)
        
        # Local search
        if time.time() < deadline - 10.0:
            ls_set = local_search(set(selected_set), max_iters=50)
            if len(ls_set) > best_size:
                best_size = len(ls_set)
                best_set = set(ls_set)
        
        if time.time() < deadline - 10.0:
            ls_set2 = local_search2(set(selected_set), max_iters=30)
            if len(ls_set2) > best_size:
                best_size = len(ls_set2)
                best_set = set(ls_set2)
    
    # Fallback
    if not best_set:
        best_set = set()
        for x in range(1, N + 1):
            if can_add(x, best_set):
                best_set.add(x)
    
    return list(best_set)
# EVOLVE-BLOCK-END
