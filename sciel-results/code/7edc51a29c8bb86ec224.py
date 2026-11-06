import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    n = 192
    universe = set(range(1, n + 1))
    
    def is_valid(S):
        """Check if set S satisfies the constraint."""
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for a in S_list:
            others = [v for v in S_list if v != a]
            if len(others) < 3:
                continue
            
            # Build pair sum map: residue -> list of (i, j) indices
            pair_map = {}
            m = len(others)
            for i in range(m):
                xi = others[i]
                for j in range(i + 1, m):
                    xj = others[j]
                    s = (xi + xj) % a
                    if s not in pair_map:
                        pair_map[s] = []
                    pair_map[s].append((i, j))
            
            # For each z, check if (-z) % a has a pair not involving z
            for z_idx in range(m):
                z = others[z_idx]
                target = (-z) % a
                if target in pair_map:
                    for i, j in pair_map[target]:
                        if i != z_idx and j != z_idx:
                            return False
        
        return True
    
    def greedy_build(order):
        """Build a set using greedy with given order of candidates."""
        selected = []
        sel_set = set()
        for x in order:
            if x in sel_set:
                continue
            candidate = sel_set | {x}
            if is_valid(candidate):
                selected.append(x)
                sel_set.add(x)
        return selected
    
    # Phase 1: Multiple greedy restarts with different orderings
    best_S = set()
    best_size = 0
    
    # Descending order (large numbers first - they're less likely to divide sums)
    if time.time() < deadline - 5:
        order_desc = list(range(n, 0, -1))
        S1 = greedy_build(order_desc)
        best_S = set(S1)
        best_size = len(S1)
    
    # Ascending order (small numbers first)
    if time.time() < deadline - 5:
        order_asc = list(range(1, n + 1))
        S2 = greedy_build(order_asc)
        if len(S2) > best_size:
            best_S = set(S2)
            best_size = len(S2)
    
    # A few random orderings
    for trial in range(3):
        if time.time() >= deadline - 5:
            break
        order_rand = list(range(1, n + 1))
        random.shuffle(order_rand)
        S3 = greedy_build(order_rand)
        if len(S3) > best_size:
            best_S = set(S3)
            best_size = len(S3)
    
    # Phase 2: Local search - try to add elements and do swaps
    S = best_S
    
    # First, try to add any remaining elements (multiple passes)
    for pass_num in range(10):
        if time.time() >= deadline - 3:
            break
        U_not_S = universe - S
        if not U_not_S:
            break
        
        added_any = False
        # Try in descending order first (large numbers are safer)
        for x in sorted(U_not_S, reverse=True):
            if time.time() >= deadline - 3:
                break
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
                added_any = True
                break
        
        if not added_any:
            # Try ascending order
            for x in sorted(U_not_S):
                if time.time() >= deadline - 3:
                    break
                new_S = S | {x}
                if is_valid(new_S):
                    S = new_S
                    added_any = True
                    break
        
        if not added_any:
            break
    
    # Phase 3: Intensive local search with swaps
    improved = True
    swap_attempts_per_iter = 20
    while improved and time.time() < deadline - 2:
        improved = False
        U_not_S = universe - S
        if not U_not_S:
            break
        
        S_list = list(S)
        U_not_S_list = list(U_not_S)
        
        for _ in range(swap_attempts_per_iter):
            if time.time() >= deadline - 2:
                break
            
            # Random swap
            if not S_list or not U_not_S_list:
                break
            remove_idx = random.randint(0, len(S_list) - 1)
            remove_candidate = S_list[remove_idx]
            add_candidate = random.choice(U_not_S_list)
            
            new_S = (S - {remove_candidate}) | {add_candidate}
            if is_valid(new_S):
                S = new_S
                improved = True
                
                # After a swap, try to add more elements
                for x in sorted(universe - S, reverse=True):
                    if time.time() >= deadline - 2:
                        break
                    new_S2 = S | {x}
                    if is_valid(new_S2):
                        S = new_S2
                        break
                
                break
    
    # Phase 4: Final cleanup - try one more pass of additions
    for x in sorted(universe - S, reverse=True):
        if time.time() >= deadline - 1:
            break
        new_S = S | {x}
        if is_valid(new_S):
            S = new_S
    
    return list(S)
