import time
import sys
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s kill)
    
    n = 256
    
    def is_valid_addition(v, S_arr, S_set):
        """Check if v can be added to S. S_arr is sorted, S_set is a set."""
        # For each multiple m of v in [2, 2n], check if exists b,c in S with b+c=m, b<=c
        # b <= c means b <= m/2
        # b in S and m-b in S, with b <= m-b
        # i.e., b in S, b <= m/2, and (m-b) in S
        L = len(S_arr)
        if L == 0:
            return True
        
        # Precompute: for each possible sum s in [2, 2n], store whether there exists a pair in S summing to s
        # Actually, let's just iterate multiples of v
        max_sum = 2 * n
        for m in range(v, max_sum + 1, v):
            # Check if exists b in S with b <= m//2 and (m-b) in S
            # b must be >= 1 and <= m-1 (since c = m-b >= 1)
            # b <= m - b => b <= m/2
            b_min = max(1, m - n)  # since c = m-b <= n
            b_max = min(m // 2, n)
            if b_min > b_max:
                continue
            # Check if any b in [b_min, b_max] is in S and (m-b) is in S
            # Iterate over smaller range
            # For efficiency, iterate over elements of S in range
            for b in S_arr:
                if b > b_max:
                    break
                if b < b_min:
                    continue
                c = m - b
                if c in S_set:
                    # Found a conflict
                    return False
        return True
    
    def greedy_construct(order):
        """Greedy construction following given order."""
        S = []
        S_set = set()
        for v in order:
            if time.time() > deadline:
                break
            if is_valid_addition(v, S, S_set):
                S.append(v)
                S_set.add(v)
        return S, S_set
    
    # Phase 1: Multiple greedy orders
    best_size = 0
    best_set = set()
    best_S = []
    
    # Order 1: descending
    S, S_set = greedy_construct(list(range(n, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S_set.copy()
        best_S = S[:]
    
    # Order 2: ascending
    S, S_set = greedy_construct(list(range(1, n + 1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S_set.copy()
        best_S = S[:]
    
    # Order 3: large numbers first (n, n-1, ..., n/2+1, then smaller)
    # The insight: for large a, the gap between 2a and 3a is a, so fewer conflicts
    # Try: all numbers > n/2 first, then n/2 down to 1
    order3 = list(range(n, n // 2, -1)) + list(range(n // 2, 0, -1))
    S, S_set = greedy_construct(order3)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S_set.copy()
        best_S = S[:]
    
    # Order 4: numbers in (n/3, n] first, then (n/4, n/3], etc.
    # Actually, let's try: numbers > 2n/3 first, then n/2 to 2n/3, etc.
    order4 = list(range(n, 2 * n // 3, -1)) + list(range(2 * n // 3, n // 2, -1)) + list(range(n // 2, 0, -1))
    S, S_set = greedy_construct(order4)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S_set.copy()
        best_S = S[:]
    
    # Random orders
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S, S_set = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S_set.copy()
            best_S = S[:]
    
    # Phase 2: Local search - try to add more elements
    # For each element not in best_set, try adding it (removing conflicts if needed)
    # Simple 1-swap: remove one element, try to add two new ones
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_set_copy = best_set.copy()
        S_arr_copy = sorted(S_set_copy, reverse=True)
        
        for v in range(1, n + 1):
            if time.time() > deadline:
                break
            if v in S_set_copy:
                continue
            # Try adding v
            if is_valid_addition(v, S_arr_copy, S_set_copy):
                best_set.add(v)
                best_size += 1
                S_arr_copy = sorted(best_set, reverse=True)
                S_set_copy = best_set.copy()
                improved = True
    
    # Phase 3: 1-swap local search
    # Try removing one element and see if we can add more
    for _ in range(5):
        if time.time() > deadline:
            break
        best_set_copy = best_set.copy()
        best_S_arr = sorted(best_set_copy, reverse=True)
        
        found_improvement = False
        for remove_v in list(best_set_copy):
            if time.time() > deadline:
                break
            # Remove remove_v
            S_after_remove = [x for x in best_S_arr if x != remove_v]
            S_set_after_remove = best_set_copy - {remove_v}
            
            # Try to add as many as possible from remaining elements
            added_any = False
            for add_v in range(n, 0, -1):
                if time.time() > deadline:
                    break
                if add_v in S_set_after_remove:
                    continue
                if is_valid_addition(add_v, S_after_remove, S_set_after_remove):
                    S_after_remove.append(add_v)
                    S_set_after_remove.add(add_v)
                    added_any = True
            
            if len(S_set_after_remove) > len(best_set_copy):
                best_set = S_set_after_remove.copy()
                best_size = len(best_set)
                found_improvement = True
                break
        
        if not found_improvement:
            break
    
    if not best_set:
        return [256]
        
    return sorted(best_set)
