import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_all(lst):
    return reduce(_lcm, lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin under 90s limit
    
    # Fallback known dissociated set
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Elements and their integer representations
    elements = list(range(1, 23))
    N = 22
    L = _lcm_all(elements)
    nums = [L // x for x in elements]
    
    def _time_ok():
        return (time.time() - start_time) < deadline
    
    def _try_greedy(order):
        """Greedy build given a specific order of elements."""
        current_set = []
        current_sums = {0}
        for idx in order:
            n = nums[idx]
            # Check collision: does adding n create a duplicate?
            collision = False
            for s in current_sums:
                if (s + n) in current_sums:
                    collision = True
                    break
            if not collision:
                new_sums = current_sums | {s + n for s in current_sums}
                current_set.append(elements[idx])
                current_sums = new_sums
        return current_set
    
    # 1. Try deterministic greedy orders first (fast)
    # Descending index (largest elements first, smallest reciprocals)
    order_desc = list(range(N - 1, -1, -1))
    set_desc = _try_greedy(order_desc)
    if len(set_desc) > best_size:
        best_set = set_desc
        best_size = len(set_desc)
    
    # Ascending index
    order_asc = list(range(N))
    set_asc = _try_greedy(order_asc)
    if len(set_asc) > best_size:
        best_set = set_asc
        best_size = len(set_asc)
    
    # 2. Branch and bound search for larger sets
    # We'll search for sets of size > best_size
    # Use DFS with pruning
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(pos, cur_list, cur_sums, remaining_count):
        nonlocal best_set, best_size
        
        # Prune: even if we add all remaining, can't beat best
        if len(cur_list) + remaining_count <= best_size:
            return
        if not _time_ok():
            return
        
        if pos == N:
            if len(cur_list) > best_size:
                best_size = len(cur_list)
                best_set = cur_list[:]
            return
        
        # Option 1: include element at pos
        n = nums[pos]
        collision = False
        for s in cur_sums:
            if (s + n) in cur_sums:
                collision = True
                break
        
        if not collision:
            new_sums = cur_sums | {s + n for s in cur_sums}
            cur_list.append(elements[pos])
            dfs(pos + 1, cur_list, new_sums, remaining_count - 1)
            cur_list.pop()
        
        if not _time_ok():
            return
        
        # Option 2: exclude element at pos
        dfs(pos + 1, cur_list, cur_sums, remaining_count - 1)
    
    # Start branch-and-bound with descending order (larger elements first)
    # This tends to find valid sets faster
    dfs(0, [], {0}, N)
    
    # 3. If we still have time, try randomized greedy for additional exploration
    if _time_ok():
        import random
        attempts = 0
        max_attempts = 50000
        while _time_ok() and attempts < max_attempts:
            attempts += 1
            order = list(range(N))
            random.shuffle(order)
            
            current_set = []
            current_sums = {0}
            for idx in order:
                n = nums[idx]
                collision = False
                for s in current_sums:
                    if (s + n) in current_sums:
                        collision = True
                        break
                if not collision:
                    new_sums = current_sums | {s + n for s in current_sums}
                    current_set.append(elements[idx])
                    current_sums = new_sums
            
            if len(current_set) > best_size:
                best_set = current_set
                best_size = len(current_set)
    
    return best_set
