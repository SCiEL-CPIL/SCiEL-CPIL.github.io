import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 24
    elements = list(range(1, N + 1))
    
    # 1. Precompute reciprocals
    recip = {i: Fraction(1, i) for i in elements}
    
    # 2. Find all bad hyper-edges
    bad_hyperedges = set()
    
    # Helper to find subsets of 'avail' summing to 'target'
    def find_subsets_sum(target, avail, idx, current_sum, current_mask, results):
        if idx == len(avail):
            if current_sum == target:
                results.append(current_mask)
            return
        
        x = avail[idx]
        
        # Exclude
        find_subsets_sum(target, avail, idx + 1, current_sum, current_mask, results)
        
        # Include
        # Prune if sum exceeds target
        if current_sum + recip[x] <= target:
            find_subsets_sum(target, avail, idx + 1, current_sum + recip[x], current_mask | (1 << (x-1)), results)

    for a in elements:
        target = recip[a]
        # Sort ascending to try large reciprocals first for better pruning
        avail = [x for x in elements if x != a]
        avail.sort() # 1, 2, 3 ...
        
        results = []
        find_subsets_sum(target, avail, 0, Fraction(0), 0, results)
        
        for sub_mask in results:
            full_mask = sub_mask | (1 << (a-1))
            bad_hyperedges.add(full_mask)

    # Build precursors
    precursors = {x: set() for x in elements}
    for mask in bad_hyperedges:
        temp = mask
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length()
            pre_mask = mask ^ lsb
            precursors[x].add(pre_mask)
            temp ^= lsb
    
    # Convert to lists for faster iteration
    for x in elements:
        precursors[x] = list(precursors[x])

    # 3. Initial Solution (Greedy)
    best_size = 0
    best_mask = 0
    
    for _ in range(100):
        random.shuffle(elements)
        current_mask = 0
        current_size = 0
        for x in elements:
            can_add = True
            for pre_mask in precursors[x]:
                if (pre_mask & ~current_mask) == 0:
                    can_add = False
                    break
            if can_add:
                current_mask |= (1 << (x-1))
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            
        if best_size == N:
            break

    # 4. Branch and Bound
    # Order elements by number of precursors descending
    search_order = list(range(1, N+1))
    search_order.sort(key=lambda x: len(precursors[x]), reverse=True)
    
    def search(idx, current_mask, current_size):
        nonlocal best_size, best_mask
        
        # Pruning 1: Upper bound
        if current_size + (N - idx) <= best_size:
            return
            
        if idx == N:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        x = search_order[idx]
        
        # Option 1: Exclude x
        search(idx + 1, current_mask, current_size)
        
        # Option 2: Include x
        can_include = True
        for pre_mask in precursors[x]:
            if (pre_mask & ~current_mask) == 0:
                can_include = False
                break
                
        if can_include:
            if time.time() > deadline:
                return
            search(idx + 1, current_mask | (1 << (x-1)), current_size + 1)

    search(0, 0, 0)
    
    result = []
    for i in range(1, N+1):
        if best_mask & (1 << (i-1)):
            result.append(i)
            
    return result
