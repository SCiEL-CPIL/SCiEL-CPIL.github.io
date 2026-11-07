import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    L = 1
    for d in divisors:
        L = L * d // math.gcd(L, d)
    
    # Precompute multiples for each divisor in [0, L)
    # We need to handle the wrap-around and positivity.
    # For a given start s, candidates are multiples m in [s, s+L) such that m % d == 0 and m > 0.
    
    best_range = float('inf')
    best_solution = None
    
    # Time guard
    start_time = time.time()
    time_limit = 50.0 # 50 seconds to be safe
    
    for s in range(L):
        if time.time() - start_time > time_limit:
            break
            
        # Generate candidates for each divisor
        candidates = []
        valid = True
        for i, d in enumerate(divisors):
            # Find first multiple >= s
            # If s == 0, first positive multiple is d.
            # If s > 0, first multiple is ceil(s/d)*d.
            if s == 0:
                first = d
            else:
                first = ((s + d - 1) // d) * d
            
            cand_list = []
            m = first
            # We only need candidates up to s + L
            while m < s + L:
                if m > 0:
                    cand_list.append(m)
                m += d
            if not cand_list:
                valid = False
                break
            candidates.append(cand_list)
        
        if not valid:
            continue
            
        # Now find distinct assignment minimizing max
        # Use DFS with backtracking
        # Order divisors by length of candidate list (most constrained first)
        # Actually, just use the natural order or sort by len(candidates[i])
        indices = list(range(len(divisors)))
        indices.sort(key=lambda i: len(candidates[i]))
        
        used = set()
        current_max = 0
        
        def dfs(idx):
            nonlocal current_max, best_range, best_solution
            if idx == len(indices):
                if current_max < best_range + s: # We are comparing absolute values? 
                    # Wait, range is max_val - min_val.
                    # min_val is at least s.
                    # So range is current_max - s.
                    # We want to minimize current_max - s.
                    pass
                return True
            
            i = indices[idx]
            found = False
            for val in candidates[i]:
                if val in used:
                    continue
                # Pruning: if val - s >= best_range, we can't improve (since min is >= s)
                # Actually, if val - s >= best_range, and we haven't finished, we might still have a smaller max?
                # No, max will be at least val. So range >= val - s.
                if val - s >= best_range:
                    break # Since candidates are sorted, subsequent vals are larger
                
                used.add(val)
                old_max = current_max
                current_max = max(current_max, val)
                
                if dfs(idx + 1):
                    found = True
                    break
                    
                used.remove(val)
                current_max = old_max
                
            return found
        
        # We need to track the actual values for the solution
        # Let's modify DFS to store the solution
        solution = [None] * len(divisors)
        
        def dfs2(idx):
            nonlocal current_max, best_range, best_solution
            if idx == len(indices):
                range_val = current_max - s
                if range_val < best_range:
                    best_range = range_val
                    best_solution = solution[:]
                return True
            
            i = indices[idx]
            found = False
            for val in candidates[i]:
                if val in used:
                    continue
                if val - s >= best_range:
                    break
                
                used.add(val)
                solution[i] = val
                old_max = current_max
                current_max = max(current_max, val)
                
                if dfs2(idx + 1):
                    found = True
                    break
                    
                used.remove(val)
                solution[i] = None
                current_max = old_max
                
            return found
        
        dfs2(0)
        
    # Return the best solution found
    if best_solution is None:
        # Fallback: just return the divisors themselves? No, they must be distinct and multiples.
        # The divisors are distinct and multiples of themselves.
        # Range of [6,10,14,15,21,22,26,33,35,39] is 39-6=33.
        return divisors[:]
    
    # Ensure types are int
    return [int(x) for x in best_solution]

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    L = 1
    for d in divisors:
        L = L * d // math.gcd(L, d)
    
    best_range = float('inf')
    best_solution = None
    
    start_time = time.time()
    time_limit = 50.0
    
    for s in range(L):
        if time.time() - start_time > time_limit:
            break
            
        candidates = []
        valid = True
        for i, d in enumerate(divisors):
            if s == 0:
                first = d
            else:
                first = ((s + d - 1) // d) * d
            
            cand_list = []
            m = first
            while m < s + L:
                if m > 0:
                    cand_list.append(m)
                m += d
            if not cand_list:
                valid = False
                break
            candidates.append(cand_list)
        
        if not valid:
            continue
            
        indices = list(range(len(divisors)))
        indices.sort(key=lambda i: len(candidates[i]))
        
        used = set()
        current_max = 0
        solution = [None] * len(divisors)
        
        def dfs2(idx):
            nonlocal current_max, best_range, best_solution
            if idx == len(indices):
                range_val = current_max - s
                if range_val < best_range:
                    best_range = range_val
                    best_solution = solution[:]
                return True
            
            i = indices[idx]
            found = False
            for val in candidates[i]:
                if val in used:
                    continue
                if val - s >= best_range:
                    break
                
                used.add(val)
                solution[i] = val
                old_max = current_max
                current_max = max(current_max, val)
                
                if dfs2(idx + 1):
                    found = True
                    break
                    
                used.remove(val)
                solution[i] = None
                current_max = old_max
                
            return found
        
        dfs2(0)
        
    if best_solution is None:
        return divisors[:]
    
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
