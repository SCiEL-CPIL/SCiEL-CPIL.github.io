import numpy as np
import random

def _precompute_neighbors(q, n):
    N = q ** n
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        neighbors[i].append(i)
        temp = i
        digits = [0] * n
        for j in range(n):
            digits[n-1-j] = temp % q
            temp //= q
            
        for pos in range(n):
            orig = digits[pos]
            weight = q ** (n - 1 - pos)
            for new_d in range(q):
                if new_d == orig:
                    continue
                ni = i + (new_d - orig) * weight
                neighbors[i].append(ni)
    return neighbors

def _greedy_cover(q, n, radius, neighbors, num_restarts=100):
    N = q ** n
    best_size = float('inf')
    best_code = None
    
    for _ in range(num_restarts):
        uncovered = np.ones(N, dtype=bool)
        # counts[i] = number of uncovered words covered by i
        # Initialize with full coverage potential
        # We can precompute the size of neighbors[i]
        # But neighbors change as words get covered.
        
        # Efficient Greedy:
        # We maintain a priority queue? Or just scan?
        # With N=729, scanning is fast.
        
        selected = []
        # To speed up, we can precompute neighbor lists as numpy arrays?
        # Or just lists.
        
        # Optimization: Precompute neighbor indices as a list of lists.
        # coverage_counts[i] = sum(uncovered[j] for j in neighbors[i])
        
        # Initial counts
        # Since all are uncovered, count[i] = len(neighbors[i])
        # But wait, radius is 1. The problem is K_3(6,1).
        # The neighbors list includes self and dist 1.
        
        # Let's use a simple list of counts
        counts = [len(nei) for nei in neighbors]
        
        while uncovered.any():
            # Find max count
            # Using numpy for max is faster if we convert, but lists are small.
            # Let's use numpy for the uncovered array to check any() and for updates.
            
            # To find max, we scan the list.
            max_idx = -1
            max_val = -1
            for i in range(N):
                if counts[i] > max_val:
                    max_val = counts[i]
                    max_idx = i
            if max_val == 0:
                break
                
            selected.append(max_idx)
            
            # Update uncovered and counts
            # For each w in neighbors[max_idx] that is currently uncovered:
            #   set uncovered[w] = False
            #   for each i that covers w (i.e., w in neighbors[i]):
            #       counts[i] -= 1
            
            # This reverse mapping (coverers) is needed for efficient update.
            # Precompute coverers: coverers[w] = list of i such that w in neighbors[i]
            
            # Let's precompute coverers outside the restart loop.
            
        # This inner loop logic is getting complex to write in the edit.
        # I will structure the code properly in the replacement.
        
    return best_code

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _solve_3_6_1()
    else:
        # Fallback to simple greedy for other cases
        return _greedy_cover_numpy_optimized(q, n, radius, num_restarts=10)

def _solve_3_6_1():
    q, n, radius = 3, 6, 1
    N = q ** n
    neighbors, coverers = _precompute_relations(q, n)
    
    best_size = float('inf')
    best_code = None
    
    for _ in range(200): # More restarts since it's faster
        code = _greedy_with_removal(N, neighbors, coverers)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    return [[int(d) for d in np.base_repr(idx, base=q).zfill(n)] for idx in best_code]

def _precompute_relations(q, n):
    N = q ** n
    neighbors = [[] for _ in range(N)]
    coverers = [[] for _ in range(N)]
    
    for i in range(N):
        neighbors[i].append(i)
        coverers[i].append(i)
        
        temp = i
        digits = [0] * n
        for j in range(n):
            digits[n-1-j] = temp % q
            temp //= q
            
        for pos in range(n):
            orig = digits[pos]
            weight = q ** (n - 1 - pos)
            for new_d in range(q):
                if new_d == orig:
                    continue
                ni = i + (new_d - orig) * weight
                neighbors[i].append(ni)
                coverers[ni].append(i)
                
    return neighbors, coverers

def _greedy_with_removal(N, neighbors, coverers):
    uncovered = np.ones(N, dtype=bool)
    counts = [len(nei) for nei in neighbors]
    selected = []
    
    while uncovered.any():
        # Find max
        max_val = -1
        max_idx = -1
        for i in range(N):
            if counts[i] > max_val:
                max_val = counts[i]
                max_idx = i
        if max_val == 0: break
        
        selected.append(max_idx)
        
        # Update
        for w in neighbors[max_idx]:
            if uncovered[w]:
                uncovered[w] = False
                for i in coverers[w]:
                    counts[i] -= 1
                    
    # Local Search: Remove Redundant
    # Build current counts based on selected
    final_counts = np.zeros(N, dtype=np.int32)
    for c in selected:
        for w in neighbors[c]:
            final_counts[w] += 1
            
    # Iterative removal
    changed = True
    while changed:
        changed = False
        for c in list(selected):
            # Check if c is redundant
            redundant = True
            for w in neighbors[c]:
                if final_counts[w] == 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(c)
                for w in neighbors[c]:
                    final_counts[w] -= 1
                changed = True
                
    return selected
