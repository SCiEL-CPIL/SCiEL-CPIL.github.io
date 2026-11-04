import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    D = 11
    indices = list(range(D))
    
    candidates = []
    
    # Weight 2 vectors
    for support in combinations(indices, 2):
        for signs in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    # Weight 3 vectors
    for support in combinations(indices, 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Compute dot products
    dots = candidates @ candidates.T
    
    # Conflict condition: dot >= 2
    # (Derived from min distance >= max norm for weights 2 and 3)
    conflict_matrix = (dots >= 2).astype(np.int8)
    np.fill_diagonal(conflict_matrix, 0)
    
    # Build bitmasks for conflicts
    # conflict_mask[i] is an integer where bit j is 1 if i and j conflict
    conflict_masks = [0] * N
    for i in range(N):
        # Get indices of conflicts
        neighbors = np.where(conflict_matrix[i] > 0)[0]
        mask = 0
        for j in neighbors:
            mask |= (1 << j)
        conflict_masks[i] = mask
        
    # Greedy initialization for lower bound
    def greedy_mis(order):
        selected = set()
        remaining_mask = (1 << N) - 1
        # We need to simulate removal
        # Actually, easier to just iterate
        current_mask = (1 << N) - 1
        sel_count = 0
        sel_list = []
        for i in order:
            if (current_mask >> i) & 1:
                sel_count += 1
                sel_list.append(i)
                # Remove i and all its neighbors from current_mask
                current_mask &= ~(conflict_masks[i] | (1 << i))
        return sel_count, sel_list

    # Get an initial solution using greedy with random orders
    best_len = 0
    best_indices = []
    
    np.random.seed(42)
    
    # Try a few greedy orders to get a good lower bound
    # Order by degree (descending)
    degrees = np.sum(conflict_matrix, axis=1)
    order_deg = np.argsort(-degrees)
    
    for r in range(20):
        if r == 0:
            order = order_deg
        else:
            order = np.random.permutation(N)
            
        l, idxs = greedy_mis(order)
        if l > best_len:
            best_len = l
            best_indices = idxs
            
    # Branch and Bound
    # We sort vertices by degree to improve pruning
    # But the bitmasks are fixed. We can just pass a mask.
    # To make it faster, we can reorder the bits? No, that's complex.
    # Just use the mask as is.
    
    global_best_len = best_len
    global_best_set = set(best_indices)
    
    def search(mask, current_set, current_size):
        nonlocal global_best_len, global_best_set
        
        if time.time() > deadline:
            return
            
        if current_size + mask.bit_count() <= global_best_len:
            return
            
        if mask == 0:
            if current_size > global_best_len:
                global_best_len = current_size
                global_best_set = set(current_set)
            return
            
        # Pick a vertex to branch on
        # Heuristic: pick the vertex with the highest degree in the subgraph induced by mask?
        # Or just the lowest index set bit?
        # Let's pick the lowest index set bit for simplicity, but higher degree might be better for pruning.
        # Finding highest degree in mask requires iterating set bits.
        
        # Simple heuristic: pick the first set bit
        # This is slow if it leads to deep paths.
        # Let's try to pick a vertex that has many conflicts within the mask.
        
        # To avoid overhead, just pick the lowest set bit.
        # v = (mask & -mask).bit_length() - 1
        v = (mask & -mask).bit_length() - 1
        
        # Branch 1: Include v
        new_mask_in = mask & ~(conflict_masks[v] | (1 << v))
        search(new_mask_in, current_set + [v], current_size + 1)
        
        # Branch 2: Exclude v
        new_mask_out = mask & ~(1 << v)
        search(new_mask_out, current_set, current_size)

    # Start search with all vertices available
    # We already have a good lower bound, so pruning will be effective.
    # However, the recursion depth is N (1540). Python recursion limit is 1000.
    # We need to increase recursion limit or use iterative DFS.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Use iterative DFS to avoid stack overflow and allow better control?
    # Or just increase limit.
    
    # The mask starts with all bits set.
    start_mask = (1 << N) - 1
    search(start_mask, [], 0)
    
    # Construct result
    result_vectors = candidates[global_best_set]
    
    # Check deadline
    if time.time() > deadline - 5:
        # Fallback to greedy result if search didn't finish or improve
        if best_indices:
            result_vectors = candidates[best_indices]
            
    return result_vectors
