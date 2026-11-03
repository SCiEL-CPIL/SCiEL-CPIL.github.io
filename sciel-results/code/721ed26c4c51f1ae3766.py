import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    # Generate all weight 3 vectors
    # Support: choose 3 indices from 11
    # Signs: 2^3 = 8 patterns
    candidates = []
    indices = list(range(D))
    
    for support in combinations(indices, 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Precompute incompatibility matrix: two vectors are incompatible if their dot product > 1
    # For weight-3 vectors: ||u-v||^2 = 6 - 2*u.v >= 3 iff u.v <= 1
    # So incompatible if u.v > 1 (i.e., u.v >= 2)
    dots = candidates @ candidates.T  # N x N matrix of dot products
    incompat = (dots > 1).astype(np.int8)
    np.fill_diagonal(incompat, 0)
    
    # Maximum independent set via greedy + local search with vectorized operations
    # We'll use multiple random restarts with greedy initialization and local search
    
    def greedy_init(order):
        """Greedy selection given an ordering of candidates."""
        selected_mask = np.zeros(N, dtype=bool)
        for i in order:
            if selected_mask[i]:
                continue
            # Check if i is compatible with all selected
            # incompat[i] & selected_mask gives incompatible selected vertices
            if not np.any(incompat[i] & selected_mask):
                selected_mask[i] = True
        return selected_mask
    
    def local_search(mask, max_iters=5000):
        """Local search: try to add unselected vertices or swap to improve."""
        mask = mask.copy()
        selected_set = np.where(mask)[0]
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Try to add any unselected vertex that's compatible with all selected
            # For each unselected vertex, check if it's compatible with all selected
            unselected = np.where(~mask)[0]
            if len(unselected) == 0:
                break
            
            # Vectorized: for each unselected vertex, count incompatible selected vertices
            # incompat[unselected] @ mask gives count of incompatible selected for each unselected
            incompat_counts = incompat[unselected] @ mask
            addable = unselected[incompat_counts == 0]
            
            if len(addable) > 0:
                # Add one (or all) addable vertices
                for idx in addable:
                    mask[idx] = True
                continue
            
            # Try swaps: remove one selected, add two unselected
            # This is more complex; for now, just try removals that allow additions
            # Actually, let's try a different approach: random restarts with different orderings
            
            break
        
        return mask
    
    best_mask = None
    best_size = 0
    
    # Multiple restarts with different random orderings
    np.random.seed(42)
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Random ordering
        order = np.random.permutation(N)
        mask = greedy_init(order)
        size = np.sum(mask)
        
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
        
        # Local search
        mask = local_search(mask)
        size = np.sum(mask)
        
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
    
    if best_mask is None or best_size == 0:
        # Fallback to seed solution
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(D, dtype=np.int8)
                        v[i] = s1
                        v[j] = s2
                        vectors.append(v)
        return np.array(vectors, dtype=np.int8)
    
    selected_vectors = candidates[best_mask]
    return selected_vectors
