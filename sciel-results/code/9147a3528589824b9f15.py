import numpy as np
import itertools
import time

def construct():
    D = 11
    deadline = time.time() + 140
    
    # Generate all weight 3 vectors
    # Support: choose 3 indices from 11
    # Signs: 2^3 = 8 combinations
    supports = list(itertools.combinations(range(D), 3))
    vectors = []
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.int8)
    n = len(V) # 1320
    
    # Compute dot products
    # V is (1320, 11)
    dots = V @ V.T
    
    # Conflict if dot > 1.5, i.e., dot >= 2
    conflict = (dots >= 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Greedy Independent Set with Local Search
    best_set = np.zeros(n, dtype=bool)
    best_size = 0
    
    for restart in range(20): # Multiple restarts
        if time.time() > deadline:
            break
            
        selected = np.zeros(n, dtype=bool)
        remaining = set(range(n))
        
        # Greedy phase: pick node with min degree in remaining subgraph
        # To speed up, we can just pick randomly from low degree nodes or sort once
        # Sorting once is O(n log n). Updating degrees is O(n^2) if naive.
        # With n=1320, n^2 is ~1.7M, which is fast.
        
        # Precompute initial degrees
        degrees = conflict.sum(axis=1).copy()
        
        while remaining:
            if time.time() > deadline:
                break
                
            # Find node with minimum degree in remaining
            # We can iterate and find min, or maintain a heap
            # Simple iteration is fine for 1320
            min_deg = float('inf')
            min_node = -1
            for node in remaining:
                if degrees[node] < min_deg:
                    min_deg = degrees[node]
                    min_node = node
            
            if min_node == -1:
                break
                
            # Select min_node
            selected[min_node] = True
            remaining.discard(min_node)
            
            # Update degrees and remaining
            neighbors = np.where(conflict[min_node] == 1)[0]
            for nb in neighbors:
                if nb in remaining:
                    degrees[nb] -= 1
                    remaining.discard(nb)
            # Note: We don't strictly need to update degrees of remaining nodes 
            # that are NOT neighbors, as we are only interested in min degree 
            # and we discard neighbors. But wait, the degree of a remaining node 
            # decreases only if one of its neighbors is removed. 
            # We remove min_node. So only neighbors of min_node have their degree decreased.
            # However, we also remove neighbors from 'remaining'.
            # So for nodes still in 'remaining', their degree has decreased by 1 
            # if they were neighbors of min_node.
            # The code above does this.
            
        # Local Search Phase
        # Try to swap one selected node for two unselected nodes
        improved = True
        while improved and time.time() < deadline:
            improved = False
            selected_indices = np.where(selected)[0]
            unselected_indices = np.where(~selected)[0]
            
            # Randomly shuffle to avoid bias
            np.random.shuffle(selected_indices)
            np.random.shuffle(unselected_indices)
            
            for i in selected_indices:
                if time.time() > deadline:
                    break
                # Temporarily remove i
                selected[i] = False
                
                # Try to add two unselected nodes
                # We need to find j, k in unselected such that 
                # they don't conflict with each other or with current selected (minus i)
                # And they don't conflict with i? No, i is removed.
                # But they must not conflict with the OTHER selected nodes.
                
                # To speed up, we can precompute conflicts with current selected set
                # But selected set changes.
                
                # Simple check:
                # For each j in unselected:
                #   if not conflicts with selected (excluding i):
                #      mark j as candidate
                # For each k in candidates (j+1):
                #   if not conflicts with selected (excluding i) and not conflicts with j:
                #      add j, k
                #      break
                
                # This is O(N_unselected^2) per swap attempt.
                # N_unselected is ~1000. 1000^2 = 1M.
                # If we do this for many i, it's slow.
                
                # Optimization:
                # Only try swaps for a subset of i's or j's.
                
                # Let's just try a few random swaps
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    j = np.random.choice(unselected_indices)
                    k = np.random.choice(unselected_indices)
                    if j == k:
                        continue
                    # Check if j and k can be added
                    # 1. j, k must not conflict with each other
                    if conflict[j, k]:
                        continue
                    # 2. j, k must not conflict with any other selected node
                    # selected currently has i=False
                    # Check j against selected
                    if np.any(conflict[j] & selected):
                        continue
                    # Check k against selected
                    if np.any(conflict[k] & selected):
                        continue
                    
                    # Add j, k
                    selected[j] = True
                    selected[k] = True
                    improved = True
                    break
                
                if not improved:
                    # Restore i
                    selected[i] = True
        
        if np.sum(selected) > best_size:
            best_size = np.sum(selected)
            best_set = selected.copy()
            
    result = V[best_set]
    
    if len(result) == 0:
        # Fallback
        return np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
        
    return result.astype(np.int8)
