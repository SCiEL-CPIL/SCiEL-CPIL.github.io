import numpy as np
import itertools
import random
import time

D = 11

def construct():
    # Generate all weight 4 vectors
    candidates = []
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            candidates.append(v)
    
    N = len(candidates)
    
    # Precompute adjacency list (conflicts)
    # Map vector to index? Or just use list of lists of indices.
    # Since vectors are small, we can use a dictionary to map tuple(v) to index.
    # But we need to find conflicts for each i.
    # Brute force O(N^2) is 5280^2 = 27M. Might be slow in Python if not optimized.
    # 27M dot products of length 11.
    # Numpy can do this fast?
    # Or we can use the combinatorial structure to generate conflicts directly.
    
    # Direct generation is faster and safer.
    # For each vector i, find its support and signs.
    # Then generate the 56 neighbors + self.
    
    adj = [[] for _ in range(N)]
    
    # We need a way to find the index of a generated neighbor.
    # Create a dictionary: tuple(v) -> index
    v_to_idx = {}
    for i, v in enumerate(candidates):
        v_to_idx[tuple(v)] = i
        
    for i, v in enumerate(candidates):
        # Self conflict
        adj[i].append(i)
        
        # Find support and signs of v
        sup = []
        signs = []
        for j in range(D):
            if v[j] != 0:
                sup.append(j)
                signs.append(v[j])
                
        # sup has 4 elements.
        # Generate 3-overlap conflicts
        # Choose 3 from 4
        for k in range(4):
            # Exclude sup[k]
            shared_sup = [sup[j] for j in range(4) if j != k]
            shared_signs = [signs[j] for j in range(4) if j != k]
            
            # The 4th index of neighbor must be in remaining 7
            remaining_indices = [j for j in range(D) if j not in sup]
            
            for new_idx in remaining_indices:
                for new_sign in [1, -1]:
                    # Construct neighbor
                    nv = [0] * D
                    for idx, val in zip(shared_sup, shared_signs):
                        nv[idx] = val
                    nv[new_idx] = new_sign
                    
                    # This neighbor has dot 3 with v
                    # Check if it exists (it should)
                    # But wait, is it possible that this neighbor is the same as another generated one?
                    # Yes, it's in the list.
                    n_idx = v_to_idx.get(tuple(nv), -1)
                    if n_idx != -1 and n_idx != i:
                        adj[i].append(n_idx)
                        
    # Greedy Independent Set
    best_set = []
    deadline = time.time() + 100 # 100s budget
    
    # We can try multiple random orders
    # Or just one good order?
    # Random restarts are cheap.
    
    for restart in range(1000):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        
        selected = []
        available = set(range(N))
        
        for i in order:
            if i in available:
                selected.append(i)
                available.remove(i)
                for j in adj[i]:
                    available.discard(j)
                    
        if len(selected) > len(best_set):
            best_set = selected
            # Optional: if we reach a very high number, we can stop?
            # But maybe we can do better.
            
    # Convert indices to vectors
    result = np.array([candidates[i] for i in best_set], dtype=np.int64)
    return result
