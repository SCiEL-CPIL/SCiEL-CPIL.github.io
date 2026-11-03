import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate all vectors with weight 3, entries in {-1, 1, 0}
    # Support is a set of 3 indices. Signs are 8 combinations.
    vectors = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([-1, 1], repeat=3):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int64)
    N = len(vectors)
    
    # Pre-compute dot products to find conflicts
    # Condition: u . v <= 1
    # Conflict if u . v > 1 (i.e., >= 2)
    
    # We can compute the full dot product matrix M = V @ V.T
    # But N=1320, matrix is 1320x1320. 1.7 million floats. Memory ~13MB. OK.
    dot_prods = vectors @ vectors.T
    
    # Build adjacency list
    # adj[i] is a set of j such that dot_prods[i,j] > 1
    adj = [set() for _ in range(N)]
    for i in range(N):
        # Find j > i where dot > 1
        # Using numpy mask
        mask = dot_prods[i] > 1
        # Exclude self
        mask[i] = False
        neighbors = np.where(mask)[0]
        for j in neighbors:
            adj[i].add(j)
            adj[j].add(i)
            
    # Greedy Independent Set
    # Strategy: Sort by degree (len(adj[i]))
    # Iterate, if active, select, deactivate neighbors.
    
    order = sorted(range(N), key=lambda x: len(adj[x]))
    
    best_set = []
    active = [True] * N
    
    # We can try multiple orders or just this one.
    # Let's try this one first.
    
    for i in order:
        if not active[i]:
            continue
        # Select i
        best_set.append(i)
        active[i] = False
        for j in adj[i]:
            if active[j]:
                active[j] = False
                
    # Convert indices to vectors
    result = vectors[best_set]
    
    # Check time
    if time.time() > deadline:
        return result
        
    # Try to improve with local search?
    # Or try random shuffles if time permits.
    
    current_size = len(best_set)
    
    # Try random shuffles
    # We have time?
    # Each run is O(N^2) to build adj (already done) + O(N log N) sort + O(N + E) greedy.
    # E is number of edges.
    # Greedy is fast.
    
    import random
    
    best_size = current_size
    best_indices = best_set
    
    # We can also try "Min-Degree in Current Graph" heuristic
    # But static degree is simpler.
    
    # Let's try a few random permutations
    # If we randomize the tie-breaking in the sorted order, we might get different sets.
    
    # Actually, we can just shuffle the list of nodes with the same degree?
    # Or just shuffle the entire list and then sort by degree?
    # Sorting by degree is deterministic.
    # To get variation, we can add random noise to the degree.
    
    for _ in range(10): # 10 restarts
        if time.time() > deadline:
            break
            
        # Add small random noise to degrees to break ties differently
        noisy_degrees = [len(adj[i]) + random.random() for i in range(N)]
        order_rand = sorted(range(N), key=lambda x: noisy_degrees[x])
        
        active_rand = [True] * N
        set_rand = []
        
        for i in order_rand:
            if not active_rand[i]:
                continue
            set_rand.append(i)
            active_rand[i] = False
            for j in adj[i]:
                if active_rand[j]:
                    active_rand[j] = False
                    
        if len(set_rand) > best_size:
            best_size = len(set_rand)
            best_indices = set_rand
            
    result = vectors[best_indices]
    return result
