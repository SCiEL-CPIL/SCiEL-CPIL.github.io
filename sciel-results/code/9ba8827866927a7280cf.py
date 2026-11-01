import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate all vectors of weight 4
    # Support: 4 indices
    # Signs: 2^4 = 16 combinations
    vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            vectors.append(v)
            
    V = np.array(vectors)
    N = len(V)
    
    # Compute dot product matrix
    # V shape (N, D)
    # M shape (N, N)
    # M[i, j] = V[i] dot V[j]
    M = V @ V.T
    
    # Condition: min dist >= max norm
    # Max norm is sqrt(4) = 2
    # Min dist squared >= 4
    # ||V[i] - V[j]||^2 = 4 + 4 - 2 M[i,j] = 8 - 2 M[i,j] >= 4
    # 4 >= 2 M[i,j] => M[i,j] <= 2
    
    # We want a subset of indices such that for all i, j in subset, M[i,j] <= 2
    # Note: M[i,i] = 4. This is > 2.
    # The condition applies to pairwise distance of DISTINCT vectors.
    # So we only care about i != j.
    
    # Conflict matrix: C[i,j] = True if M[i,j] > 2 (and i != j)
    # We need to find a large independent set in the graph defined by C.
    
    # Greedy approach with multiple random restarts
    best_set = []
    best_size = 0
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(N)
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if idx conflicts with any selected
            # Conflict if M[idx, sel] > 2
            # We can vectorize this check against the current selected list
            if not selected:
                selected.append(idx)
                selected_set.add(idx)
            else:
                # Get dot products of idx with all selected
                # M[idx, selected]
                dots = M[idx, selected]
                # If all dots <= 2, add idx
                if np.all(dots <= 2):
                    selected.append(idx)
                    selected_set.add(idx)
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
    if best_size == 0:
        # Fallback to seed solution if something went wrong or time out immediately
        # But seed is 220. This should be much better.
        pass

    result = V[best_set]
    return result
