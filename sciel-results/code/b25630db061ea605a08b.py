# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import scipy.sparse

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Uses Multi-Start Greedy with Sparse Conflict Matrix.
    Includes norm^2=1, 2, 3, 4 candidates.
    """
    D = 11
    deadline = time.time() + 110.0 # Safety margin
    
    # Generate candidate vectors
    candidates = []
    
    # Type 0: one ±1 (norm^2 = 1)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type 1: two ±1's (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 2: three ±1's (norm^2 = 3)
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Type 3: one ±2 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 4: two ±2's (norm^2 = 8) - WAIT. 
    # If we include norm^2=8, max norm becomes sqrt(8) ~ 2.82.
    # Dist must be >= 2.82, i.e. dist^2 >= 8.
    # This makes constraints MUCH tighter.
    # Current max norm^2 in code was 4.
    # Let's stick to max norm^2 = 4 to allow denser packing.
    # So NO norm^2=8 vectors.
    
    # Type 5: four ±1's (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = np.max(norms_sq) # Should be 4
    
    # Compute pairwise dot products
    dots = candidates @ candidates.T
    
    # Compute squared distances
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict matrix: conflict[i,j] = True if dist_sq[i,j] < max_norm_sq
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # Convert to sparse for faster neighbor retrieval
    conflict_sparse = scipy.sparse.csr_matrix(conflict)
    
    best_size = 0
    best_selection = []
    
    np.random.seed(42)
    
    # Multi-Start Greedy
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        active = np.ones(n, dtype=bool)
        # Initial degrees
        degrees = conflict_sparse.sum(axis=1).A1.astype(np.int32)
        
        selected = []
        
        # To speed up finding min degree, we can use a heap?
        # But degrees change.
        # Simple approach: find min among active.
        # Optimization: If we just want a good independent set, 
        # randomizing the order of processing low-degree nodes is good.
        
        # Create a list of active indices
        active_indices = np.arange(n)
        
        # We will maintain a list of "alive" nodes.
        # Instead of scanning all N every time, we can use a set or list.
        # But N is small (~7000). Scanning is fast.
        
        for _ in range(n):
            if not np.any(active):
                break
                
            # Find active indices
            # This is O(N)
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            # Get degrees of active nodes
            active_degrees = degrees[active_indices]
            
            # Find min degree
            min_deg = np.min(active_degrees)
            
            # Find all active nodes with min degree
            min_mask = (active_degrees == min_deg)
            min_candidates = active_indices[min_mask]
            
            # Randomly pick one
            chosen = np.random.choice(min_candidates)
            
            selected.append(chosen)
            
            # Update active and degrees
            # Get neighbors of chosen
            neighbors = np.array(conflict_sparse[chosen].indices)
            
            # Mark chosen and neighbors as inactive
            to_remove = np.concatenate(([chosen], neighbors))
            
            # Update degrees for remaining active nodes
            # We subtract 1 from degree of all neighbors of removed nodes.
            # Vectorized: 
            # For each u in to_remove, degrees[neighbors(u)] -= 1
            # This is a loop over to_remove.
            
            # Optimization: 
            # We can accumulate the degree changes?
            # No, just loop. It's fast enough for 1000 restarts if N is small.
            
            for u in to_remove:
                if not active[u]:
                    continue
                active[u] = False
                neigh_u = np.array(conflict_sparse[u].indices)
                degrees[neigh_u] -= 1
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_selection = selected.copy()
            
    result = candidates[best_selection]
    return result
# EVOLVE-BLOCK-END
