import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Generate candidate vectors with norm^2 = 2 and norm^2 = 4.
       - Norm^2 = 2: Two entries are +/-1, rest 0. (220 vectors)
       - Norm^2 = 4: One entry is +/-2, rest 0 (22 vectors) OR Four entries are +/-1, rest 0 (5280 vectors).
    2. The condition for a valid set S is min_{u,v in S} ||u-v|| >= max_{w in S} ||w||.
       If S contains any vector with norm^2=4, then max_norm^2 = 4, so we need ||u-v||^2 >= 4 for all pairs.
       If S contains only norm^2=2 vectors, we need ||u-v||^2 >= 2.
       
       The pool of norm^2=4 vectors is much larger (~5300 vs 220). We assume the maximum independent set
       under the stricter constraint ||u-v||^2 >= 4 will yield a larger k than the looser constraint on the smaller pool.
       
    3. Build a conflict graph where an edge exists between u and v if ||u-v||^2 < 4.
    4. Find a large independent set using a greedy heuristic with random restarts.
    """
    D = 11
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 120s limit
    
    candidates = []
    
    # Type 1: Two +/-1's (norm^2 = 2)
    # Count: C(11,2) * 4 = 55 * 4 = 220
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 2: One +/-2 (norm^2 = 4)
    # Count: 11 * 2 = 22
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 3: Four +/-1's (norm^2 = 4)
    # Count: C(11,4) * 16 = 330 * 16 = 5280
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
    
    # Compute pairwise squared distances
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2<u, v>
    # We need dist_sq >= 4 for validity (assuming max norm^2 is 4)
    
    # To save memory/time, we can compute the dot product matrix
    # N is ~5522. 5522x5522 float64 is ~240MB. This is acceptable.
    dots = candidates @ candidates.T
    
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict if dist_sq < 4
    # Note: dist_sq is an integer. So conflict if dist_sq <= 3.
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Greedy Maximum Independent Set with random restarts
    best_selected = []
    best_size = 0
    
    np.random.seed(42)
    
    # Number of restarts depends on time budget
    # Each restart is O(N^2) roughly, but with vectorized ops it's fast.
    # Let's run a few hundred restarts if time permits.
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        remaining = np.ones(n, dtype=bool)
        # degrees[i] is the number of conflicts for node i among remaining nodes
        # Initially, degree is the total number of conflicts
        degrees = conflict.sum(axis=1).astype(np.int32)
        
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Find nodes with minimum degree in the remaining subgraph
            # We update degrees dynamically
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            
            # Get all nodes with this min degree
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            # Randomly pick one of them to break symmetry
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen node and its neighbors from remaining
            neigh_mask = conflict[chosen_global]
            
            to_remove = np.zeros(n, dtype=bool)
            to_remove[chosen_global] = True
            to_remove[neigh_mask] = True
            
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                # Update degrees for the remaining nodes
                # For each node r in remaining, its degree decreases by the number of its neighbors in actual_remove
                # deg_new[r] = deg_old[r] - sum(conflict[r, actual_remove])
                
                # Vectorized update:
                # conflict[remaining][:, actual_remove] is a boolean matrix
                # Sum along axis 1 gives the number of removed neighbors for each remaining node
                deg_decrease = conflict[remaining][:, actual_remove].sum(axis=1)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
            
            # Time check inside the loop for very large graphs
            if time.time() > deadline:
                break
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            
    result = candidates[best_selected]
    return result

