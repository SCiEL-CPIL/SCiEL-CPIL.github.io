"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START

def generate_vectors(weight):
    vectors = []
    for support in itertools.combinations(range(D), weight):
        for signs in itertools.product([1, -1], repeat=weight):
            v = np.zeros(D, dtype=np.int8)
            for idx, val in zip(support, signs):
                v[idx] = val
            vectors.append(v)
    return np.array(vectors)

def find_large_code_greedy(V, limit, deadline, num_restarts=2000):
    N = V.shape[0]
    if N == 0:
        return np.array([], dtype=int).reshape(0, D)
    
    # Compute dot products
    dots = V @ V.T
    
    best_indices = None
    best_count = 0
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random permutation of indices
        perm = np.random.permutation(N)
        
        active = np.ones(N, dtype=bool)
        selected = []
        
        for idx in perm:
            if active[idx]:
                selected.append(idx)
                active &= (dots[idx] <= limit)
                active[idx] = False
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_indices = np.array(selected, dtype=int)
            
    if best_indices is None:
        return np.array([], dtype=int).reshape(0, D)
        
    return best_indices

def local_search_improve(V, indices, limit, deadline, max_iter=50):
    """
    Local search: try to add vectors not in indices, or swap to improve.
    Here we focus on adding: for each candidate not in selected, check if it fits.
    Also try 1-for-1 swaps: remove one, add two? That's complex. 
    Let's stick to adding first, then maybe 1-for-1 swaps of a selected vector 
    with two unselected ones? No, that's 2-for-1.
    Simplest: try to add any unselected vector that is compatible with ALL selected.
    If we can add, we do it. Repeat until no more can be added.
    """
    N = V.shape[0]
    dots = V @ V.T
    selected_set = set(indices.tolist())
    unselected = np.array([i for i in range(N) if i not in selected_set])
    
    if len(unselected) == 0:
        return indices
    
    current_indices = list(indices)
    current_set = set(current_indices)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Check each unselected vector
        # To speed up, we can vectorize the check against current selected
        # But current selected changes.
        
        # Efficient check:
        # For each unselected u, check max(dots[u, s]) for s in current_set <= limit
        # If max <= limit, u is compatible.
        
        # We can compute compatibility for all unselected at once?
        # dots[unselected][:, current_set] -> shape (M, K)
        # max over axis 1 <= limit
        
        if len(current_indices) == 0:
            # Can add any
            # Just pick one? Or all?
            # If limit is large, maybe all.
            # Let's just pick the first one for now, or random.
            # Actually, if current is empty, we can add all compatible with each other?
            # No, local search assumes we start from a valid set.
            # If start is empty, greedy would have picked something.
            pass
            
        # Vectorized check
        # Note: current_indices might be large.
        # dots[unselected][:, current_indices]
        # If M*K is large, this is expensive.
        # M ~ 5000, K ~ 1000. 5M ops. OK.
        
        compat_matrix = dots[unselected][:, current_indices]
        max_compat = np.max(compat_matrix, axis=1)
        
        # Find unselected vectors that are compatible with ALL current selected
        good_mask = max_compat <= limit
        
        if np.any(good_mask):
            # Pick one to add. 
            # Which one? 
            # Maybe the one that leaves the most options for future?
            # Greedy: pick the one with the smallest number of neighbors in unselected?
            # Or just random.
            # Let's pick the first one for determinism in local search step, 
            # or random to explore.
            # Random is better for escaping local optima.
            candidates = unselected[good_mask]
            pick_idx = np.random.choice(candidates)
            
            current_indices.append(pick_idx)
            current_set.add(pick_idx)
            
            # Remove from unselected
            unselected = np.array([i for i in unselected if i != pick_idx])
            improved = True
            
            # Check time
            if time.time() > deadline - 2:
                break
                
    return np.array(current_indices)

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds
    deadline = start_time + 140.0
    
    best_vectors = None
    best_count = 0
    
    # Try weights 4, 3, 2
    # W=4 is the most promising for large k
    
    for weight in [4, 3, 2]:
        if time.time() > deadline - 10:
            break
            
        # Generate vectors
        V = generate_vectors(weight)
        N = V.shape[0]
        
        if N == 0:
            continue
            
        # Calculate limit
        # Condition: u.v <= w/2
        limit = weight // 2
        
        # Run search
        # Use remaining time for greedy restarts + local search
        # Allocate 50% for greedy, 50% for local search?
        # Or do greedy first, then local search on the best found.
        
        # Let's do greedy with a time limit, then local search.
        greedy_deadline = start_time + (deadline - start_time) * 0.6
        ls_deadline = deadline - 5 # Leave 5s for overhead
        
        # Adjust restarts based on time available for greedy
        # Rough estimate: 1 restart ~ 1ms for N=5280?
        # Let's just set a high number and let the deadline handle it.
        
        num_restarts = 5000 if weight == 4 else 10000
        
        indices = find_large_code_greedy(V, limit, greedy_deadline, num_restarts=num_restarts)
        
        if indices is not None and len(indices) > 0:
            # Apply local search
            improved_indices = local_search_improve(V, indices, limit, ls_deadline)
            
            if len(improved_indices) > best_count:
                best_count = len(improved_indices)
                best_vectors = V[improved_indices]
            
            # Also check if the original greedy result was better (shouldn't be, but safe)
            if len(indices) > best_count:
                best_count = len(indices)
                best_vectors = V[indices]

    # Fallback
    if best_vectors is None or len(best_vectors) == 0:
        V2 = generate_vectors(2)
        if len(V2) > 0:
            best_vectors = V2
            best_count = len(V2)

    return best_vectors

# EVOLVE-BLOCK-END
