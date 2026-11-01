"""Rewritten program for kissing number in dimension 11 using Simulated Annealing."""
import numpy as np
import itertools
import time
import sys

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate Candidate Vectors
    # We focus on vectors with small squared norms to allow high density.
    # Norm squared 4: weight 1 (coeff 2), weight 4 (coeff 1), weight 2 (coeff 1,1 -> norm 2? No, 1^2+1^2=2).
    # Let's stick to norm squared 4 for uniformity, which simplifies the constraint to dot <= 2.
    # Candidates:
    # - Weight 1, coeff +/- 2: 2 * 11 = 22 vectors. Norm^2 = 4.
    # - Weight 4, coeff +/- 1: C(11,4) * 2^4 = 330 * 16 = 5280 vectors. Norm^2 = 4.
    # - Weight 2, coeff +/- 1, +/- 1? No, 1+1=2 != 4.
    # - Weight 2, coeff +/- sqrt(2)? Not integer.
    # - Weight 1, coeff +/- 1? Norm^2 = 1. If we mix norms, max norm dominates.
    #   If we have a vector of norm 2 and a vector of norm 1, distance >= 2.
    #   ||u-v||^2 >= 4. If ||u||=2, ||v||=1, 4+1 - 2u.v >= 4 => u.v <= 0.5 => u.v <= 0 (integer).
    #   This allows more vectors! We can mix norms.
    
    # Let's generate a pool of vectors with norm squared in {1, 2, 4}.
    # Constraint: For all i, j in S, ||v_i - v_j|| >= max_{k in S} ||v_k||.
    # This is harder to check dynamically. 
    # To keep it simple and fast, let's restrict to **equal norm** vectors first, as the current code does.
    # If we use norm squared 4, we need dot <= 2.
    
    candidates = []
    
    # Weight 1, coeff 2
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
            
    # Weight 4, coeff 1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    # Weight 2, coeff 1? Norm^2 = 2.
    # If we include these, max norm might be 2 (from weight 1 or 4).
    # Let's try to include Weight 2 vectors (norm sqrt(2)) and see if we can pack more.
    # If max norm is 2, we need ||u-v|| >= 2.
    # If u has norm 2, v has norm sqrt(2):
    # ||u-v||^2 = 4 + 2 - 2 u.v = 6 - 2 u.v >= 4 => 2 u.v <= 2 => u.v <= 1.
    # If u, v both have norm sqrt(2):
    # ||u-v||^2 = 2 + 2 - 2 u.v = 4 - 2 u.v >= 4 => u.v <= 0.
    
    # This mixed-norm approach is more complex. Let's stick to the uniform norm squared 4 approach 
    # but use a better solver: Simulated Annealing.
    
    V = np.array(candidates, dtype=np.int64)
    N = len(V)
    
    # Compute Dot Products
    dots = V @ V.T
    
    # Conflict Matrix: dot > 2
    # Note: For i=j, dot=4 > 2, but we don't care about self-conflict in the set logic, 
    # we just need to ensure we don't add the same vector twice.
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists for speed
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    
    # Initialize Set
    # Start with a greedy random independent set
    rng = np.random.default_rng(42)
    in_set = np.zeros(N, dtype=bool)
    
    # Greedy Initialization
    alive = np.ones(N, dtype=bool)
    # Sort indices by degree (descending) to pick "safe" nodes first? 
    # Or just random.
    order = rng.permutation(N)
    for i in order:
        if not alive[i]:
            continue
        # Check if i conflicts with any in_set
        if not np.any(conflicts[i] & in_set):
            in_set[i] = True
            # Mark neighbors as not alive for greedy? No, just check against in_set.
            # This is O(N^2) total. N=5302. 5302^2 ~ 28M. Fast enough.
            
    current_size = in_set.sum()
    best_size = current_size
    best_set = in_set.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.05
    
    # Precompute conflict counts for fast evaluation?
    # conflict_count[i] = number of neighbors of i currently in the set.
    conflict_count = np.zeros(N, dtype=np.int32)
    for i in np.where(in_set)[0]:
        conflict_count[neighbors[i]] += 1
        
    sa_deadline = deadline - 5.0
    
    while time.time() < sa_deadline:
        # Pick a random move
        # Move Type 1: Add a random node
        # Move Type 2: Remove a random node from set
        # Move Type 3: Swap a node in set with a node not in set
        
        move_type = rng.integers(3)
        
        if move_type == 0:
            # Add
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0: continue
            v = not_in[rng.integers(len(not_in))]
            
            # To add v, we must remove all its neighbors in the set.
            to_remove = neighbors[v]
            to_remove_in_set = to_remove[in_set[to_remove]]
            
            delta = 1 - len(to_remove_in_set)
            
            if delta >= 0 or rng.random() < np.exp(delta / T):
                # Apply
                in_set[to_remove_in_set] = False
                for u in to_remove_in_set:
                    conflict_count[neighbors[u]] -= 1
                in_set[v] = True
                conflict_count[neighbors[v]] += 1
                current_size += delta
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = in_set.copy()
                    
        elif move_type == 1:
            # Remove
            in_indices = np.where(in_set)[0]
            if len(in_indices) == 0: continue
            v = in_indices[rng.integers(len(in_indices))]
            
            delta = -1
            # Always accept removal if it helps? No, SA should allow bad moves.
            # But removing decreases size. We only accept if T is high enough.
            if rng.random() < np.exp(delta / T):
                in_set[v] = False
                conflict_count[neighbors[v]] -= 1
                current_size += delta
        else:
            # Swap
            in_indices = np.where(in_set)[0]
            not_in = np.where(~in_set)[0]
            if len(in_indices) == 0 or len(not_in) == 0: continue
            
            v_in = in_indices[rng.integers(len(in_indices))]
            v_out = not_in[rng.integers(len(not_in))]
            
            # Remove v_in, add v_out.
            # v_out must not conflict with (in_set \ {v_in}).
            # Conflict with v_in is irrelevant since v_in is removed.
            
            # Check if v_out conflicts with any other node in in_set
            # neighbors[v_out] & in_set. If v_in is a neighbor, it will be cleared.
            # We need: (neighbors[v_out] & in_set) == {v_in} or empty.
            
            conflicts_with_out = np.where(conflicts[v_out] & in_set)[0]
            
            # Valid swap if all conflicts are just v_in
            if len(conflicts_with_out) == 0 or (len(conflicts_with_out) == 1 and conflicts_with_out[0] == v_in):
                delta = 0 # Size stays same
                # We might accept this to escape local minima
                if rng.random() < np.exp(0 / T): # Always 1.0? 
                    # Let's add a slight bias or just accept to explore
                    # Actually, if delta=0, we should accept with prob 1 to explore.
                    in_set[v_in] = False
                    conflict_count[neighbors[v_in]] -= 1
                    in_set[v_out] = True
                    conflict_count[neighbors[v_out]] += 1
                    # current_size unchanged
            else:
                # If v_out conflicts with others, we would need to remove them too.
                # This is a "large move". Let's skip for now to keep it fast.
                pass
                
        T *= cooling
        if T < min_T:
            T = min_T
            
    # Extract final vectors
    final_indices = np.where(best_set)[0]
    result_vectors = V[final_indices]
    
    # Ensure non-empty
    if len(result_vectors) == 0:
         # Fallback to just the 22 weight-1 vectors
         return V[:22]
         
    return result_vectors

# EVOLVE-BLOCK-END
