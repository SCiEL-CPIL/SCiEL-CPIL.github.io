"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Generate Candidate Pool
    # We include weights 2, 3, and 4.
    # Weight 2: norm^2 = 2. Count = C(11,2)*4 = 220
    # Weight 3: norm^2 = 3. Count = C(11,3)*8 = 165*8 = 1320
    # Weight 4: norm^2 = 4. Count = C(11,4)*16 = 330*16 = 5280
    # Total candidates ~ 6820. This is manageable for a dynamic conflict graph.
    
    candidates = []
    
    # Generate weight-4 vectors
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    # Generate weight-3 vectors
    for comb in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    # Generate weight-2 vectors
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Precompute Gram matrix
    # 6820 x 6820 matrix of int64. Memory ~ 370MB. Might be tight but usually OK in modern envs.
    # If memory is a concern, we could chunk it, but let's try full matrix first for speed.
    # Actually, 6820^2 * 8 bytes = ~370 MB. This is acceptable for most CI/CD runners with >1GB RAM.
    G = candidates @ candidates.T
    
    # Precompute conflict limits
    # Condition: G[i,j] <= min(norms_sq[i], norms_sq[j]) / 2.0
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    limits = min_norms_sq / 2.0
    np.fill_diagonal(limits, np.inf) # Self is never a conflict in our logic (we don't add self)
    
    # Conflict mask: True if incompatible
    conflict_mask = G > limits
    np.fill_diagonal(conflict_mask, False)
    
    # 2. Simulated Annealing
    # State: Set of selected indices.
    # We represent the current set as a boolean array `in_set`.
    # We also maintain a dynamic count of conflicts for each candidate to speed up moves.
    # conflict_count[i] = number of selected vectors that conflict with candidate i.
    
    def get_conflict_counts(in_set):
        # in_set: boolean array of size n_cand
        # Returns: array of size n_cand, where count[i] is # of j in in_set s.t. conflict_mask[i,j] is True
        # This is essentially a matrix-vector product: conflict_mask @ in_set
        # Since conflict_mask is boolean, we can use sum along axis 1.
        return np.sum(conflict_mask[in_set], axis=1)
    
    # Initial solution: Greedy random
    random.seed(42)
    indices = list(range(n_cand))
    random.shuffle(indices)
    
    in_set = np.zeros(n_cand, dtype=bool)
    current_conflicts = np.zeros(n_cand, dtype=np.int32)
    
    # Greedy initialization
    for idx in indices:
        if current_conflicts[idx] == 0:
            in_set[idx] = True
            # Update conflicts for all neighbors of idx
            # neighbors are those j where conflict_mask[idx, j] is True
            # We can use the precomputed conflict_mask row
            current_conflicts[conflict_mask[idx]] += 1
            # Note: self is False in conflict_mask, so no self-update needed
            
    best_in_set = in_set.copy()
    best_score = np.sum(best_in_set)
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    # Number of steps: We have ~140s. 
    # Each step involves:
    # 1. Pick a random move (add or remove)
    # 2. Check delta (fast with precomputed counts)
    # 3. Update state (vectorized update of conflict counts for neighbors)
    # Vectorized update is O(degree). Average degree might be high.
    # Let's estimate speed. 6820 candidates. 
    # If we do 10,000 steps per second, that's 1.4M steps.
    # Let's aim for as many as possible.
    
    step_count = 0
    max_steps = 5000000 # Hard cap, but time is the real limit
    
    # We need to track "current_score" (size of in_set)
    current_score = best_score
    
    # Precompute neighbor lists for faster updates? 
    # conflict_mask is dense-ish? 
    # For weight 4 vs weight 4: dot product 2 is max compatible. Conflict if > 2. 
    # Most pairs are compatible? 
    # Let's check density.
    # If graph is sparse, neighbor lists are good. If dense, matrix mult is good.
    # Given the geometry, most vectors are "far apart" in dot product terms (close to 0), 
    # so most pairs are compatible. The conflict graph is likely sparse.
    # Let's use sparse representation for updates if possible, or just vectorized boolean ops.
    
    # To speed up updates:
    # When adding i:
    #   current_conflicts[neighbors(i)] += 1
    # When removing i:
    #   current_conflicts[neighbors(i)] -= 1
    # neighbors(i) are indices where conflict_mask[i] is True.
    
    # Precompute list of neighbors for each node? 
    # 6820 lists. Total edges? 
    # If average degree is 100, that's 682k edges. Very manageable.
    neighbor_indices = []
    for i in range(n_cand):
        neighbor_indices.append(np.nonzero(conflict_mask[i])[0])
        
    # SA Loop
    # We alternate between Add and Remove moves.
    # Probability of Add: p_add = 0.5 initially, maybe adjust based on score?
    # Simple approach: 50% Add, 50% Remove.
    
    # Optimization: 
    # Instead of picking a random index from 0..N-1 and checking if it's in_set/out_set,
    # we can maintain two lists: active_indices (in_set) and inactive_indices (out_set).
    # But updating lists is O(N) if we remove. 
    # Better: Just pick random index. 
    # If we pick i:
    #   if in_set[i]: try remove
    #   else: try add
    
    while time.time() < deadline and step_count < max_steps:
        # Anneal temperature
        # Linear or geometric? Geometric is better for SA.
        # But we don't know total steps in advance. 
        # Let's use a time-based cooling or a large fixed number of steps with a schedule.
        # Let's assume we run for ~130s. 
        # Let's just use a simple schedule based on step_count relative to a guessed max.
        # Or better: Restart with new T if stuck?
        
        # Let's use a fixed large number of steps and a geometric decay.
        # T = T_start * (T_end/T_start)^(step_count / max_steps)
        
        # Pick random index
        i = random.randint(0, n_cand - 1)
        
        if in_set[i]:
            # Try Remove
            # Delta score = -1
            # Always accept removal if it doesn't violate anything? 
            # Removal always improves "validity" (fewer conflicts) but decreases score.
            # In SA for Max Independent Set, we accept removal with probability exp((delta)/T) where delta = -1.
            # This allows us to escape local maxima by shrinking the set.
            delta = -1
            if random.random() < np.exp(delta / T_start): # Using T_start as a proxy for current T in this simple loop
                # Accept removal
                in_set[i] = False
                current_score += delta
                # Update conflicts: decrement neighbors
                current_conflicts[neighbor_indices[i]] -= 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_in_set = in_set.copy()
        else:
            # Try Add
            # Only possible if current_conflicts[i] == 0
            if current_conflicts[i] == 0:
                delta = 1
                # Always accept addition
                in_set[i] = True
                current_score += delta
                # Update conflicts: increment neighbors
                current_conflicts[neighbor_indices[i]] += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_in_set = in_set.copy()
            else:
                # Cannot add. Do nothing.
                pass
                
        step_count += 1
        
        # Periodic check to avoid infinite loops in tight branches
        if step_count % 10000 == 0:
             # Optional: Restart if no improvement for a long time?
             # For now, just continue.
             pass

    # Final extraction
    final_indices = np.nonzero(best_in_set)[0]
    final_vecs = candidates[final_indices]
    
    return final_vecs

# EVOLVE-BLOCK-END
