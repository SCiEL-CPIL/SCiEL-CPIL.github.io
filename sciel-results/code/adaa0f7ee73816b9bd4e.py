import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate a diverse pool of integer vectors.
    # Norms are sqrt(weight). We want a mix to allow for denser packing.
    # The constraint is: min_dist >= max_norm.
    # d(u,v)^2 = |u|^2 + |v|^2 - 2 u.v
    # We need |u|^2 + |v|^2 - 2 u.v >= max_norm^2
    # Let M = max_norm^2.
    # For u, v in set: |u|^2 + |v|^2 - 2 u.v >= M
    
    # Strategy:
    # 1. Generate candidates with weights 2 to 10 (norms ~1.4 to ~3.16).
    # 2. Use Simulated Annealing to maximize the number of vectors in the set.
    # 3. The "energy" is negative count (we want to minimize -count).
    # 4. Moves: Add a random vector, Remove a random vector, Swap a vector.
    # 5. Constraint check: The set must be valid.
    
    # To make this efficient, we precompute dot products between all pairs in the pool.
    # Pool size estimation:
    # Weight 2: C(11,2)*4 = 55*4 = 220
    # Weight 3: C(11,3)*8 = 165*8 = 1320
    # Weight 4: C(11,4)*16 = 330*16 = 5280
    # Weight 5: C(11,5)*32 = 462*32 = 14784
    # Weight 6: C(11,6)*64 = 462*64 = 29568
    # This is getting large. Dot product matrix for 50k vectors is 2.5e9 entries. Too big for memory (20GB).
    
    # Alternative: Smaller pool, or on-the-fly dot products.
    # Let's stick to a moderate pool size, e.g., weights 2, 3, 4, 5.
    # Total ~ 15000 vectors. 15000x15000 float32 is ~900MB. Still large but maybe okay?
    # Better: Use a smaller pool or check validity incrementally.
    
    # Let's try a pool of weights 2, 3, 4. Total ~ 6800 vectors.
    # 6800x6800 float32 ~ 180MB. This is manageable.
    
    candidates = []
    # Weight 2
    for comb in itertools.combinations(range(D), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0]*D
                v[comb[0]] = s1
                v[comb[1]] = s2
                candidates.append(v)
    # Weight 3
    for comb in itertools.combinations(range(D), 3):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    v = [0]*D
                    v[comb[0]] = s1
                    v[comb[1]] = s2
                    v[comb[2]] = s3
                    candidates.append(v)
    # Weight 4
    for comb in itertools.combinations(range(D), 4):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = [0]*D
                        v[comb[0]] = s1
                        v[comb[1]] = s2
                        v[comb[2]] = s3
                        v[comb[3]] = s4
                        candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # Precompute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Precompute dot product matrix? 
    # n_cands ~ 220 + 1320 + 5280 = 6820.
    # 6820 * 6820 * 4 bytes = ~185 MB.
    # This is acceptable.
    dots = candidates @ candidates.T
    
    # Initialize set with a few random valid vectors or just start empty?
    # Starting empty and adding is hard for SA. 
    # Better: Start with a greedy construction (like the parent) to get a good initial state.
    
    # Greedy Init
    current_set = []
    current_set_indices = set()
    
    # Sort candidates by norm? Or just random order?
    # Let's do a fast greedy pass first.
    # Random order to avoid bias
    order = np.random.permutation(n_cands)
    
    for idx in order:
        if time.time() > deadline - 30: break # Leave time for SA
        v = candidates[idx]
        n_sq = norms_sq[idx]
        valid = True
        for j in current_set:
            # Check constraint: dist^2 >= max_norm^2
            # max_norm^2 is the max norm in the current set including v
            # This is tricky because adding v changes max_norm.
            # Let M_old = max(norms_sq[j] for j in current_set)
            # M_new = max(M_old, n_sq)
            # Constraint for pair (j, idx): norms_sq[j] + n_sq - 2*dots[j,idx] >= M_new
            # We don't know M_new until we check all.
            # However, if n_sq > M_old, M_new = n_sq.
            # If n_sq <= M_old, M_new = M_old.
            
            # Let's just compute M_new candidate for each j? No, M_new is global.
            # Let M_candidate = max(M_old, n_sq)
            # Check if for all j in current_set: norms_sq[j] + n_sq - 2*dots[j,idx] >= M_candidate
            pass # This logic is complex inside the loop. 
            # Simpler: Just check against the current max norm M_old first?
            # If v is smaller, M stays M_old.
            # If v is larger, M becomes n_sq.
            
            # Let's just use a simple check: 
            # For each j, dist_sq = norms_sq[j] + n_sq - 2*dots[j,idx]
            # We need dist_sq >= max(norms_sq[j], n_sq, M_old) ? No, max_norm of the SET.
            # Let M_set = max(norms_sq[j] for j in current_set)
            # New M = max(M_set, n_sq)
            # Condition: dist_sq >= New M
            
        # This greedy is getting complicated to verify quickly.
        # Let's restart SA from a simple random valid set.
        
    # Reset current_set to empty and use a simpler greedy:
    # Add vectors one by one. Maintain M_max.
    current_set = []
    M_max = 0
    for idx in order:
        if time.time() > deadline - 30: break
        n_sq = norms_sq[idx]
        new_M = max(M_max, n_sq)
        valid = True
        for j in current_set:
            dist_sq = norms_sq[j] + n_sq - 2 * dots[j, idx]
            if dist_sq < new_M:
                valid = False
                break
        if valid:
            current_set.append(idx)
            M_max = new_M
            
    best_set = current_set[:]
    best_score = len(best_set)
    
    # Simulated Annealing
    # State: list of indices
    # Move: Replace one index with another, or Add/Remove (if we allow variable size)
    # Let's allow variable size. 
    # Objective: Maximize len(set)
    # Validity: Strict. If move invalid, reject.
    
    # To speed up validity check, we can cache M_max and norms.
    # When adding/removing, we only need to check the new vector against existing ones.
    # But M_max might change if we remove the vector that had the max norm.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    current_set = best_set[:]
    # Sort current_set? No need.
    
    # Precompute norms for current set?
    # We have global norms_sq and dots.
    
    # Helper to check validity of a set (for initialization or rare cases)
    def is_valid(set_indices):
        if not set_indices:
            return True
        M = max(norms_sq[i] for i in set_indices)
        for i in set_indices:
            for j in set_indices:
                if i >= j: continue
                dist_sq = norms_sq[i] + norms_sq[j] - 2 * dots[i, j]
                if dist_sq < M:
                    return False
        return True
        
    # For SA, we do incremental checks.
    # State: current_set (list of indices)
    # M_max = max(norms_sq[i] for i in current_set)
    
    # Re-calculate M_max for current_set
    M_max = max(norms_sq[i] for i in current_set) if current_set else 0
    
    # To make SA efficient, we need to quickly check if adding/removing is valid.
    # Add idx:
    #   new_M = max(M_max, norms_sq[idx])
    #   Check all j in current_set: dist_sq(j, idx) >= new_M
    # Remove idx:
    #   If norms_sq[idx] < M_max, M_max doesn't change. Validity of remaining set is preserved (subset of valid set).
    #   If norms_sq[idx] == M_max, M_max might decrease. The remaining set is a subset of the old valid set, 
    #   BUT the constraint uses the NEW max norm. 
    #   dist_sq(i,j) >= OLD_M implies dist_sq(i,j) >= NEW_M (since NEW_M <= OLD_M).
    #   So removing a vector ALWAYS results in a valid set!
    #   This is a huge simplification. 
    #   So "Remove" moves are always valid.
    #   "Add" moves require checking.
    #   "Swap" (Remove i, Add j): 
    #      1. Remove i (always valid).
    #      2. Add j to the resulting set. Check j against all k in set\{i}.
    
    while time.time() < deadline - 5 and T > T_min:
        # Generate move
        move_type = random.random()
        if move_type < 0.5:
            # Add
            idx = random.randint(0, n_cands - 1)
            if idx in current_set:
                continue
            new_M = max(M_max, norms_sq[idx])
            valid = True
            for j in current_set:
                dist_sq = norms_sq[j] + norms_sq[idx] - 2 * dots[j, idx]
                if dist_sq < new_M:
                    valid = False
                    break
            if valid:
                current_set.append(idx)
                M_max = new_M
                # Accept
                if len(current_set) > best_score:
                    best_score = len(current_set)
                    best_set = current_set[:]
            # If invalid, reject
        elif move_type < 0.8:
            # Remove
            if not current_set:
                continue
            idx = random.choice(current_set)
            current_set.remove(idx)
            # Update M_max if needed
            if norms_sq[idx] == M_max:
                M_max = max(norms_sq[i] for i in current_set) if current_set else 0
            # Always valid, so accept. 
            # But removing reduces score. SA usually accepts worse moves with prob exp(-dE/T).
            # Here "score" is count. dE = -1.
            # We should probably not always accept removals unless we are stuck?
            # Actually, removing allows us to add different vectors later.
            # Let's accept removals with high probability initially, lower later?
            # Or just accept if random < exp(1/T)? No, dE is negative (score drops).
            # Let's accept removals with prob 0.5 to explore.
            if random.random() > 0.5:
                # Revert? No, just leave it removed. It's a valid state.
                pass
        else:
            # Swap
            if not current_set:
                continue
            idx_out = random.choice(current_set)
            idx_in = random.randint(0, n_cands - 1)
            if idx_in == idx_out or idx_in in current_set:
                continue
            
            # Remove idx_out
            current_set.remove(idx_out)
            temp_M = max(norms_sq[i] for i in current_set) if current_set else 0
            
            # Try to add idx_in
            new_M = max(temp_M, norms_sq[idx_in])
            valid = True
            for j in current_set:
                dist_sq = norms_sq[j] + norms_sq[idx_in] - 2 * dots[j, idx_in]
                if dist_sq < new_M:
                    valid = False
                    break
            
            if valid:
                current_set.append(idx_in)
                M_max = new_M
                # Accept
                if len(current_set) > best_score:
                    best_score = len(current_set)
                    best_set = current_set[:]
            else:
                # Revert removal
                current_set.append(idx_out)
                # Restore M_max
                M_max = max(temp_M, norms_sq[idx_out])
                
        T *= cooling
        
    # Final check and return
    result_vectors = candidates[best_set]
    return result_vectors
# EVOLVE-BLOCK-END
