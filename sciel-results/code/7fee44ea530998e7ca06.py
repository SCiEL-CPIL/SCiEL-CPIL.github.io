import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Generate Vectors
    # Weight 2 vectors
    w2_vectors = []
    w2_indices = []
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = s1
                    v[j] = s2
                    w2_vectors.append(v)
                    w2_indices.append((i, j, s1, s2))
    N_W2 = len(w2_vectors)
    w2_arr = np.array(w2_vectors, dtype=np.int8)
    
    # Weight 3 vectors
    w3_vectors = []
    w3_indices = []
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(D, dtype=np.int8)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            w3_vectors.append(v)
                            w3_indices.append((i, j, k, s1, s2, s3))
    N_W3 = len(w3_vectors)
    w3_arr = np.array(w3_vectors, dtype=np.int8)
    
    # 2. Precompute Conflicts
    # Conflict W3 vs W2: True if dot product > 1 (i.e., == 2)
    # dot = w3_arr @ w2_arr.T
    # We need dot > 1.
    
    if time.time() > deadline - 10:
        # Fallback to W2 only
        return w2_arr.astype(np.int64)
        
    dots_w3_w2 = w3_arr @ w2_arr.T # (N_W3, N_W2)
    conflict_w3_w2 = dots_w3_w2 > 1 # (N_W3, N_W2)
    
    # Conflict W3 vs W3: True if dot product > 1
    # This matrix is 1320x1320 ~ 1.7M entries. Manageable.
    dots_w3_w3 = w3_arr @ w3_arr.T # (N_W3, N_W3)
    conflict_w3_w3 = dots_w3_w3 > 1
    
    # 3. State
    active_w2 = np.ones(N_W2, dtype=bool)
    active_w3 = np.zeros(N_W3, dtype=bool)
    
    best_count = N_W2
    best_state_w2 = active_w2.copy()
    best_state_w3 = active_w3.copy()
    
    # 4. Local Search
    # We look for "profitable" moves.
    # Move Type A: Add W3 that conflicts with 0 active W2s. (Gain +1)
    # Move Type B: Remove 1 W2, Add 2 W3s that both conflicted ONLY with that W2 (and are mutually compatible). (Gain +1)
    
    improved = True
    while improved:
        improved = False
        
        if time.time() > deadline - 5:
            break
            
        # Check Type A moves first (easier)
        # For each inactive W3, check if it conflicts with any active W2.
        # conflicts_with_active_w2 = conflict_w3_w2[:, active_w2].any(axis=1)
        # If any inactive W3 has no conflicts, add it.
        
        inactive_w3_indices = np.where(~active_w3)[0]
        
        if len(inactive_w3_indices) > 0:
            # Vectorized check for Type A
            # We need to know if inactive_w3[i] conflicts with ANY active_w2.
            # sub_conflict = conflict_w3_w2[inactive_w3_indices, :][:, active_w2]
            # This creates a large temporary array. 
            # Better: precompute active_w2 mask.
            
            # Let's just loop over inactive W3s if count is small, or use vectorization carefully.
            # N_W3 is 1320. N_W2 is 220.
            # conflict_w3_w2 is 1320x220.
            # active_w2 is 220.
            # np.any(conflict_w3_w2[:, active_w2], axis=1) gives (1320,) bool array.
            
            conflicts_with_active = np.any(conflict_w3_w2[:, active_w2], axis=1)
            
            # Find inactive W3s with no conflicts
            addable = inactive_w3_indices[~conflicts_with_active[inactive_w3_indices]]
            
            if len(addable) > 0:
                # Add all of them? 
                # They must be compatible with each other too.
                # Let's just add the first one to be safe, or check compatibility.
                # Given the structure, likely few or zero.
                for idx in addable:
                    # Check compatibility with existing active W3s
                    if not np.any(conflict_w3_w3[idx, active_w3]):
                        active_w3[idx] = True
                        improved = True
                        best_count = np.sum(active_w2) + np.sum(active_w3)
                        best_state_w2 = active_w2.copy()
                        best_state_w3 = active_w3.copy()
                        # Re-evaluate conflicts_with_active? 
                        # Adding W3 doesn't change W2 conflicts.
                        # But it might block other W3s.
                        # Break to restart loop? Yes, safer.
                        break
                if improved:
                    continue
                    
        # Check Type B moves
        # Find pairs of inactive W3s (u, v) such that:
        # 1. u, v compatible (conflict_w3_w3[u, v] == False)
        # 2. Conflicts of u with active W2 is a set {w2_a}
        # 3. Conflicts of v with active W2 is a set {w2_a}
        # 4. w2_a is the ONLY conflict for both.
        
        # This is expensive to check all pairs.
        # Optimization: Group inactive W3s by their conflict set with active W2.
        
        # For each inactive W3, compute its conflict set with active W2.
        # Since active_w2 changes, we recompute.
        
        inactive_mask = ~active_w3
        inactive_idx = np.where(inactive_mask)[0]
        
        if len(inactive_idx) == 0:
            break
            
        # Get conflict rows for inactive W3s
        # conflict_sub = conflict_w3_w2[inactive_idx, :][:, active_w2]
        # This is (num_inactive, num_active_w2)
        
        # We want to find W3s that conflict with exactly 1 active W2.
        # And group them by which W2 they conflict with.
        
        conflict_sub = conflict_w3_w2[inactive_idx, :][:, active_w2]
        
        # Count conflicts for each inactive W3
        conflict_counts = np.sum(conflict_sub, axis=1)
        
        # Candidates are those with count == 1
        single_conflict_mask = conflict_counts == 1
        single_conflict_indices = inactive_idx[single_conflict_mask]
        
        if len(single_conflict_indices) > 0:
            # Find which W2 each of these conflicts with
            # conflict_sub is (num_single, num_active_w2)
            # The True position gives the index in active_w2.
            
            # Map active_w2 indices to original W2 indices
            active_w2_indices = np.where(active_w2)[0]
            
            # For each single conflict W3, find the active W2 index
            # conflict_sub_row has one True.
            # np.argmax(conflict_sub_row) gives position in active_w2 array.
            
            # This loop is over single_conflict_indices.
            # We can vectorize?
            # conflict_sub_single = conflict_sub[single_conflict_mask, :]
            # positions = np.argmax(conflict_sub_single, axis=1)
            # original_w2_idx = active_w2_indices[positions]
            
            conflict_sub_single = conflict_sub[single_conflict_mask, :]
            positions = np.argmax(conflict_sub_single, axis=1)
            original_w2_idx = active_w2_indices[positions]
            
            # Group single_conflict_indices by original_w2_idx
            # We want to find pairs (u, v) in the same group that are compatible.
            
            # Use a dictionary: w2_idx -> list of w3_idx
            groups = {}
            for w3_idx, w2_idx in zip(single_conflict_indices, original_w2_idx):
                if w2_idx not in groups:
                    groups[w2_idx] = []
                groups[w2_idx].append(w3_idx)
                
            # For each group, check if any pair is compatible
            for w2_idx, w3_list in groups.items():
                if len(w3_list) < 2:
                    continue
                    
                # Check compatibility within w3_list
                # w3_list are indices into w3_arr
                # We need conflict_w3_w3[u, v] == False
                
                # Get sub-matrix
                # w3_sub = w3_arr[w3_list]
                # dots = w3_sub @ w3_sub.T
                # This is small (size of group).
                
                # If group size is large, this is costly.
                # But groups are likely small.
                
                for i in range(len(w3_list)):
                    u = w3_list[i]
                    for j in range(i + 1, len(w3_list)):
                        v = w3_list[j]
                        
                        if not conflict_w3_w3[u, v]:
                            # Also check if u and v are compatible with all OTHER active W3s
                            # If they are, we can make the move.
                            
                            # Check u against active_w3
                            if np.any(conflict_w3_w3[u, active_w3]):
                                continue
                            if np.any(conflict_w3_w3[v, active_w3]):
                                continue
                                
                            # Move: Remove w2_idx, Add u, v
                            active_w2[w2_idx] = False
                            active_w3[u] = True
                            active_w3[v] = True
                            
                            improved = True
                            best_count = np.sum(active_w2) + np.sum(active_w3)
                            best_state_w2 = active_w2.copy()
                            best_state_w3 = active_w3.copy()
                            
                            # Break out of loops to restart
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        if time.time() > deadline - 5:
            break

    # Construct result
    result_vectors = []
    if np.any(best_state_w2):
        result_vectors.append(w2_arr[best_state_w2])
    if np.any(best_state_w3):
        result_vectors.append(w3_arr[best_state_w3])
        
    if not result_vectors:
        # Fallback
        return w2_arr.astype(np.int64)
        
    final_arr = np.vstack(result_vectors)
    return final_arr.astype(np.int64)

# EVOLVE-BLOCK-END
