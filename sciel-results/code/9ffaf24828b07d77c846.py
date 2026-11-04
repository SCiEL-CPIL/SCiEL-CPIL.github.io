import numpy as np
import itertools
import random
import time

D = 11
DEADLINE_S = 145.0  # Safety margin for 150s limit

def generate_candidates(max_norm_sq=5):
    vectors = []
    norms_sq = []
    
    # Norm squared 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                norms_sq.append(2)
                
    # Norm squared 3
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            norms_sq.append(3)
            
    # Norm squared 4
    # Type 1: four +/-1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            norms_sq.append(4)
    # Type 2: one +/-2
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            vectors.append(v)
            norms_sq.append(4)
            
    # Norm squared 5
    # Type 1: five +/-1
    if max_norm_sq >= 5:
        for combo in itertools.combinations(range(D), 5):
            for signs in itertools.product([1, -1], repeat=5):
                v = [0] * D
                for idx, s in zip(combo, signs):
                    v[idx] = s
                vectors.append(v)
                norms_sq.append(5)
    # Type 2: one +/-2, one +/-1
    if max_norm_sq >= 5:
        for i in range(D):
            for j in range(D):
                if i == j: continue
                for s2 in (2, -2):
                    for s1 in (1, -1):
                        v = [0] * D
                        v[i] = s2
                        v[j] = s1
                        vectors.append(v)
                        norms_sq.append(5)

    V = np.array(vectors, dtype=np.int64)
    norms_sq_arr = np.array(norms_sq, dtype=np.int64)
    return V, norms_sq_arr

def construct():
    start_time = time.time()
    
    V, norms_sq = generate_candidates(max_norm_sq=5)
    N_total = len(V)
    
    # Precompute Gram matrix for all candidates? 
    # N is ~11000. 11000x11000 int64 is ~1GB. Too big for RAM? 
    # 11000 * 11000 * 8 bytes = 968 MB. Might be tight.
    # Let's compute conflict matrices per R^2 subset to save memory/time.
    
    best_k = 0
    best_set_indices = []
    
    # Try different R^2 values
    # R^2 = 2, 3, 4, 5
    for R_sq in [2, 3, 4, 5]:
        if time.time() - start_time > DEADLINE_S:
            break
            
        # Filter candidates with norm^2 <= R_sq
        mask = norms_sq <= R_sq
        indices = np.where(mask)[0]
        N = len(indices)
        if N == 0:
            continue
            
        V_sub = V[indices]
        norms_sub = norms_sq[indices]
        
        # Compute Gram matrix for this subset
        # G = V_sub @ V_sub.T
        # To save memory, we can compute in chunks if N is large, but 11k is manageable.
        G = V_sub @ V_sub.T
        
        # Compute conflict mask
        # Condition: 2 * G[i,j] > norms[i]^2 + norms[j]^2 - R_sq
        sum_norms = norms_sub[:, None] + norms_sub[None, :]
        threshold = sum_norms - R_sq
        # Conflict if 2*G > threshold
        # Note: G[i,i] = norms[i]^2. 
        # 2*norms[i]^2 > 2*norms[i]^2 - R_sq => R_sq > 0 (True). 
        # So diagonal is always "conflict" with self, which is correct (we don't pick same vector).
        
        conflict = 2 * G > threshold
        np.fill_diagonal(conflict, False) # Self is not a conflict in graph theory sense for IS, but we don't pick it anyway.
        
        # Degrees
        degrees = np.sum(conflict, axis=1)
        
        # Greedy Init
        # Sort by degree ascending
        order = np.argsort(degrees)
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if time.time() - start_time > DEADLINE_S:
                break
            if selected_mask[idx]:
                continue
            if not np.any(conflict[idx] & selected_mask):
                selected.append(idx)
                selected_mask[idx] = True
                
        if len(selected) > best_k:
            best_k = len(selected)
            # Map back to original indices
            best_set_indices = [indices[i] for i in selected]
            
        # Local Search
        # Maintain conflict counts for nodes NOT in selected
        # count[v] = number of neighbors of v in selected
        # If count[v] == 0, v can be added.
        
        # Initialize counts
        # For each v, count = sum(conflict[v, selected])
        # We can compute this as: 
        # counts = np.zeros(N)
        # for s in selected: counts += conflict[:, s]
        # This is O(|S| * N).
        
        # Better: 
        # selected_idx = np.array(selected)
        # counts = conflict[:, selected_idx].sum(axis=1)
        
        selected_arr = np.array(selected)
        counts = conflict[:, selected_arr].sum(axis=1)
        
        improved = True
        stagnation_counter = 0
        
        while improved and (time.time() - start_time < DEADLINE_S):
            improved = False
            stagnation_counter += 1
            if stagnation_counter > 50:
                break
                
            # 1. Try to add any node with count == 0
            addable = np.where((counts == 0) & (~selected_mask))[0]
            if len(addable) > 0:
                # Add all of them?
                # Adding one might make others invalid?
                # No, if they are all compatible with S, and compatible with each other?
                # We don't know if they are compatible with each other.
                # So we must add them one by one or check pairwise.
                # Greedy add:
                added_any = False
                for v in addable:
                    if selected_mask[v]: continue
                    # Check if v is compatible with all currently selected
                    # It was compatible with S_old.
                    # It might conflict with nodes added in this loop.
                    if not np.any(conflict[v] & selected_mask):
                        selected.append(v)
                        selected_mask[v] = True
                        selected_arr = np.append(selected_arr, v)
                        # Update counts
                        # counts -= conflict[:, v] ? No, v is added to S.
                        # For any u, count[u] increases by 1 if conflict[u,v] is True.
                        counts += conflict[:, v]
                        added_any = True
                if added_any:
                    improved = True
                    stagnation_counter = 0
                    if len(selected) > best_k:
                        best_k = len(selected)
                        best_set_indices = [indices[i] for i in selected]
                    continue
                    
            # 2. Try Remove 1, Add K > 1
            # Pick a random node u from selected
            if not selected:
                break
                
            # To avoid bias, we can pick the node with highest degree in S?
            # Or just random.
            u_pos = random.randint(0, len(selected) - 1)
            u = selected[u_pos]
            
            # Remove u
            selected.pop(u_pos)
            selected_mask[u] = False
            selected_arr = np.delete(selected_arr, u_pos)
            
            # Update counts: remove u's contribution
            counts -= conflict[:, u]
            
            # Find nodes that are now addable (count == 0)
            # These are nodes v such that conflict[v, u] was True (so count dropped by 1) 
            # and count is now 0.
            # Actually, any node with count 0 is addable.
            # But we know that nodes with count 1 that conflicted with u are now 0.
            # Nodes with count 0 were already 0 (and would have been added in step 1).
            # So we look for nodes v such that:
            # 1. v not in selected
            # 2. conflict[v, u] is True (so it was blocked by u)
            # 3. counts[v] == 0 (now unblocked)
            
            # Efficiently:
            # candidates = np.where((conflict[:, u]) & (~selected_mask) & (counts == 0))[0]
            
            candidates = np.where(conflict[:, u])[0]
            # Filter: not in selected, and count == 0
            valid_candidates = [v for v in candidates if not selected_mask[v] and counts[v] == 0]
            
            if len(valid_candidates) > 1:
                # Add all valid candidates
                for v in valid_candidates:
                    selected.append(v)
                    selected_mask[v] = True
                    selected_arr = np.append(selected_arr, v)
                    counts += conflict[:, v]
                
                improved = True
                stagnation_counter = 0
                if len(selected) > best_k:
                    best_k = len(selected)
                    best_set_indices = [indices[i] for i in selected]
            elif len(valid_candidates) == 1:
                # Swap: Add the 1 candidate
                v = valid_candidates[0]
                selected.append(v)
                selected_mask[v] = True
                selected_arr = np.append(selected_arr, v)
                counts += conflict[:, v]
                # Net change 0. Accept to explore?
                # Yes, with some probability or always.
                improved = True
                stagnation_counter = 0
                # Update best only if size increased? No, size same.
                # But might lead to better config later.
            else:
                # Revert removal of u
                selected.insert(u_pos, u)
                selected_mask[u] = True
                selected_arr = np.insert(selected_arr, u_pos, u)
                counts += conflict[:, u]
                # No improvement

    if not best_set_indices:
        # Fallback: return empty or smallest valid set
        # Find any valid set
        # Just return first few vectors?
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return V[best_set_indices]

if __name__ == "__main__":
    res = construct()
    print(f"Found {res.shape[0]} vectors")
