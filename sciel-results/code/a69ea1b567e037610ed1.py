import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Precompute masks for all triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        
    # Precompute pair masks for linearity check
    # Pair index: x * n + y for x < y. Max pair index < 16*16 = 256. Fits in int.
    tpair = [0] * num_triples
    for i, (a, b, c) in enumerate(all_triples):
        pm = 0
        for x, y in ((a, b), (a, c), (b, c)):
            if x > y:
                x, y = y, x
            pm |= 1 << (x * n + y)
        tpair[i] = pm
        
    # Precompute popcount array for all 2^16 masks
    pop_arr = np.zeros(1 << n, dtype=np.int8)
    for i in range(1, 1 << n):
        pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        
    # Precompute subsets array for vectorized coverage check
    subsets = np.arange(1 << n, dtype=np.int32)
    
    def compute_alpha_fast(triple_indices):
        """
        Compute the independence number of the hypergraph defined by triple_indices.
        Returns the size of the largest subset of points containing no triple.
        """
        if not triple_indices:
            return n
        
        covered = np.zeros(1 << n, dtype=bool)
        
        for idx in triple_indices:
            t_mask = triple_masks[idx]
            covered |= (subsets & t_mask) == t_mask
            
        max_alpha = np.max(pop_arr[~covered])
        return int(max_alpha)
        
    def find_independent_set(triple_indices, max_size):
        """
        Find an independent set of size max_size.
        Returns a mask of the set or 0 if not found.
        """
        if not triple_indices:
            return (1 << n) - 1
        
        covered = np.zeros(1 << n, dtype=bool)
        for idx in triple_indices:
            t_mask = triple_masks[idx]
            covered |= (subsets & t_mask) == t_mask
            
        # Find subsets with popcount == max_size and not covered
        candidates = np.where((pop_arr == max_size) & (~covered))[0]
        if len(candidates) > 0:
            return int(candidates[0])
        return 0

    def greedy_packing(order):
        sys_set = set()
        pair_union = 0
        for idx in order:
            if (tpair[idx] & pair_union) == 0:
                sys_set.add(idx)
                pair_union |= tpair[idx]
        return list(sys_set), pair_union

    # Initial solution: Random greedy
    order = list(range(num_triples))
    random.shuffle(order)
    current_packing, current_pair_union = greedy_packing(order)
    current_alpha = compute_alpha_fast(current_packing)
    current_set = set(current_packing)
    
    global_best_alpha = current_alpha
    global_best_packing = list(current_packing)
    
    # Map from triple mask to index for quick lookup? 
    # Or just iterate.
    
    while time.time() < deadline:
        # 1. Compute current alpha
        # We already have it, but after changes we update.
        
        if current_alpha <= 5:
            # Found optimal or near-optimal. Try to polish or break ties.
            # If alpha is 5, we can stop or try to ensure stability.
            # For now, we can just keep searching for better distributions or confirm.
            pass
        
        # Find a specific independent set of size current_alpha
        ind_mask = find_independent_set(current_packing, current_alpha)
        
        if ind_mask == 0:
            # Should not happen if current_alpha is correct
            continue
            
        # Get list of points in the independent set
        ind_points = [i for i in range(n) if (ind_mask >> i) & 1]
        
        improved = False
        
        # Strategy A: Try to add a triple that is a subset of the independent set
        # This directly breaks the independent set.
        # We need a triple T such that T is a subset of ind_points, T is not in current_set,
        # and T is linear with all triples in current_set.
        
        # Generate all triples within ind_points
        # ind_points has size current_alpha.
        # If current_alpha is 6, there are C(6,3)=20 triples.
        # If current_alpha is 5, there are C(5,3)=10 triples.
        
        candidate_adds = []
        for t in itertools.combinations(ind_points, 3):
            # Find index of this triple
            # We can precompute a dict from tuple to index
            # Let's create it once outside the loop if not done
            pass # We'll use a helper or precomputed map

    # Precompute map for fast lookup
    triple_to_idx = {}
    for i, t in enumerate(all_triples):
        triple_to_idx[t] = i
        
    # Reset loop with map available
    # Re-initialize state if needed? No, just continue.
    # But we need to restart the main loop logic with the map.
    
    # Let's restructure the main loop to use the map.
    
    # Re-run initial setup just in case, or just use the existing state.
    # The state is valid.
    
    # Refine the main loop
    while time.time() < deadline:
        if current_alpha <= 5:
            # Try to reduce further? Unlikely if 5 is lower bound.
            # Just break or do minor polish.
            # For now, we assume 5 is the target.
            # If we get 5, we can stop early to save time? 
            # But we might get 4? 
            # Lower bound for v=16 is likely 5.
            # Let's try to break if we have 5 for a while? 
            # Or just keep going to find a robust 5.
            pass
        
        # Find independent set
        ind_mask = find_independent_set(current_packing, current_alpha)
        if ind_mask == 0:
            continue
        
        ind_points = [i for i in range(n) if (ind_mask >> i) & 1]
        
        # 1. Try to add a triple contained in ind_points
        found_add = False
        for t in itertools.combinations(ind_points, 3):
            idx = triple_to_idx.get(t, -1)
            if idx == -1:
                continue
            if idx in current_set:
                continue
            # Check linearity
            if (tpair[idx] & current_pair_union) == 0:
                # Valid addition
                current_packing.append(idx)
                current_set.add(idx)
                current_pair_union |= tpair[idx]
                found_add = True
                break
        
        if found_add:
            current_alpha = compute_alpha_fast(current_packing)
            if current_alpha < global_best_alpha:
                global_best_alpha = current_alpha
                global_best_packing = list(current_packing)
            continue
        
        # 2. Try to swap: Remove a triple T_out, Add a triple T_in subset of ind_points
        # T_in must be linear with all T in current_set \ {T_out}
        # T_out must be one that conflicts with T_in or just any triple.
        # Actually, if T_in is not linear with current_set, it must conflict with some T_out.
        # If we remove T_out, does T_in become linear?
        
        # Find all T_in candidates in ind_points
        candidates_in = []
        for t in itertools.combinations(ind_points, 3):
            idx = triple_to_idx.get(t, -1)
            if idx == -1 or idx in current_set:
                continue
            candidates_in.append(idx)
            
        if not candidates_in:
            # No triples to add. Try random perturbation or swap not involving ind_points?
            # If no triple in ind_points is available, we are stuck for this independent set.
            # Try a random swap or removal.
            if current_packing:
                rem = random.choice(current_packing)
                current_packing.remove(rem)
                current_set.discard(rem)
                current_pair_union = 0
                for idx in current_packing:
                    current_pair_union |= tpair[idx]
                current_alpha = compute_alpha_fast(current_packing)
                # Greedy refill
                candidates = list(range(num_triples))
                random.shuffle(candidates)
                for cand in candidates:
                    if cand not in current_set and (tpair[cand] & current_pair_union) == 0:
                        current_packing.append(cand)
                        current_set.add(cand)
                        current_pair_union |= tpair[cand]
                current_alpha = compute_alpha_fast(current_packing)
                if current_alpha < global_best_alpha:
                    global_best_alpha = current_alpha
                    global_best_packing = list(current_packing)
            continue
            
        # Try each candidate T_in
        # For each T_in, find the set of conflicts in current_set
        # If we remove one of the conflicts, is T_in valid?
        # We want to minimize the number of conflicts. Ideally 1.
        
        best_swap = None
        best_conflict_count = 100
        
        for idx_in in candidates_in:
            conflicts = []
            for idx_out in current_set:
                if (tpair[idx_in] & tpair[idx_out]) != 0:
                    conflicts.append(idx_out)
                    
            if len(conflicts) < best_conflict_count:
                best_conflict_count = len(conflicts)
                best_swap = (idx_in, conflicts)
                
                if best_conflict_count == 0:
                    # This shouldn't happen because we checked for add first
                    break
                if best_conflict_count == 1:
                    # Minimal conflict, good candidate
                    break
        
        if best_swap and best_conflict_count == 1:
            idx_in, conflict = best_swap
            idx_out = conflict[0]
            
            # Perform swap
            current_packing.remove(idx_out)
            current_set.discard(idx_out)
            current_packing.append(idx_in)
            current_set.add(idx_in)
            
            # Recompute pair_union
            current_pair_union = 0
            for idx in current_packing:
                current_pair_union |= tpair[idx]
                
            current_alpha = compute_alpha_fast(current_packing)
            if current_alpha < global_best_alpha:
                global_best_alpha = current_alpha
                global_best_packing = list(current_packing)
            continue
            
        # If no single conflict swap works, try removing 2 conflicts?
        # Or just do a random perturbation.
        
        # Random Perturbation
        n_remove = random.randint(1, 3)
        for _ in range(n_remove):
            if current_packing:
                rem = random.choice(current_packing)
                current_packing.remove(rem)
                current_set.discard(rem)
                
        current_pair_union = 0
        for idx in current_packing:
            current_pair_union |= tpair[idx]
            
        # Greedy refill
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        for cand in candidates:
            if cand not in current_set and (tpair[cand] & current_pair_union) == 0:
                current_packing.append(cand)
                current_set.add(cand)
                current_pair_union |= tpair[cand]
                
        current_alpha = compute_alpha_fast(current_packing)
        if current_alpha < global_best_alpha:
            global_best_alpha = current_alpha
            global_best_packing = list(current_packing)

    out = [list(all_triples[i]) for i in global_best_packing]
    return out

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Num triples: {len(res)}")
    
    # Verify linearity
    masks = set()
    for t in res:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        if m in masks:
            print("Duplicate triple found!")
        masks.add(m)
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            mi = (1 << res[i][0]) | (1 << res[i][1]) | (1 << res[i][2])
            mj = (1 << res[j][0]) | (1 << res[j][1]) | (1 << res[j][2])
            inter = mi & mj
            count = bin(inter).count('1')
            if count > 1:
                print(f"Non-linear: {res[i]} and {res[j]} share {count} points")
    
    # Compute alpha
    def check_alpha(triples):
        n = 16
        pop_arr = np.zeros(1 << n, dtype=np.int8)
        for i in range(1, 1 << n):
            pop_arr[i] = pop_arr[i >> 1] + (i & 1)
        covered = np.zeros(1 << n, dtype=bool)
        subsets = np.arange(1 << n, dtype=np.int32)
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            covered |= (subsets & m) == m
        return int(np.max(pop_arr[~covered]))
    
    print(f"Alpha: {check_alpha(res)}")
