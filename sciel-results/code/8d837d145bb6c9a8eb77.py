# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Uses Bitset-based Local Search (Add/Swap) with Multi-Start.
    """
    D = 11
    deadline = time.time() + 115.0 # Safety margin
    
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
            
    # Type 5: four ±1's (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Type 6: one ±2 and one ±1 (norm^2 = 5)
    # If max norm^2 becomes 5, dist^2 >= 5.
    # Let's see if this helps. It increases max_norm_sq.
    # For now, stick to max norm^2 = 4 to keep constraints loose.
    # If we add norm^2=5, we must remove norm^2=4 vectors that conflict?
    # No, the constraint is global: min dist >= max norm.
    # If we include a vector with norm^2=5, ALL pairs must have dist^2 >= 5.
    # This is a much tighter constraint.
    # It's usually better to stick to lower norms if we want density, unless high norm vectors are very sparse.
    # Given D=11, low norm vectors are dense.
    # So we stick to max norm^2 = 4.
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute pairwise dot products
    dots = candidates @ candidates.T
    
    # Compute squared distances
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict matrix: conflict[i,j] = True if dist_sq[i,j] < max_norm_sq
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # Convert to bitsets for fast operations
    # Python integers can handle large bitsets efficiently
    conflict_bits = [0] * n
    for i in range(n):
        # Get indices of conflicts
        conflicts = np.where(conflict[i])[0]
        bits = 0
        for j in conflicts:
            bits |= (1 << j)
        conflict_bits[i] = bits
        
    # Helper to count bits
    def popcount(x):
        return bin(x).count("1")
        
    best_size = 0
    best_selection = []
    
    np.random.seed(42)
    
    def local_search(selected_set):
        """
        Improve the independent set using Add-if-possible and Swap strategies.
        """
        selected_bits = 0
        for idx in selected_set:
            selected_bits |= (1 << idx)
            
        size = len(selected_set)
        
        # Precompute list of all indices for random sampling
        all_indices = list(range(n))
        
        # Iterative improvement
        # We try to add nodes that don't conflict.
        # Then we do random swaps to escape local optima.
        
        # Phase 1: Greedy Add
        # We scan all unselected nodes. If one fits, we add it.
        # This is O(N * popcount) which is fast.
        changed = True
        while changed:
            changed = False
            for v in all_indices:
                if (selected_bits >> v) & 1:
                    continue
                if (conflict_bits[v] & selected_bits) == 0:
                    # Can add v
                    selected_bits |= (1 << v)
                    size += 1
                    changed = True
                    # Restart scan to ensure we catch any newly unblocked nodes?
                    # Actually, adding a node only blocks more, so it doesn't unblock others.
                    # But it might make a previously "blocked" node unblocked? No.
                    # So we can just continue scanning?
                    # Wait, if we add v, it might conflict with some u.
                    # u was unselected. It remains unselected.
                    # So we don't need to restart. We can just continue the loop.
                    # However, if we break, we might miss other additions.
                    # Let's just continue the loop.
            
        # Phase 2: Random Swaps
        # Try to swap a selected node with an unselected node to change the configuration.
        # We do this for a fixed number of steps or until no improvement in a window.
        
        # We track the best size found during this local search
        local_best_size = size
        local_best_bits = selected_bits
        
        # To allow escape, we do random swaps.
        # A swap is valid if we remove u (selected) and add v (unselected)
        # such that v does not conflict with (selected - {u}).
        
        # We will perform a number of random swap attempts.
        # If we find a state with larger size, we keep it.
        # But swaps keep size constant.
        # So we need a sequence of swaps that leads to a state where an add is possible.
        
        # Strategy:
        # 1. Pick random v (unselected).
        # 2. Let mask = conflict_bits[v] & selected_bits.
        # 3. If popcount(mask) == 0: Add v (handled in Phase 1, but maybe we missed it? No, Phase 1 was exhaustive).
        #    Wait, Phase 1 was exhaustive on the *current* selected set.
        #    If we do swaps, the selected set changes.
        #    So after a swap, we should check for adds again.
        
        # Let's integrate Add check into the Swap loop.
        
        max_swap_attempts = 10000
        attempts = 0
        no_improve_count = 0
        threshold = 1000
        
        while attempts < max_swap_attempts and time.time() < deadline:
            attempts += 1
            
            # Pick a random unselected node v
            # To pick efficiently, we can just pick random from all_indices and check if selected
            v = np.random.choice(all_indices)
            if (selected_bits >> v) & 1:
                continue
                
            mask = conflict_bits[v] & selected_bits
            
            if mask == 0:
                # Can add v
                selected_bits |= (1 << v)
                size += 1
                no_improve_count = 0
                if size > local_best_size:
                    local_best_size = size
                    local_best_bits = selected_bits
            elif popcount(mask) == 1:
                # Find the single conflict u
                # mask is a power of 2
                u = mask.bit_length() - 1
                
                # Perform swap: remove u, add v
                # Check if valid: v should not conflict with selected - {u}
                # Since mask had only u, v does not conflict with others.
                # So swap is valid.
                
                selected_bits &= ~(1 << u)
                selected_bits |= (1 << v)
                # size remains same
                no_improve_count += 1
            else:
                # Multiple conflicts. 
                # We could try to resolve them, but for now, skip.
                # Or try to remove 2 nodes? 
                # Let's stick to 1-for-1 swaps for speed.
                no_improve_count += 1
                
            if no_improve_count > threshold:
                # We are stuck. Break and try a different restart?
                # Or just stop local search for this restart.
                break
        
        # Extract best selection from this run
        # We need to reconstruct the set from bits
        best_set = set()
        temp_bits = local_best_bits
        while temp_bits:
            # Get lowest set bit
            lsb = temp_bits & -temp_bits
            idx = lsb.bit_length() - 1
            best_set.add(idx)
            temp_bits ^= lsb
            
        return list(best_set), local_best_size

    # Multi-Start
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Randomized Greedy Initialization
        # We can use Min-Degree or just Random selection with conflict checking.
        # Let's use a simple greedy: pick random valid node, add it.
        # This is fast and provides diversity.
        
        selected_set = set()
        selected_bits = 0
        
        # To speed up init, we can shuffle candidates and pick valid ones
        # But we need to check conflicts.
        
        # Alternative: Use the "Add-if-possible" loop on an empty set?
        # That's just picking all non-conflicting? No, they conflict with each other.
        
        # Let's do a quick greedy:
        # 1. Shuffle indices
        # 2. Iterate, if node doesn't conflict with current selected, add it.
        
        shuffled_indices = np.random.permutation(n)
        for idx in shuffled_indices:
            if (conflict_bits[idx] & selected_bits) == 0:
                selected_bits |= (1 << idx)
                selected_set.add(idx)
                
        # Run Local Search
        improved_selection, current_size = local_search(selected_set)
        
        if current_size > best_size:
            best_size = current_size
            best_selection = improved_selection
            # Optional: print progress if debug
            # print(f"Restart {restart}: Found {best_size}")
            
    result = candidates[best_selection]
    return result
# EVOLVE-BLOCK-END
