"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # 90s limit, use 85s to be safe
    deadline = start_time + 85

    # 1. Generate all 4-subsets of [1, 16] as bitmasks
    # Elements 1..16 map to bits 0..15
    all_masks = []
    for c in combinations(range(16), 4):
        m = 0
        for x in c:
            m |= (1 << x)
        all_masks.append(m)
        
    n_sets = len(all_masks)
    
    # 2. Precompute pair properties for fast conflict checking
    # For a pair (i, j), we need:
    # K = M_i & M_j
    # DA = M_i & ~K  (Elements in M_i not in K)
    # DB = M_j & ~K  (Elements in M_j not in K)
    # Conflict with new M if:
    # (M & K) == K  AND (M & DA) == 0 AND (M & DB) == 0
    # Which simplifies to:
    # (M & K) == K AND (M & (DA | DB)) == 0
    
    # We will store these in lists indexed by pair index
    # Pair index p corresponds to (i, j) where i < j
    # We can iterate i, j and append to lists
    
    pair_K = []
    pair_DA = []
    pair_DB = []
    pair_i = []
    pair_j = []
    
    for i in range(n_sets):
        mi = all_masks[i]
        for j in range(i + 1, n_sets):
            mj = all_masks[j]
            k = mi & mj
            da = mi & ~k
            db = mj & ~k
            pair_K.append(k)
            pair_DA.append(da)
            pair_DB.append(db)
            pair_i.append(i)
            pair_j.append(j)
            
    # To make lookup faster during greedy, we might need to map (i,j) to index in these lists.
    # Or, we can just iterate over the selected sets and compute on the fly?
    # Computing on the fly: 3 bitwise ops.
    # Precomputation: 1 lookup.
    # Given n_sets=1820, pairs ~1.6M.
    # In the greedy loop, we check O(|F|^2) pairs.
    # If |F| is 100, pairs ~5000.
    # 5000 lookups vs 5000 bitwise ops. Bitwise ops are very fast in Python C implementation.
    # Precomputation might not be worth the memory overhead and index management complexity.
    # Let's try computing on the fly first, but optimize the loop structure.
    
    best_family = []
    
    # Helper to check if a mask can be added to a family (list of masks)
    def can_add(m, family):
        n_fam = len(family)
        for i in range(n_fam):
            mi = family[i]
            for j in range(i + 1, n_fam):
                mj = family[j]
                k = mi & mj
                # Check if m forms sunflower with mi, mj
                # Condition: m & k == k AND m & (mi ^ k) == 0 AND m & (mj ^ k) == 0
                # Note: mi ^ k is same as mi & ~k
                if (m & k) == k:
                    if (m & (mi ^ k)) == 0:
                        if (m & (mj ^ k)) == 0:
                            return False
        return True

    # Optimization: 
    # The inner loop is the bottleneck.
    # We can pre-compute the "conflict signature" for each pair in the family?
    # No, the family changes.
    
    # Let's try a different greedy strategy:
    # Instead of checking all pairs for each new item, 
    # we can maintain a list of "forbidden" conditions?
    # A new set M is forbidden if it matches the pattern of any pair (A, B) in F.
    # Pattern for (A, B): 
    #   Core K = A & B
    #   Forbidden1 = A \ K
    #   Forbidden2 = B \ K
    #   M must contain K and be disjoint from Forbidden1 U Forbidden2.
    
    # We can store for each pair (i, j) in F:
    #   K_ij, F_ij (union of forbidden parts)
    #   Check: (M & K_ij) == K_ij AND (M & F_ij) == 0
    
    def can_add_optimized(m, family, pair_props):
        # pair_props is a list of (K, F) for each pair in family
        # We need to update pair_props when family grows.
        # This is O(|F|) to add new pair props.
        # Total O(|F|^2) storage and O(|F|^2) time per insertion?
        # No, we check all existing pairs.
        
        for K, F in pair_props:
            if (m & K) == K:
                if (m & F) == 0:
                    return False
        return True

    def add_to_family(m, family, pair_props, all_masks_set):
        # Check if m can be added
        if not can_add_optimized(m, family, pair_props):
            return False
        
        # Add m
        family.append(m)
        
        # Update pair_props with new pairs involving m
        n = len(family) - 1
        for i in range(n):
            mi = family[i]
            k = mi & m
            f = (mi ^ k) | (m ^ k) # (mi \ k) U (m \ k)
            # Note: mi ^ k is mi & ~k
            pair_props.append((k, f))
            
        return True

    # Randomized Greedy with Restart
    # Strategy:
    # 1. Standard random shuffle greedy.
    # 2. "Min-degree" greedy: Select the candidate with the fewest conflicts with the current family.
    #    This is more expensive per step (O(N * |F|^2)) but might yield better local optima.
    #    Given the time budget, we will mix these.
    
    best_len = 0
    best_family_list = []
    
    # We will run several restarts.
    # To maximize exploration, we alternate between:
    # - Fast random greedy (many restarts)
    # - Slower min-degree greedy (fewer restarts, better quality)
    
    def greedy_random(order):
        family = []
        pair_props = []
        for m in order:
            if time.time() > deadline - 1:
                break
            if add_to_family(m, family, pair_props, set(all_masks)):
                pass
        return family

    def greedy_min_degree():
        # Maintain a set of available candidates
        available = set(all_masks)
        family = []
        pair_props = []
        
        while available:
            if time.time() > deadline - 1:
                break
            
            best_m = None
            min_conflicts = float('inf')
            
            # To avoid O(N * |F|^2) being too slow, we can sample or just do it if |F| is small
            # But |F| grows. Let's just iterate through available.
            # Optimization: If |F| is large, this is very slow.
            # We can break early if we find a valid one with 0 conflicts? 
            # No, min-degree implies we want the one that "blocks" the least future options.
            # A simple heuristic: pick the one that is valid and has the smallest size? No, all size 4.
            # Let's stick to: pick the valid candidate that minimizes the number of *new* forbidden pairs created?
            # That's hard to compute exactly without adding it.
            
            # Alternative Min-Degree:
            # Count how many other available sets would be blocked if we add m?
            # Too expensive.
            
            # Let's try a simpler "greedy with lookahead" or just random is often good enough for this size.
            # However, the directive asks for a structural change.
            # Let's implement a "weighted" greedy where we prioritize sets that are "rare" or have low intersection density with existing?
            
            # Actually, let's just do a few passes of random greedy, but with a twist:
            # "Reverse" greedy: Start with the set that has the most conflicts (most central), remove it and its neighbors? No, that's for independent set in graphs.
            
            # Let's try: Sort candidates by a heuristic score based on the current family.
            # Score(m) = number of sets in family that intersect m in exactly 2 elements?
            # Or simply: Random is robust. Let's just do many random restarts.
            
            # To satisfy "structural change", let's implement a simple local search improvement on the result of a greedy pass.
            # But local search is hard to define here.
            
            # Let's just run a large number of random restarts, which is the most reliable way to beat the seed.
            # The seed uses 500 restarts. Let's use the full time budget for as many as possible.
            
            # We'll just return an empty family here to force the main loop to use greedy_random
            break 
            
        return family

    # Main loop: Many random restarts
    # Estimate time per restart:
    # 1820 candidates.
    # Each addition check is O(|F|^2).
    # Average |F| ~ 50? Pairs ~ 1250.
    # 1820 * 1250 ~ 2.2M operations per restart.
    # In Python, 2.2M simple ops might take ~0.1s - 0.5s?
    # Let's say 0.2s per restart.
    # 85s / 0.2s = 425 restarts.
    # The seed did 500 restarts and got 0.7.
    # Why? Maybe the seed's `add_to_family` is slower due to list appends in the inner loop?
    # My `add_to_family` appends to `pair_props` inside the loop for each new pair.
    # This is O(|F|) per addition.
    
    # Let's optimize `add_to_family` to be faster.
    # Instead of `pair_props` list, can we use a different structure?
    # The check `can_add_optimized` iterates `pair_props`.
    # If we just use `family` and compute on the fly in `can_add`, it's 3 bitwise ops per pair.
    # `pair_props` stores (K, F). Check is 2 bitwise ops + 2 comparisons.
    # Storing (K, F) means we do 1 OR and 1 XOR during update.
    # The overhead of maintaining `pair_props` is O(|F|) per insertion.
    # The check is O(|F|^2).
    # For small |F|, update is negligible.
    
    # Let's try to speed up the random restarts by using a faster conflict check.
    # Precomputing all pair properties for ALL 1820 sets is ~1.6M pairs.
    # Memory: 1.6M * (4 bytes * 3) ~ 20MB. Acceptable.
    # Time to precompute: 1.6M * (3 bitwise ops) ~ 5M ops. Fast (<0.1s).
    
    # Let's precompute a dictionary or list for all pairs of ALL sets?
    # Then `can_add` just looks up the pair properties for (i, j) in the family?
    # But the family indices change? No, if we keep the original indices.
    
    # Let's restructure:
    # `family_indices` = list of indices into `all_masks`.
    # Precomputed `pair_props_all[i][j]` = (K, F) for all i < j.
    # Then `can_add(m_idx, family_indices)`:
    #   for i in range(len(family_indices)):
    #     for j in range(i+1, len(family_indices)):
    #       K, F = pair_props_all[family_indices[i]][family_indices[j]]
    #       if (m & K) == K and (m & F) == 0: return False
    # This avoids the update cost of `pair_props` and the bitwise ops in the inner loop.
    # It replaces bitwise ops with list lookups.
    
    # Let's implement this.
    
    # Precompute pair props for all pairs
    # n_sets = 1820
    # pair_props_all = [[None]*n_sets for _ in range(n_sets)]
    # This is 1820*1820 = 3.3M entries.
    # Each entry is a tuple of 2 ints.
    # Memory might be high but let's try.
    # Or a flat list indexed by i*n_sets + j.
    
    pair_props_flat = [None] * (n_sets * n_sets)
    
    for i in range(n_sets):
        mi = all_masks[i]
        for j in range(i + 1, n_sets):
            mj = all_masks[j]
            k = mi & mj
            f = (mi ^ k) | (mj ^ k)
            # Store for both (i,j) and (j,i) to avoid index check?
            # Or just store for i<j and check in loop.
            pair_props_flat[i * n_sets + j] = (k, f)
            pair_props_flat[j * n_sets + i] = (k, f)
            
    def can_add_fast(m_idx, family_indices):
        m = all_masks[m_idx]
        n_fam = len(family_indices)
        for i in range(n_fam):
            idx_i = family_indices[i]
            for j in range(i + 1, n_fam):
                idx_j = family_indices[j]
                k, f = pair_props_flat[idx_i * n_sets + idx_j]
                if (m & k) == k:
                    if (m & f) == 0:
                        return False
        return True

    def greedy_random_indices(order):
        family_indices = []
        for m_idx in order:
            if time.time() > deadline - 1:
                break
            if can_add_fast(m_idx, family_indices):
                family_indices.append(m_idx)
        return family_indices

    # Run many random restarts
    best_len = 0
    best_family_indices = []
    
    # Pre-shuffled orders? No, shuffle each time.
    # We can try to be smart:
    # 1. Random
    # 2. Sorted by "popularity" (number of elements in common with others)?
    
    # Let's just do random first.
    
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        order = list(range(n_sets))
        random.shuffle(order)
        
        fam = greedy_random_indices(order)
        
        if len(fam) > best_len:
            best_len = len(fam)
            best_family_indices = fam[:]
            
            # If we get a very good result, we might want to stop?
            # Max is 250.
            if best_len > 200:
                # Try a few more to see if we can break 200
                pass
            if best_len >= 250:
                break

    # Convert best_family_indices (indices) back to sorted lists of 1..16
    result = []
    for idx in best_family_indices:
        m = all_masks[idx]
        s = []
        for i in range(16):
            if m & (1 << i):
                s.append(i + 1)
        result.append(s)
        
    return result
# EVOLVE-BLOCK-END
