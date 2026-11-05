import itertools
import time
import random
import math

def construct(v, k, t):
    all_points = list(range(v))
    
    # Pre-compute all t-subset masks
    all_t_masks = []
    for combo in itertools.combinations(all_points, t):
        m = 0
        for p in combo:
            m |= (1 << p)
        all_t_masks.append(m)
        
    num_t_subsets = len(all_t_masks)
    
    # Map mask to index for faster array access
    mask_to_idx = {m: i for i, m in enumerate(all_t_masks)}
    
    def get_block_subsets_mask(block_pts):
        """Returns a list of t-subset masks covered by the block."""
        subs = []
        for sub in itertools.combinations(block_pts, t):
            m = 0
            for p in sub:
                m |= (1 << p)
            subs.append(m)
        return subs

    def run_fast_greedy():
        """Generate a valid covering quickly using standard greedy."""
        blocks = []
        cover_count = [0] * num_t_subsets
        uncovered_count = num_t_subsets
        
        # Pre-compute subsets for all possible blocks? No, too much memory/time.
        # Just compute on the fly.
        
        while uncovered_count > 0:
            # Pick a random uncovered subset to break symmetry
            # Find an index with count 0
            # Optimization: Keep a set of uncovered indices?
            # For speed in initial phase, just pick a random index and check.
            # Or iterate.
            
            # To speed up, let's pick a random point and find an uncovered subset containing it?
            # No, just random index.
            
            # Find an uncovered mask
            start_idx = -1
            for _ in range(100): # Try a few random picks
                idx = random.randint(0, num_t_subsets - 1)
                if cover_count[idx] == 0:
                    start_idx = idx
                    break
            if start_idx == -1:
                # Fallback: linear scan
                for i in range(num_t_subsets):
                    if cover_count[i] == 0:
                        start_idx = i
                        break
            
            if start_idx == -1:
                break # Should not happen if uncovered_count > 0
                
            start_mask = all_t_masks[start_idx]
            
            # Generate candidate blocks containing start_mask
            start_pts = []
            for i in range(v):
                if (start_mask >> i) & 1:
                    start_pts.append(i)
            
            available = [p for p in all_points if p not in start_pts]
            
            best_gain = -1
            best_block_pts = None
            
            for extra in itertools.combinations(available, k - t):
                full_pts = start_pts + list(extra)
                
                # Calculate gain
                gain = 0
                for sub in itertools.combinations(full_pts, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    idx = mask_to_idx[m]
                    if cover_count[idx] == 0:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_block_pts = full_pts
            
            if best_block_pts is None:
                break
                
            blocks.append(best_block_pts)
            
            # Update counts
            for sub in itertools.combinations(best_block_pts, t):
                m = 0
                for p in sub:
                    m |= (1 << p)
                idx = mask_to_idx[m]
                if cover_count[idx] == 0:
                    uncovered_count -= 1
                cover_count[idx] += 1
                
        return blocks, cover_count, uncovered_count

    def local_search_sa(initial_blocks, initial_counts, initial_uncovered):
        """
        Simulated Annealing to minimize number of blocks.
        State: list of blocks.
        Energy: len(blocks) + penalty * uncovered_count
        """
        blocks = [list(b) for b in initial_blocks]
        cover_count = list(initial_counts)
        uncovered_count = initial_uncovered
        
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        best_uncovered = uncovered_count
        
        start_time = time.time()
        time_limit = 110
        T = 100.0
        cooling = 0.999
        
        # Pre-compute subsets for current blocks to speed up updates?
        # No, updating is O(56).
        
        def apply_move(move_type, idx1, idx2=None):
            """
            move_type: 'add', 'remove', 'swap'
            idx1: index in blocks list
            idx2: index of block to add/swap with (if applicable)
            Returns (delta_len, delta_uncovered, new_block_pts_or_None, removed_block_pts_or_None)
            """
            delta_len = 0
            delta_unc = 0
            
            if move_type == 'remove':
                b = blocks[idx1]
                # Check how many unique subsets this block covers
                # If it covers any subset with count 1, removing it increases uncovered.
                # If it covers only subsets with count > 1, removing it is safe.
                
                # Calculate delta_uncovered
                for sub in itertools.combinations(b, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    idx = mask_to_idx[m]
                    if cover_count[idx] == 1:
                        delta_unc += 1
                
                delta_len = -1
                return delta_len, delta_unc, None, b

            elif move_type == 'add':
                # idx1 is the block to add (passed as points)
                b = idx1 # idx1 is actually points here in this helper logic, let's adjust
                # This helper is getting messy. Let's do it inline in the loop.
                pass
            
            return 0, 0, None, None

        # Simpler inline SA loop
        while time.time() - start_time < time_limit:
            # Pick a random move
            r = random.random()
            
            if r < 0.5:
                # Remove a random block
                if not blocks:
                    continue
                idx = random.randint(0, len(blocks) - 1)
                b = blocks[idx]
                
                # Calculate cost of removal
                delta_unc = 0
                for sub in itertools.combinations(b, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    idx_m = mask_to_idx[m]
                    if cover_count[idx_m] == 1:
                        delta_unc += 1
                
                delta_len = -1
                # Energy change: -1 + penalty * delta_unc
                # We want to minimize len + M * unc
                # Let's use M = 1000
                M = 1000
                delta_E = delta_len + M * delta_unc
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Accept
                    # Update counts
                    for sub in itertools.combinations(b, t):
                        m = 0
                        for p in sub:
                            m |= (1 << p)
                        idx_m = mask_to_idx[m]
                        cover_count[idx_m] -= 1
                        if cover_count[idx_m] == 0:
                            uncovered_count += 1
                    blocks.pop(idx)
                    
                    if uncovered_count == 0 and len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = [list(x) for x in blocks]
                        
            elif r < 0.8:
                # Add a random block
                # Generate a random k-subset
                new_b = random.sample(all_points, k)
                
                # Calculate gain (reduction in uncovered)
                delta_unc = 0
                for sub in itertools.combinations(new_b, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    idx_m = mask_to_idx[m]
                    if cover_count[idx_m] == 0:
                        delta_unc -= 1 # Decreases uncovered count
                
                delta_len = 1
                M = 1000
                delta_E = delta_len + M * delta_unc
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Accept
                    blocks.append(new_b)
                    for sub in itertools.combinations(new_b, t):
                        m = 0
                        for p in sub:
                            m |= (1 << p)
                        idx_m = mask_to_idx[m]
                        if cover_count[idx_m] == 0:
                            uncovered_count -= 1
                        cover_count[idx_m] += 1
                        
                    if uncovered_count == 0 and len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = [list(x) for x in blocks]

            else:
                # Swap: Remove one, Add one
                if not blocks:
                    continue
                idx = random.randint(0, len(blocks) - 1)
                b_old = blocks[idx]
                b_new = random.sample(all_points, k)
                
                # Calculate net delta
                # Remove b_old
                delta_unc_remove = 0
                for sub in itertools.combinations(b_old, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    idx_m = mask_to_idx[m]
                    if cover_count[idx_m] == 1:
                        delta_unc_remove += 1
                
                # Add b_new
                # Note: b_new might cover some subsets that b_old covered.
                # We need to calculate the net change carefully.
                # It's easier to just simulate the counts temporarily?
                # Or calculate:
                # 1. Subsets covered ONLY by b_old (count becomes 0 if removed, then b_new might cover them)
                # 2. Subsets covered ONLY by b_new (count was 0, becomes 1)
                
                # Let's just calculate the new uncovered count directly for the affected subsets.
                # Affected subsets are union of subsets in b_old and b_new.
                
                # For efficiency, let's just do the add/remove logic if accepted,
                # but for decision, we need delta_E.
                
                # Simplified Delta:
                # Assume we remove b_old. Uncovered increases by delta_unc_remove.
                # Then we add b_new. Uncovered decreases by (number of subsets in b_new that are currently uncovered *AFTER* removal).
                
                # This is complex. Let's just use a heuristic:
                # If b_new covers more "unique" subsets than b_old, it's good.
                
                # For simplicity and speed, let's skip Swap in the first version of SA?
                # Or implement a simple swap:
                # 1. Tentatively remove b_old.
                # 2. Count how many of b_new's subsets are now uncovered.
                # 3. Net delta_unc = delta_unc_remove - (count of b_new subsets that were 0 after removal).
                
                # To do this, we need to know the state after removal.
                # We can compute "would be uncovered" for b_new's subsets.
                
                # Let's optimize:
                # For each subset in b_new:
                #   if cover_count[sub] == 0:
                #       if sub is in b_old:
                #           # It was covered by b_old. After removal, count is 0.
                #           # So adding b_new covers it.
                #           # Net change: 0 (was covered, removed, added back)
                #           # BUT if count was 1, it becomes 0 then 1.
                #       else:
                #           # It was uncovered. Adding b_new covers it.
                #           # Net change: -1 (uncovered -> covered)
                #   if cover_count[sub] > 0:
                #       if sub is in b_old and cover_count[sub] == 1:
                #           # It was covered ONLY by b_old.
                #           # After removal, it is uncovered.
                #           # Adding b_new covers it.
                #           # Net change: 0
                #       else:
                #           # Net change: 0
                
                # So, net delta_unc = (number of subsets in b_old with count 1 that are NOT in b_new)
                #                    - (number of subsets in b_new with count 0 that are NOT in b_old)
                
                # Let's compute this.
                
                # Sets of masks
                # To avoid set creation overhead, we can use a dict or just iterate.
                # Given small sizes (56), sets are fine.
                
                masks_old = set()
                for sub in itertools.combinations(b_old, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    masks_old.add(m)
                    
                masks_new = set()
                for sub in itertools.combinations(b_new, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    masks_new.add(m)
                    
                # Count 1s in old not in new
                c1 = 0
                for m in masks_old:
                    if m not in masks_new and cover_count[mask_to_idx[m]] == 1:
                        c1 += 1
                        
                # Count 0s in new not in old
                c2 = 0
                for m in masks_new:
                    if m not in masks_old and cover_count[mask_to_idx[m]] == 0:
                        c2 += 1
                        
                delta_unc = c1 - c2
                delta_len = 0 # Swap doesn't change length
                
                M = 1000
                delta_E = delta_len + M * delta_unc
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    # Accept
                    # Remove b_old
                    for m in masks_old:
                        idx_m = mask_to_idx[m]
                        cover_count[idx_m] -= 1
                        if cover_count[idx_m] == 0:
                            uncovered_count += 1
                    # Add b_new
                    for m in masks_new:
                        idx_m = mask_to_idx[m]
                        if cover_count[idx_m] == 0:
                            uncovered_count -= 1
                        cover_count[idx_m] += 1
                        
                    blocks[idx] = b_new
                    
                    if uncovered_count == 0 and len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = [list(x) for x in blocks]

            T *= cooling
            
        return best_blocks

    # 1. Generate initial solution
    init_blocks, init_counts, init_unc = run_fast_greedy()
    
    # 2. Prune initial solution
    # Simple prune: remove blocks that don't cover any unique subset
    changed = True
    while changed:
        changed = False
        for i in range(len(init_blocks) - 1, -1, -1):
            b = init_blocks[i]
            essential = False
            for sub in itertools.combinations(b, t):
                m = 0
                for p in sub:
                    m |= (1 << p)
                idx = mask_to_idx[m]
                if init_counts[idx] == 1:
                    essential = True
                    break
            if not essential:
                # Remove
                for sub in itertools.combinations(b, t):
                    m = 0
                    for p in sub:
                        m |= (1 << p)
                    idx = mask_to_idx[m]
                    init_counts[idx] -= 1
                init_blocks.pop(i)
                changed = True
    
    # Recalculate uncovered count for SA
    init_unc = 0
    for c in init_counts:
        if c == 0:
            init_unc += 1
            
    # 3. Run SA
    best_blocks = local_search_sa(init_blocks, init_counts, init_unc)
    
    return best_blocks
