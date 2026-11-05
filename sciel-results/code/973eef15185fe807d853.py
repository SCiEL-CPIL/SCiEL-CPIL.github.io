import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Evaluator kills at 150s. We aim to finish well before to be safe.
    # Let's set a deadline of 140s to leave 10s buffer.
    deadline = start_time + 140.0

    # Precompute all t-subsets (triplets)
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    # For v=24, k=7, this is 346,104 blocks.
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the triplets it covers
    # n_triplets is 2024, fits in a Python int easily.
    block_masks = [0] * n_blocks
    # Also store the list of triplet indices for faster iteration if needed, 
    # but bitmasks are great for "count newly covered"
    block_triplet_indices = [None] * n_blocks
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        tids = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            tids.append(tid)
        block_masks[b_idx] = mask
        block_triplet_indices[b_idx] = tids

    # Full mask of all triplets
    all_mask = (1 << n_triplets) - 1

    def greedy_set_cover(seed=0, time_limit_frac=0.6):
        """
        Runs a greedy set cover algorithm.
        Returns a list of blocks (tuples of points).
        """
        rng = random.Random(seed)
        uncovered = all_mask
        solution_indices = []
        used_blocks = [False] * n_blocks
        
        # We don't strictly need 'used_blocks' if we just track 'uncovered', 
        # because a used block will have gain 0 for uncovered.
        # However, re-using a block is never beneficial for minimization.
        
        while uncovered:
            if time.time() > start_time + time_limit_frac * (deadline - start_time):
                break
                
            best_gain = -1
            best_candidates = []
            
            # Optimization: We can pre-filter blocks that are subsets of already used blocks?
            # No, that's complex. Just scan.
            # 346k blocks is fast in Python if we use bit_count.
            
            for b_idx in range(n_blocks):
                # If we already picked this block, gain is 0
                if used_blocks[b_idx]:
                    continue
                
                # Count how many currently uncovered triplets this block covers
                gain = (block_masks[b_idx] & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(b_idx)
            
            if best_gain == 0:
                # Should not happen if uncovered is not 0, unless we ran out of blocks
                # or there's a bug. But if we pick all blocks, we cover everything.
                break
                
            # Randomly pick from best candidates to diversify search paths
            chosen_idx = rng.choice(best_candidates)
            solution_indices.append(chosen_idx)
            used_blocks[chosen_idx] = True
            
            # Update uncovered mask
            uncovered &= ~block_masks[chosen_idx]
            
        # Convert indices to block tuples
        return [all_blocks[i] for i in solution_indices]

    def local_search_polish(blocks, seed=0, max_time=30.0):
        """
        Polishes the solution by removing redundant blocks and attempting merges.
        """
        rng = random.Random(seed)
        blocks = list(blocks)
        
        # Precompute coverage counts
        # count[tid] = number of blocks covering triplet tid
        count = np.zeros(n_triplets, dtype=np.int32)
        
        def update_counts(block, delta):
            for tid in block_triplet_indices[all_blocks.index(block)]: # This is slow lookup
                count[tid] += delta
                
        # Faster update using precomputed indices
        # Let's map block tuple to its index in all_blocks for O(1) lookup?
        # Or just recompute indices list for the specific blocks we have.
        
        # Let's maintain a list of triplet IDs for the current blocks
        block_tids_list = []
        for b in blocks:
            tids = []
            for triplet in itertools.combinations(b, t):
                tids.append(triplet_to_id[triplet])
            block_tids_list.append(tids)
            
        for tids in block_tids_list:
            for tid in tids:
                count[tid] += 1

        def is_redundant(block_idx):
            for tid in block_tids_list[block_idx]:
                if count[tid] == 1:
                    return False
            return True

        def remove_redundant():
            nonlocal blocks, block_tids_list, count
            changed = True
            while changed:
                changed = False
                new_blocks = []
                new_tids = []
                for i, b in enumerate(blocks):
                    if is_redundant(i):
                        # Remove
                        for tid in block_tids_list[i]:
                            count[tid] -= 1
                        changed = True
                    else:
                        new_blocks.append(b)
                        new_tids.append(block_tids_list[i])
                if changed:
                    blocks = new_blocks
                    block_tids_list = new_tids

        def try_merge():
            nonlocal blocks, block_tids_list, count
            n = len(blocks)
            if n < 2:
                return False
            
            # Find unique triplets for each block
            unique_tids = []
            for i in range(n):
                ut = [tid for tid in block_tids_list[i] if count[tid] == 1]
                unique_tids.append(ut)
                
            # Try to merge pairs
            # For efficiency, only try pairs where the union of unique points is <= k
            # We can precompute the set of points for unique triplets
            
            for i in range(n):
                if not unique_tids[i]:
                    continue
                points_i = set()
                for tid in unique_tids[i]:
                    points_i.update(all_triplets[tid])
                    
                if len(points_i) > k:
                    continue
                    
                for j in range(i + 1, n):
                    if not unique_tids[j]:
                        continue
                    points_j = set()
                    for tid in unique_tids[j]:
                        points_j.update(all_triplets[tid])
                        
                    if len(points_j) > k:
                        continue
                        
                    all_points = points_i | points_j
                    if len(all_points) <= k:
                        # We can merge
                        # Create new block
                        new_block_set = list(all_points)
                        # Fill with random points if necessary? 
                        # Actually, we just need a valid block containing these points.
                        # The "merge" creates a block that covers the unique parts.
                        # It might cover other things too, which is fine.
                        
                        # Let's just pick a random valid block containing all_points
                        # Or construct one greedily?
                        # Since len(all_points) <= k, we can just pick random points to fill up to k.
                        # But wait, does the new block need to be one of the precomputed `all_blocks`?
                        # Yes, for consistency with `block_masks` if we were to re-add it to the pool.
                        # But in local search, we are just modifying the solution list.
                        # We don't need to be in `all_blocks` for the solution, 
                        # but we do need to update `count` correctly.
                        
                        # Let's construct the block:
                        remaining_points = [p for p in range(v) if p not in all_points]
                        needed = k - len(all_points)
                        fill = rng.sample(remaining_points, needed) if needed > 0 else []
                        new_block = tuple(sorted(list(all_points) + fill))
                        
                        # Update counts
                        # Remove i and j
                        for tid in block_tids_list[i]:
                            count[tid] -= 1
                        for tid in block_tids_list[j]:
                            count[tid] -= 1
                            
                        # Add new block
                        new_tids = []
                        for triplet in itertools.combinations(new_block, t):
                            tid = triplet_to_id[triplet]
                            count[tid] += 1
                            new_tids.append(tid)
                            
                        # Update lists
                        blocks.pop(j) # Remove j first (higher index)
                        blocks.pop(i) # Remove i
                        blocks.append(new_block)
                        block_tids_list.pop(j)
                        block_tids_list.pop(i)
                        block_tids_list.append(new_tids)
                        
                        return True
            return False

        # Main polish loop
        polish_deadline = time.time() + max_time
        while time.time() < polish_deadline:
            remove_redundant()
            if try_merge():
                continue
            # If no merge, try some random swaps? 
            # For now, just break if no progress in merge/redundancy.
            # A more advanced local search would try point swaps.
            # Given time constraints, redundancy + merge is a strong baseline.
            # Let's add a simple point swap heuristic if stuck.
            
            # Simple point swap: pick a block, swap a point with another, 
            # see if it allows redundancy removal.
            # This is complex to implement robustly in the remaining time.
            # Let's just do a few iterations of "random block replacement" if stuck?
            # No, let's stick to the effective ones.
            
            # If we can't merge or remove, we might be in a local minimum.
            # We can try "kick" moves: randomly replace a block with a different one 
            # that covers its unique triplets, even if it's temporarily worse or neutral.
            
            # Let's implement a simple "1-for-1" replacement if stuck.
            n = len(blocks)
            if n > 0:
                # Pick a random block
                idx = rng.randrange(n)
                unique_tids_idx = [tid for tid in block_tids_list[idx] if count[tid] == 1]
                
                if unique_tids_idx:
                    # Find a new block that covers these unique triplets
                    points_needed = set()
                    for tid in unique_tids_idx:
                        points_needed.update(all_triplets[tid])
                    
                    if len(points_needed) <= k:
                        # Find a random block in all_blocks that is a superset of points_needed
                        # This is a subsumption query. 
                        # For small v, we can just scan all_blocks? 346k scan is ~0.1s.
                        # Let's do it if we have time.
                        
                        # Optimization: Precompute point mask for blocks?
                        # point_mask[b_idx] = bitmask of points in block.
                        # Then check if (block_point_mask & needed_point_mask) == needed_point_mask.
                        
                        # We didn't precompute point masks. Let's do it on the fly for this specific check?
                        # Or just skip this advanced move to save coding time/complexity.
                        # The merge and redundancy removal are the main drivers.
                        
                        pass
            
            # If we are stuck, break to save time for other seeds.
            # Actually, let's just break the polish loop if no changes in a full pass.
            # The `try_merge` returns False if no merge. `remove_redundant` changes `changed` internally.
            # We don't know if `remove_redundant` found anything in this specific call easily without refactoring.
            # Let's assume if `try_merge` fails and we just ran `remove_redundant`, 
            # and we don't have a "kick" move, we are stuck.
            
            # To avoid infinite loop if nothing changes:
            # We can track the number of blocks. If it doesn't change after a full iteration, break.
            # But `remove_redundant` might change it.
            
            # Simple safeguard: break if we've been here too long without improvement?
            # Let's just run for a fixed number of iterations or until time runs out.
            # The `while time.time() < polish_deadline` handles the time.
            # To prevent tight loop if no changes:
            # We can add a small sleep or just accept it might loop fast.
            # Better: break if no merge and no redundancy was found.
            # Let's refactor `remove_redundant` to return count of removed.
            
            # For simplicity in this combination, I will rely on the fact that 
            # `try_merge` is the expensive part. If it fails, and redundancy is clean, 
            # we are stuck. I will add a simple counter to break after N failed merge attempts.
            # Actually, let's just let it run until deadline. Python loop overhead is small.
            # But to be safe, I'll break if `len(blocks)` hasn't changed in last 50 iterations.
            pass

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Greedy Set Cover with multiple seeds
    # Use 60% of the time for this phase
    greedy_deadline = start_time + 0.6 * (deadline - start_time)
    
    num_seeds = 100
    for seed in range(num_seeds):
        if time.time() > greedy_deadline:
            break
            
        blocks = greedy_set_cover(seed, time_limit_frac=1.0) # The internal check uses global deadline logic? 
        # My greedy_set_cover uses a local time check. 
        # Let's modify it to respect the global greedy_deadline.
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_set_cover(0, time_limit_frac=1.0)

    # Phase 2: Local Search Polish
    # Use remaining time
    if time.time() < deadline - 5:
        polish_time = deadline - 5 - time.time()
        polished_blocks = local_search_polish(best_blocks, seed=42, max_time=polish_time)
        if len(polished_blocks) < best_count:
            best_blocks = polished_blocks
            best_count = len(polished_blocks)

    # Final validation and cleanup
    # Ensure no duplicates
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(b))
            
    return unique_blocks

# Helper for greedy_set_cover to respect global time better
# The current greedy_set_cover has a hardcoded time_limit_frac relative to start_time.
# I will rewrite the main loop to be cleaner.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute all t-subsets (triplets)
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the triplets it covers
    block_masks = [0] * n_blocks
    block_triplet_indices = [None] * n_blocks
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        tids = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            tids.append(tid)
        block_masks[b_idx] = mask
        block_triplet_indices[b_idx] = tids

    all_mask = (1 << n_triplets) - 1

    def greedy_set_cover(seed, time_limit):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_indices = []
        used_blocks = [False] * n_blocks
        
        while uncovered:
            if time.time() > time_limit:
                break
                
            best_gain = -1
            best_candidates = []
            
            for b_idx in range(n_blocks):
                if used_blocks[b_idx]:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(b_idx)
            
            if best_gain == 0:
                break
                
            chosen_idx = rng.choice(best_candidates)
            solution_indices.append(chosen_idx)
            used_blocks[chosen_idx] = True
            uncovered &= ~block_masks[chosen_idx]
            
        return [all_blocks[i] for i in solution_indices]

    def polish(blocks, time_limit):
        rng = random.Random(42)
        blocks = list(blocks)
        
        # Precompute tids for current blocks
        block_tids_list = []
        for b in blocks:
            tids = []
            for triplet in itertools.combinations(b, t):
                tids.append(triplet_to_id[triplet])
            block_tids_list.append(tids)
            
        count = np.zeros(n_triplets, dtype=np.int32)
        for tids in block_tids_list:
            for tid in tids:
                count[tid] += 1

        def remove_redundant():
            nonlocal blocks, block_tids_list, count
            changed = True
            while changed:
                changed = False
                new_blocks = []
                new_tids = []
                for i, b in enumerate(blocks):
                    redundant = True
                    for tid in block_tids_list[i]:
                        if count[tid] == 1:
                            redundant = False
                            break
                    if redundant:
                        for tid in block_tids_list[i]:
                            count[tid] -= 1
                        changed = True
                    else:
                        new_blocks.append(b)
                        new_tids.append(block_tids_list[i])
                if changed:
                    blocks = new_blocks
                    block_tids_list = new_tids

        def try_merge():
            nonlocal blocks, block_tids_list, count
            n = len(blocks)
            if n < 2:
                return False
            
            unique_tids = []
            for i in range(n):
                ut = [tid for tid in block_tids_list[i] if count[tid] == 1]
                unique_tids.append(ut)
                
            for i in range(n):
                if not unique_tids[i]:
                    continue
                points_i = set()
                for tid in unique_tids[i]:
                    points_i.update(all_triplets[tid])
                if len(points_i) > k:
                    continue
                    
                for j in range(i + 1, n):
                    if not unique_tids[j]:
                        continue
                    points_j = set()
                    for tid in unique_tids[j]:
                        points_j.update(all_triplets[tid])
                    if len(points_j) > k:
                        continue
                        
                    all_points = points_i | points_j
                    if len(all_points) <= k:
                        # Merge
                        remaining_points = [p for p in range(v) if p not in all_points]
                        needed = k - len(all_points)
                        fill = rng.sample(remaining_points, needed) if needed > 0 else []
                        new_block = tuple(sorted(list(all_points) + fill))
                        
                        # Update counts
                        for tid in block_tids_list[i]:
                            count[tid] -= 1
                        for tid in block_tids_list[j]:
                            count[tid] -= 1
                            
                        new_tids = []
                        for triplet in itertools.combinations(new_block, t):
                            tid = triplet_to_id[triplet]
                            count[tid] += 1
                            new_tids.append(tid)
                            
                        blocks.pop(j)
                        blocks.pop(i)
                        blocks.append(new_block)
                        block_tids_list.pop(j)
                        block_tids_list.pop(i)
                        block_tids_list.append(new_tids)
                        
                        return True
            return False

        while time.time() < time_limit:
            remove_redundant()
            if not try_merge():
                # If no merge, we might be stuck. 
                # To avoid tight loop, we can break if no changes in a while.
                # For now, just let it run, it's fast enough for 2024 triplets.
                pass
                
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Allocate 70% of time to greedy, 30% to polish
    greedy_deadline = start_time + 0.7 * (deadline - start_time)
    
    for seed in range(200):
        if time.time() > greedy_deadline:
            break
        blocks = greedy_set_cover(seed, greedy_deadline)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_set_cover(0, deadline - 5)

    if time.time() < deadline - 5:
        polish_deadline = deadline - 5
        polished = polish(best_blocks, polish_deadline)
        if len(polished) < best_count:
            best_blocks = polished

    # Deduplicate
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(b))
            
    return unique_blocks

if __name__ == "__main__":
    pass
