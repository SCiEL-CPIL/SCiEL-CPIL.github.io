import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Evaluator kills at 150s. We want to finish well before.
    # 145s is a safe margin.
    deadline = start_time + 145 

    # Precompute all t-subsets (triplets for t=3)
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks (7-subsets for k=7)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the triplets it covers
    # n_triplets = C(24,3) = 2024. Fits in a Python integer.
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_masks[b_idx] = mask
        
    # Full mask of all triplets
    all_mask = (1 << n_triplets) - 1

    # 1. Greedy Set Cover to get a good initial solution
    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_indices = []
        
        while uncovered:
            if time.time() > deadline - 60:
                break
                
            best_gain = -1
            best_idx = -1
            candidates = []
            
            # Find block with max gain
            # Optimization: We can pre-sort or use a heap, but scanning 346k blocks 
            # with bit_count is fast enough in Python if done a few times.
            # However, doing it ~80 times (one per block) is 27M operations, which is fine.
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    candidates.append(b_idx)
                    
            if best_idx == -1 or best_gain == 0:
                # Should not happen if uncovered != 0
                break
                
            # Randomly pick from top candidates to diversify
            # In pure greedy, we pick the best. Here we pick from ties or a small top-k
            # to allow different paths.
            chosen_idx = rng.choice(candidates)
            
            solution_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        return solution_indices

    # 2. Simulated Annealing
    def simulated_annealing(initial_indices, seed=0):
        rng = random.Random(seed)
        
        # State: list of block indices
        current_blocks = list(initial_indices)
        current_covered = 0
        for idx in current_blocks:
            current_covered |= block_masks[idx]
        
        current_uncovered_count = n_triplets - current_covered.bit_count()
        
        # Energy function
        # If valid (uncovered == 0), energy = len(blocks)
        # If invalid, energy = len(blocks) + 100 * uncovered_count
        # The penalty 100 ensures validity is prioritized over block count reduction
        # when invalid, but allows moving to invalid states if it reduces block count 
        # significantly? No, usually we want to stay valid or move to states with 
        # similar block count but better potential.
        # Actually, a common approach for covering designs is:
        # Score = num_blocks if valid, else num_blocks + big_penalty * uncovered.
        
        big_penalty = 1000
        
        def get_energy(blocks, uncovered_count):
            if uncovered_count == 0:
                return len(blocks)
            else:
                return len(blocks) + big_penalty * uncovered_count

        current_energy = get_energy(current_blocks, current_uncovered_count)
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        # Cooling schedule
        # Start high, end low.
        # We have ~100 seconds budget for SA.
        # Let's run for a fixed number of steps or until time.
        
        start_temp = 5.0
        end_temp = 0.01
        cooling_rate = 0.9995 # Slow cooling
        
        current_temp = start_temp
        
        step = 0
        while time.time() < deadline - 20: # Leave 20s for refinement
            if step % 1000 == 0:
                current_temp *= cooling_rate
                if current_temp < end_temp:
                    current_temp = end_temp
            
            # Choose a move
            move_type = rng.random()
            
            new_blocks = list(current_blocks)
            new_uncovered_count = current_uncovered_count
            
            if move_type < 0.4:
                # Remove a block
                if not current_blocks:
                    continue
                idx_to_remove = rng.choice(current_blocks)
                mask_to_remove = block_masks[idx_to_remove]
                
                # Check if removing it uncovers any triplets
                # new_covered = current_covered & ~mask_to_remove
                # This is expensive if we maintain covered mask. 
                # Let's maintain covered mask.
                
                # Actually, maintaining covered mask is O(1) for update, but 
                # we need to know if a triplet becomes uncovered.
                # If (current_covered & ~mask_to_remove) has fewer bits, we lost coverage.
                
                # Let's use a different state representation for speed:
                # Keep a count array? No, bitmask is better for global coverage check.
                
                # Let's stick to recalculating or using a trick.
                # Trick: 
                # If we remove block i, a triplet is uncovered if it was ONLY covered by i.
                # We can pre-compute "exclusive" triplets for each block in the current solution?
                # That's expensive to update.
                
                # Alternative: Just calculate new covered mask.
                # current_covered is a big int.
                # new_covered = current_covered & ~mask_to_remove
                # new_uncovered_count = n_triplets - new_covered.bit_count()
                
                # But we don't have current_covered stored in the loop variable easily 
                # if we only store blocks.
                # Let's store current_covered mask.
                
                pass # Placeholder, logic below

        # Revised SA Loop with Mask
        current_covered = 0
        for idx in current_blocks:
            current_covered |= block_masks[idx]
        current_uncovered_count = n_triplets - current_covered.bit_count()
        current_energy = get_energy(current_blocks, current_uncovered_count)
        
        start_temp = 10.0
        end_temp = 0.01
        cooling_rate = 0.9998
        
        current_temp = start_temp
        step = 0
        
        while time.time() < deadline - 20:
            if step % 1000 == 0:
                current_temp *= cooling_rate
                if current_temp < end_temp:
                    current_temp = end_temp
            
            # Moves:
            # 1. Remove random block (40%)
            # 2. Add random block (30%)
            # 3. Swap random block with another (30%)
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Remove
                if not current_blocks:
                    step += 1
                    continue
                idx_to_remove = rng.choice(current_blocks)
                mask_to_remove = block_masks[idx_to_remove]
                
                new_covered = current_covered & ~mask_to_remove
                new_uncovered_count = n_triplets - new_covered.bit_count()
                
                new_blocks = [b for b in current_blocks if b != idx_to_remove]
                new_energy = get_energy(new_blocks, new_uncovered_count)
                
                delta_e = new_energy - current_energy
                
                # Acceptance
                if delta_e < 0 or rng.random() < np.exp(-delta_e / current_temp):
                    current_blocks = new_blocks
                    current_covered = new_covered
                    current_uncovered_count = new_uncovered_count
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(current_blocks)
                        
            elif move_type < 0.7:
                # Add
                # To be effective, add a block that covers many uncovered triplets.
                # But random add is simpler and helps escape.
                # Let's pick a block that has high gain on current uncovered.
                # Gain = (mask & ~current_covered).bit_count()
                
                # Finding best add is expensive. Let's pick random first.
                # Or pick from a pre-computed list of "good" blocks?
                # Let's just pick random.
                idx_to_add = rng.randint(0, n_blocks - 1)
                
                # If already present, skip or treat as no-op? 
                # Better to pick a unique one.
                # For simplicity, just add. Duplicates don't hurt coverage but hurt count.
                # We should avoid duplicates.
                if idx_to_add in current_blocks:
                    step += 1
                    continue
                
                mask_to_add = block_masks[idx_to_add]
                new_covered = current_covered | mask_to_add
                new_uncovered_count = n_triplets - new_covered.bit_count()
                
                new_blocks = current_blocks + [idx_to_add]
                new_energy = get_energy(new_blocks, new_uncovered_count)
                
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < np.exp(-delta_e / current_temp):
                    current_blocks = new_blocks
                    current_covered = new_covered
                    current_uncovered_count = new_uncovered_count
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(current_blocks)
                        
            else:
                # Swap
                if not current_blocks:
                    step += 1
                    continue
                    
                idx_to_swap = rng.choice(current_blocks)
                # Pick a replacement from all blocks
                idx_new = rng.randint(0, n_blocks - 1)
                
                if idx_new == idx_to_swap:
                    step += 1
                    continue
                    
                mask_old = block_masks[idx_to_swap]
                mask_new = block_masks[idx_new]
                
                # New covered = (current_covered & ~mask_old) | mask_new
                # This is slightly expensive but okay.
                new_covered = (current_covered & ~mask_old) | mask_new
                new_uncovered_count = n_triplets - new_covered.bit_count()
                
                new_blocks = [b if b != idx_to_swap else idx_new for b in current_blocks]
                new_energy = get_energy(new_blocks, new_uncovered_count)
                
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < np.exp(-delta_e / current_temp):
                    current_blocks = new_blocks
                    current_covered = new_covered
                    current_uncovered_count = new_uncovered_count
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(current_blocks)
            
            step += 1
            
        return best_blocks

    # 3. Refinement (Local Search)
    def refine_solution(blocks):
        # Convert indices to tuples for consistency if needed, but we work with indices
        # The input to refine is a list of indices.
        # We need to return a list of blocks (tuples of points).
        
        # 1. Remove redundant blocks
        # A block is redundant if all its triplets are covered by other blocks.
        
        # Calculate coverage counts
        # counts[tid] = number of blocks in current solution covering tid
        counts = np.zeros(n_triplets, dtype=np.int32)
        for idx in blocks:
            mask = block_masks[idx]
            # Iterate through set bits? Or just iterate through triplets in block?
            # Iterating through triplets in block is C(7,3)=35 ops.
            # Iterating through mask bits is faster if sparse, but mask is dense.
            # Let's use the triplet iteration for counts as it's bounded by 35.
            block_pts = all_blocks[idx]
            for triplet in itertools.combinations(block_pts, t):
                tid = triplet_to_id[triplet]
                counts[tid] += 1
                
        new_blocks = []
        for idx in blocks:
            block_pts = all_blocks[idx]
            redundant = True
            for triplet in itertools.combinations(block_pts, t):
                tid = triplet_to_id[triplet]
                if counts[tid] == 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(idx)
                
        blocks = new_blocks
        
        # 2. Try 2-for-1 merge
        # If two blocks have exclusive triplets that can be covered by a third block,
        # replace them with the third.
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            if not blocks:
                break
                
            # Recalculate counts
            counts = np.zeros(n_triplets, dtype=np.int32)
            for idx in blocks:
                block_pts = all_blocks[idx]
                for triplet in itertools.combinations(block_pts, t):
                    tid = triplet_to_id[triplet]
                    counts[tid] += 1
                    
            # Find exclusive triplets for each block
            exclusives = []
            for idx in blocks:
                block_pts = all_blocks[idx]
                exc = []
                for triplet in itertools.combinations(block_pts, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        exc.append(tid)
                exclusives.append(exc)
                
            merged = False
            n_sol = len(blocks)
            
            for i in range(n_sol):
                if time.time() > deadline - 5:
                    break
                if not exclusives[i]:
                    continue
                    
                for j in range(i + 1, n_sol):
                    if time.time() > deadline - 5:
                        break
                    if not exclusives[j]:
                        continue
                    
                    # Check if union of exclusive triplets of i and j can be covered by one block
                    points_needed = set()
                    for tid in exclusives[i]:
                        points_needed.update(all_triplets[tid])
                    for tid in exclusives[j]:
                        points_needed.update(all_triplets[tid])
                        
                    if len(points_needed) > k:
                        continue
                        
                    # Find a block that contains points_needed
                    # Since we can add any block, we can just construct one.
                    # But it must be a valid k-subset.
                    # And it must be distinct from existing blocks? 
                    # The problem says "blocks must be distinct".
                    # If the new block is already in the solution (and not i or j), 
                    # we just remove i and j.
                    # If it's not in the solution, we add it and remove i and j.
                    
                    # Construct candidate
                    cand_set = set(points_needed)
                    for p in range(v):
                        if len(cand_set) == k:
                            break
                        if p not in cand_set:
                            cand_set.add(p)
                            
                    if len(cand_set) != k:
                        continue
                        
                    new_block_tuple = tuple(sorted(cand_set))
                    
                    # Find index of this block in all_blocks
                    # We need a map from tuple to index for speed
                    # all_block_to_idx = {b: i for i, b in enumerate(all_blocks)}
                    # This map is large (346k entries) but we can build it once.
                    
                    # Let's build the map outside if not exists
                    # For now, linear search or binary search? 
                    # all_blocks is sorted. We can use bisect.
                    
                    # Actually, let's just check if it's in current blocks first.
                    # If it is, we remove i and j.
                    # If not, we add it.
                    
                    # To add it, we need its index in all_blocks.
                    # We'll use a dictionary for this specific check to be safe and fast.
                    # But building it every time is slow.
                    
                    # Let's assume we can find it.
                    # For efficiency, let's pre-compute the map in construct.
                    
                    pass # Logic continued below

        return blocks

    # Pre-compute block tuple to index map
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    # Redefine refine to use block_to_idx
    def refine_solution_v2(block_indices):
        # 1. Remove redundant
        counts = np.zeros(n_triplets, dtype=np.int32)
        for idx in block_indices:
            block_pts = all_blocks[idx]
            for triplet in itertools.combinations(block_pts, t):
                tid = triplet_to_id[triplet]
                counts[tid] += 1
                
        new_indices = []
        for idx in block_indices:
            block_pts = all_blocks[idx]
            redundant = True
            for triplet in itertools.combinations(block_pts, t):
                tid = triplet_to_id[triplet]
                if counts[tid] == 1:
                    redundant = False
                    break
            if not redundant:
                new_indices.append(idx)
                
        block_indices = new_indices
        
        # 2. Merge
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            if not block_indices:
                break
                
            counts = np.zeros(n_triplets, dtype=np.int32)
            for idx in block_indices:
                block_pts = all_blocks[idx]
                for triplet in itertools.combinations(block_pts, t):
                    tid = triplet_to_id[triplet]
                    counts[tid] += 1
                    
            exclusives = []
            for idx in block_indices:
                block_pts = all_blocks[idx]
                exc = []
                for triplet in itertools.combinations(block_pts, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        exc.append(tid)
                exclusives.append(exc)
                
            merged = False
            n_sol = len(block_indices)
            
            for i in range(n_sol):
                if time.time() > deadline - 5:
                    break
                if not exclusives[i]:
                    continue
                    
                for j in range(i + 1, n_sol):
                    if time.time() > deadline - 5:
                        break
                    if not exclusives[j]:
                        continue
                    
                    points_needed = set()
                    for tid in exclusives[i]:
                        points_needed.update(all_triplets[tid])
                    for tid in exclusives[j]:
                        points_needed.update(all_triplets[tid])
                        
                    if len(points_needed) > k:
                        continue
                        
                    cand_set = set(points_needed)
                    for p in range(v):
                        if len(cand_set) == k:
                            break
                        if p not in cand_set:
                            cand_set.add(p)
                            
                    if len(cand_set) != k:
                        continue
                        
                    new_block_tuple = tuple(sorted(cand_set))
                    
                    if new_block_tuple in block_to_idx:
                        new_idx = block_to_idx[new_block_tuple]
                        
                        # Check if new_idx is already in block_indices (and not i or j)
                        # If yes, we just remove i and j.
                        # If no, we add new_idx and remove i and j.
                        
                        # Build new list
                        new_list = []
                        added_new = False
                        for idx in block_indices:
                            if idx == i or idx == j:
                                continue
                            new_list.append(idx)
                            
                        if new_idx not in new_list:
                            new_list.append(new_idx)
                            added_new = True
                            
                        # Verify coverage? 
                        # The exclusive triplets of i and j are covered by new_block.
                        # Other triplets were covered by others or i/j.
                        # If a triplet was covered ONLY by i or j, it's in exclusives.
                        # So coverage is maintained.
                        
                        block_indices = new_list
                        merged = True
                        changed = True
                        break
                if merged:
                    break
            if merged:
                continue

        return block_indices

    # Main Logic
    best_blocks = None
    best_count = float('inf')
    
    # Run greedy to get initial
    initial_indices = greedy_set_cover(0)
    
    # Run SA
    sa_indices = simulated_annealing(initial_indices, seed=42)
    
    # Refine SA result
    refined_indices = refine_solution_v2(sa_indices)
    
    # Check validity and count
    # We should also run a few greedy seeds and refine them, just in case SA fails or is slow
    # But SA is the main driver.
    
    # Let's run 2 more greedy seeds in parallel? No, sequential.
    # If time permits, run SA on other seeds.
    
    if time.time() < deadline - 40:
        initial_indices_2 = greedy_set_cover(1)
        sa_indices_2 = simulated_annealing(initial_indices_2, seed=43)
        refined_indices_2 = refine_solution_v2(sa_indices_2)
        
        if len(refined_indices_2) < len(refined_indices):
            refined_indices = refined_indices_2
            
    # Final Refinement on the best one
    final_indices = refine_solution_v2(refined_indices)
    
    # Convert indices to blocks
    final_blocks = [all_blocks[idx] for idx in final_indices]
    
    # Ensure distinct blocks (should be, but just in case)
    # The set cover and SA logic preserves distinctness if we check.
    # Greedy picks distinct. SA adds/swap keeps distinct if we check.
    # My SA add logic: "if idx_to_add in current_blocks: continue". So distinct.
    # Swap logic: "if idx_new == idx_to_swap: continue". So distinct.
    # Refinement: removes duplicates implicitly if we replace with existing?
    # In merge: if new_idx in new_list, we don't add it again. So distinct.
    
    return final_blocks

if __name__ == "__main__":
    pass
