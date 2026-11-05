import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute triplets for all possible blocks? 
    # C(24, 7) = 346,104 blocks. 
    # Each block has C(7,3) = 35 triplets.
    # Total triplet-block pairs = 346,104 * 35 ~ 12M. 
    # Storing this as a list of lists might be memory heavy but doable.
    # Or we can compute on the fly. 
    # On the fly is C(7,3)=35 lookups. Fast.
    
    def get_triplets(block):
        return [triplet_to_id[tup] for tup in itertools.combinations(block, t)]

    # Greedy Construction
    def greedy_construct(seed=0):
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_list = list(range(n_triplets))
        pos = np.arange(n_triplets)
        blocks = []
        
        while uncovered_list:
            if time.time() > deadline - 20:
                break
                
            # Pick random uncovered
            idx = rng.randrange(len(uncovered_list))
            start_tid = uncovered_list[idx]
            start_triplet = all_triplets[start_tid]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                
                # Precompute pairs in current block for gain calculation
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = triplet_to_id[triplet]
                        if is_uncovered[tid]:
                            gain += 1
                            
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p
                        if max_gain == len(pairs_in_block):
                            break
                            
                if best_point == -1:
                    best_point = rng.choice(candidates)
                elif max_gain == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    cur_idx = pos[tid]
                    last_idx = len(uncovered_list) - 1
                    if cur_idx != last_idx:
                        last_tid = uncovered_list[last_idx]
                        uncovered_list[cur_idx] = last_tid
                        pos[last_tid] = cur_idx
                    uncovered_list.pop()
                    pos[tid] = -1

        return blocks

    # Local Search: Remove Redundant, Merge, Swap
    def local_search(blocks, seed=0):
        rng = random.Random(seed)
        blocks = list(blocks)
        
        # Count coverage
        count = np.zeros(n_triplets, dtype=np.int32)
        
        def update_counts(blks, delta):
            for b in blks:
                for tid in get_triplets(b):
                    count[tid] += delta

        update_counts(blocks, 1)
        
        def is_redundant(block):
            for tid in get_triplets(block):
                if count[tid] == 1:
                    return False
            return True

        def try_remove_redundant():
            removed = False
            new_blocks = []
            for b in blocks:
                if is_redundant(b):
                    update_counts([b], -1)
                    removed = True
                else:
                    new_blocks.append(b)
            return new_blocks, removed

        def try_merge():
            n = len(blocks)
            if n < 2:
                return blocks, False

            # Try to merge any pair
            # We check if the union of unique triplets of two blocks fits in one block
            # To speed up, we can precompute unique triplets for each block
            block_unique = []
            for b in blocks:
                unique = [tid for tid in get_triplets(b) if count[tid] == 1]
                block_unique.append(unique)

            for i in range(n):
                if time.time() > deadline - 5:
                    return blocks, False
                if not block_unique[i]:
                    continue
                for j in range(i + 1, n):
                    if not block_unique[j]:
                        continue
                    
                    all_unique = set(block_unique[i]) | set(block_unique[j])
                    points_needed = set()
                    for tid in all_unique:
                        points_needed.update(all_triplets[tid])
                    
                    if len(points_needed) <= k:
                        new_block = list(points_needed)
                        while len(new_block) < k:
                            p = rng.randint(0, v - 1)
                            if p not in new_block:
                                new_block.append(p)
                        new_block.sort()
                        
                        # Check if new_block covers all unique triplets
                        # It does by construction, but we should ensure it's a valid block (distinct points)
                        if len(set(new_block)) == k:
                            update_counts([blocks[i], blocks[j]], -1)
                            update_counts([new_block], 1)
                            new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                            new_blocks.append(new_block)
                            return new_blocks, True

            return blocks, False

        def try_swap_to_enable_removal():
            """
            For each block B, check if its unique triplets can be covered by a different block B'.
            If so, replace B with B'. Then check for redundancies.
            """
            n = len(blocks)
            # Precompute unique triplets
            block_unique = []
            for b in blocks:
                unique = [tid for tid in get_triplets(b) if count[tid] == 1]
                block_unique.append(unique)

            for i in range(n):
                if time.time() > deadline - 5:
                    return blocks, False
                if not block_unique[i]:
                    continue
                    
                # Get points needed to cover unique triplets of block i
                points_needed = set()
                for tid in block_unique[i]:
                    points_needed.update(all_triplets[tid])
                
                if len(points_needed) > k:
                    continue
                
                # Try to form a new block from points_needed
                # If size < k, fill with random points
                # We want a block B' != blocks[i]
                # To avoid cycling, we can try a few variations
                
                base_points = list(points_needed)
                # Try 5 random completions
                for _ in range(5):
                    new_block = base_points[:]
                    while len(new_block) < k:
                        p = rng.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    if new_block == blocks[i]:
                        continue
                    
                    # Check if new_block covers all unique triplets
                    # It should, as it contains points_needed
                    # But let's be safe: check if it covers the specific unique tids
                    new_tids = set(get_triplets(new_block))
                    if not set(block_unique[i]).issubset(new_tids):
                        continue
                    
                    # Perform swap
                    update_counts([blocks[i]], -1)
                    update_counts([new_block], 1)
                    
                    # Check if this swap made any block redundant
                    # We just swapped, so we check the new state
                    # We return the new list and let the main loop handle redundancy
                    new_blocks = blocks[:]
                    new_blocks[i] = new_block
                    return new_blocks, True

            return blocks, False

        progress_made = True
        iterations = 0
        while progress_made and time.time() < deadline - 10:
            progress_made = False
            iterations += 1
            
            # 1. Remove redundant
            blocks, removed = try_remove_redundant()
            if removed:
                progress_made = True
                continue

            # 2. Try merge
            blocks, merged = try_merge()
            if merged:
                progress_made = True
                continue

            # 3. Try swap
            blocks, swapped = try_swap_to_enable_removal()
            if swapped:
                progress_made = True
                continue

            # If no progress, we might be stuck.
            # We could try random perturbations, but for now, break.
            # Or try swap again with different random seed?
            # The swap function uses rng, so it's already random.
            
            # If stuck, break to save time for other seeds
            if iterations > 50:
                break

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run many greedy seeds
    num_seeds = 300
    candidates = []
    
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        blocks = greedy_construct(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            candidates = [blocks]
        elif len(blocks) == best_count:
            candidates.append(blocks)
        elif len(blocks) <= best_count + 2:
            # Keep close candidates for local search
            if len(candidates) < 20:
                candidates.append(blocks)
            else:
                # Replace worst candidate if this is better than the worst
                # Sort candidates by length?
                pass

    # Ensure we have at least one candidate
    if not candidates:
        candidates = [greedy_construct(0)]

    # Run local search on top candidates
    # Sort candidates by length
    candidates.sort(key=lambda x: len(x))
    
    # Take top 10
    top_candidates = candidates[:10]
    
    for i, blocks in enumerate(top_candidates):
        if time.time() > deadline - 5:
            break
        ls_blocks = local_search(blocks, seed=1000 + i)
        if len(ls_blocks) < best_count:
            best_count = len(ls_blocks)
            best_blocks = ls_blocks

    return best_blocks

if __name__ == "__main__":
    pass
