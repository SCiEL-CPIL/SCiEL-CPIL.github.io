import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Data structures
    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Bitsets for fast set operations
    tsub_mask = [0] * n_blocks       # tsub_mask[b] has bits for tsubs covered by block b
    tsub_blocks_mask = [0] * n_tsubsets # tsub_blocks_mask[ts] has bits for blocks covering ts
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask

    # Precompute list of blocks for each tsub for fast access
    for i in range(n_tsubsets):
        m = tsub_blocks_mask[i]
        blocks_list = []
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocks_list.append(idx)
            m ^= lsb
        tsub_blocks[i] = blocks_list

    full_tsub_mask = (1 << n_tsubsets) - 1
    
    def greedy_cover(seed):
        """
        Standard greedy using heap for gain calculation.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        
        # Initial gains: number of tsubs covered by each block
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()
            
        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break

            best_block = -1
            best_gain = 0

            # Pop stale entries
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and not (selected_mask & (1 << b_idx)):
                    best_block = b_idx
                    best_gain = g
                    break
                # Stale or already selected, discard

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1 # Mark as selected
            
            # Update gains for blocks covering newly covered tsubs
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                # For every block covering this ts_idx, decrease its gain
                for b_idx2 in tsub_blocks[ts_idx]:
                    if not (selected_mask & (1 << b_idx2)):
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], b_idx2))
                m ^= lsb

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        # Count coverage for each tsub
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue

                # Find tsubs covered only by b
                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    # b is redundant? Should have been caught in Phase 1, but just in case
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidates that cover all unique_tsubs
                # Start with the most constrained unique tsub (fewest covering blocks)
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts_idx])

                if not candidates:
                    continue

                candidates.discard(b)
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                # Limit candidates to avoid too much work per swap attempt
                if len(cand_list) > 20:
                    cand_list = cand_list[:20]

                swapped = False
                for r in cand_list:
                    if r in block_set:
                        continue

                    # Check if swapping b with r improves coverage
                    # We need to know if r covers any tsub that b doesn't, and if removing b leaves any tsub uncovered
                    # Since r covers all unique_tsubs of b, removing b won't leave them uncovered.
                    # Does r cover any NEW tsub?
                    # If yes, it's a valid swap (might even be an improvement in "efficiency" though count stays same)
                    # Wait, the goal is to minimize number of blocks.
                    # A swap keeps the number of blocks constant.
                    # So why swap?
                    # To potentially enable a future removal?
                    # Or maybe I missed something.
                    # In set cover, swapping doesn't reduce size.
                    # However, it can change the structure to allow a deletion later.
                    # Let's just do it if it's "better" in some heuristic sense?
                    # Or maybe we should just try it.
                    
                    # Actually, a swap is only useful if it allows a subsequent deletion.
                    # Let's just perform the swap and let the next Phase 1 handle deletions.
                    
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    
                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break
                
                if swapped:
                    break

        return list(block_set)

    # 4. Main Loop with Restarts
    best_blocks = []
    best_len = float('inf')
    
    # Initial greedy
    init_blocks = greedy_cover(1)
    best_blocks = init_blocks
    best_len = len(init_blocks)
    
    # Run local search on init
    if time.time() < deadline - 10.0:
        refined = local_search(init_blocks, 1)
        if len(refined) < best_len:
            best_blocks = refined
            best_len = len(refined)
            
    # Try restarts with random perturbations
    seed = 2
    while time.time() < deadline - 5.0:
        # Perturb best solution
        perturbed = list(best_blocks)
        # Randomly remove a few blocks
        n_remove = min(3, len(perturbed))
        for _ in range(n_remove):
            if perturbed:
                perturbed.pop(random.randint(0, len(perturbed)-1))
        
        # Run local search
        refined = local_search(perturbed, seed)
        if len(refined) < best_len:
            best_blocks = refined
            best_len = len(refined)
        seed += 1

    # Convert indices back to blocks
    return [list(all_blocks[b]) for b in best_blocks]
