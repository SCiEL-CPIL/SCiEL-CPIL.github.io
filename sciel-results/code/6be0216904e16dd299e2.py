import itertools
import time
import random
import heapq
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    # If parameters are not the specific task, use a simple fallback
    if v != 19 or k != 7 or t != 4:
        return _simple_greedy(v, k, t, deadline)

    # Precomputation for v=19, k=7, t=4
    # Points: 0..18
    points = list(range(v))
    
    # Generate all possible k-blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Generate all possible t-subsets
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Incidence: block_idx -> list of tsubset_indices
    block_tsubs = [[] for _ in range(n_blocks)]
    # Incidence: tsubset_idx -> list of block_indices
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        # Calculate t-subsets for this block
        # itertools.combinations(block, t) generates tuples
        # We need to map them to indices
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            tsubs.append(idx)
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            
    # Convert tsub_blocks to sets for faster intersection during swaps
    tsub_block_sets = [set(lst) for lst in tsub_blocks]

    def greedy_cover(seed):
        """
        Greedy set cover using a max-heap.
        Returns list of block indices.
        """
        rng = random.Random(seed)
        # uncovered[ts_idx] = True if not covered
        uncovered = [True] * n_tsubsets
        # gain[bi] = number of currently uncovered t-subsets covered by block bi
        gain = [len(block_tsubs[bi]) for bi in range(n_blocks)]
        
        # Max-heap: (-gain, bi)
        heap = [(-gain[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Pop until we find a valid (non-stale) entry with positive gain
            best_bi = -1
            max_gain = 0
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                # Check if this gain is still valid (matches current gain array)
                if g == gain[bi] and g > 0:
                    best_bi = bi
                    max_gain = g
                    break
                # Otherwise, it's a stale entry, discard
            else:
                break # Heap empty or all gains 0
                
            if max_gain <= 0:
                break
                
            selected.append(best_bi)
            
            # Update gains for all blocks that share t-subsets with best_bi
            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrement gain for all blocks covering this ts_idx
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        # Push updated gain into heap
                        heapq.heappush(heap, (-gain[bj], bj))
                        
            # Mark best_bi as used (gain -1 to prevent re-selection)
            gain[best_bi] = -1
            heapq.heappush(heap, (-gain[best_bi], best_bi))
            
        return selected

    def local_search(blocks, seed):
        """
        Local search: Remove redundant blocks and perform swaps to enable further removals.
        blocks: list of block indices
        Returns: list of block indices
        """
        rng = random.Random(seed)
        # Work with a set for O(1) lookup and removal
        block_set = set(blocks)
        
        # cover_count[ts_idx] = number of blocks in block_set covering ts_idx
        cover_count = [0] * n_tsubsets
        
        # Initialize cover_count
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        max_iterations = 1000
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            if time.time() > deadline - 2:
                break
            
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue # Restart loop as structure changed
            
            # Phase 2: Try swaps
            # We look for a block B in the set and a block B' not in the set
            # such that swapping B for B' keeps the cover valid.
            # This doesn't reduce size immediately, but might allow a removal in the next iteration.
            
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            swap_found = False
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                # Find unique t-subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # This block is redundant (should have been caught in Phase 1)
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart local search loop
                    
                # Find candidate blocks that cover ALL unique_tsubs
                # Start with the rarest unique tsubset to minimize candidates
                # Sort unique_tsubs by the number of blocks covering them
                unique_tsubs.sort(key=lambda x: len(tsub_blocks[x]))
                
                # Initialize candidates with the blocks covering the first (rarest) unique tsubset
                candidates = set(tsub_block_sets[unique_tsubs[0]])
                
                # Intersect with blocks covering other unique tsubsets
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(tsub_block_sets[ts_idx])
                    
                # Remove current block from candidates
                candidates.discard(bi)
                
                if not candidates:
                    continue
                
                # Try candidates - limit to a reasonable number to avoid long stalls
                # We prefer candidates that cover many other t-subsets already in the set?
                # Or just pick a random one.
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bj in cand_list[:20]: # Limit candidates to check
                    if bj in block_set:
                        continue
                    
                    # Perform swap
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart local search loop to check for removals
                    
                if swapped:
                    break
                    
            if not improved:
                break
                
        return list(block_set)

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy instances
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        block_indices = greedy_cover(seed)
        
        # Post-process with local search
        block_indices = local_search(block_indices, seed)
        
        # Check validity of the final set
        # (Local search should maintain validity, but double check)
        cov_count = [0] * n_tsubsets
        for b_idx in block_indices:
            for idx in block_tsubs[b_idx]:
                cov_count[idx] += 1
                
        if any(c == 0 for c in cov_count):
            continue # Invalid
            
        if len(block_indices) < best_size:
            best_size = len(block_indices)
            best_blocks = [list(all_blocks[i]) for i in block_indices]
            
    if best_blocks:
        return best_blocks
    else:
        # Fallback if all failed (should not happen)
        return _simple_greedy(v, k, t, deadline)

def _simple_greedy(v, k, t, deadline):
    """
    Fallback greedy for general v, k, t.
    """
    start_time = time.time()
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_block = None
        max_covered = 0
        
        # Generate candidates. For large v, this is expensive.
        # For v=19, we can iterate all blocks.
        # But this is a fallback.
        
        # Pick a random uncovered t-subset to guide candidate generation?
        # Or just iterate all blocks if v is small.
        
        if v <= 20:
            for block in itertools.combinations(range(v), k):
                covered = 0
                block_set = set(block)
                for ts in itertools.combinations(block, t):
                    if ts in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_block = block
                    
            if best_block:
                blocks.append(list(best_block))
                block_set = set(best_block)
                for ts in itertools.combinations(best_block, t):
                    if ts in uncovered:
                        uncovered.remove(ts)
        else:
            # Heuristic for larger v
            # Pick random uncovered t-subset
            target = random.choice(list(uncovered))
            block = list(target)
            # Complete greedily
            while len(block) < k:
                best_x = None
                best_gain = 0
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is not None:
                    block.append(best_x)
                else:
                    break
            if len(block) == k:
                blocks.append(sorted(block))
                block_set = set(block)
                for ts in itertools.combinations(best_block, t):
                    if ts in uncovered:
                        uncovered.remove(ts)
                        
    return blocks
