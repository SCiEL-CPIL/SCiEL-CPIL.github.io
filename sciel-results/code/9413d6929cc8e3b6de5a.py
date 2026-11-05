import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit

    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)

    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Map block index to list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[triple] for triple in itertools.combinations(b, t)]
        block_triples.append(ts)

    # Map triple index to list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)

    def greedy_randomized(seed):
        random.seed(seed)
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)

        selected_blocks = []

        while is_uncovered.any():
            max_gain = np.max(gain)
            if max_gain == 0:
                break

            candidates = np.where(gain == max_gain)[0]
            b_idx = random.choice(candidates.tolist())

            selected_blocks.append(b_idx)

            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        """
        Removes redundant blocks from the cover.
        A block is redundant if all triples it covers are covered by at least one other block in the cover.
        """
        if not selected_blocks:
            return []

        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        active = [True] * len(selected_blocks)
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True

        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

    def local_search_2for1(cover, seed, time_budget=10.0):
        """
        Local search using 2-for-1 swaps.
        Tries to remove two blocks and replace them with one block that covers all uniquely covered triples.
        Also handles 2-for-0 (removing two redundant blocks).
        """
        random.seed(seed)
        if len(cover) < 2:
            return cover

        cover = list(cover)
        cover_set = set(cover)
        
        # Initial count
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > timeout - time_budget:
                break

            # Shuffle order to avoid bias
            indices = list(range(len(cover)))
            random.shuffle(indices)
            
            # Iterate over pairs
            # To be efficient, we can break early if no improvement in a full pass
            found_swap = False
            for i in range(len(indices)):
                idx_a = indices[i]
                b_a = cover[idx_a]
                
                # Skip if already removed in this pass? 
                # We are modifying 'cover' in place, so indices shift.
                # It's safer to work with a list and pop/append, but that's O(N).
                # Given N~80, O(N^3) is fine.
                
                for j in range(i + 1, len(indices)):
                    idx_b = indices[j]
                    # Need to be careful if we modified the list.
                    # Let's just restart the pass if we make a change to keep logic simple and correct.
                    # Or use a "active" mask.
                    pass
            
            # Simpler approach for 2-for-1:
            # Pick a random pair, check, if success, restart pass.
            # This is stochastic local search.
            
            # Let's try a systematic but randomized approach:
            # Try up to K random pairs per pass.
            num_pairs_to_try = 100
            for _ in range(num_pairs_to_try):
                if time.time() - start_time > timeout - time_budget:
                    break
                
                if len(cover) < 2:
                    break
                
                idx_a = random.randint(0, len(cover) - 1)
                idx_b = random.randint(0, len(cover) - 1)
                
                if idx_a == idx_b:
                    continue
                
                b_a = cover[idx_a]
                b_b = cover[idx_b]
                
                # Temporarily remove b_a and b_b
                # Update counts
                for t_idx in block_triples[b_a]:
                    cover_count[t_idx] -= 1
                for t_idx in block_triples[b_b]:
                    cover_count[t_idx] -= 1
                    
                # Find uncovered triples
                uncovered = []
                for t_idx in set(block_triples[b_a] + block_triples[b_b]):
                    if cover_count[t_idx] == 0:
                        uncovered.append(t_idx)
                        
                if len(uncovered) == 0:
                    # 2-for-0 swap: remove both
                    # Remove from list (order matters)
                    # Remove higher index first to keep indices valid
                    if idx_a > idx_b:
                        cover.pop(idx_a)
                        cover.pop(idx_b)
                    else:
                        cover.pop(idx_b)
                        cover.pop(idx_a)
                    cover_set.discard(b_a)
                    cover_set.discard(b_b)
                    improved = True
                    found_swap = True
                    break
                else:
                    # 2-for-1 swap: find a block C that covers all 'uncovered'
                    # Candidates must be in triple_blocks[uncovered[0]]
                    candidates = set(triple_blocks[uncovered[0]])
                    candidates.difference_update(cover_set)
                    
                    valid_candidate = -1
                    for c_idx in candidates:
                        # Check if c_idx covers all other uncovered triples
                        c_triples = set(block_triples[c_idx])
                        # Optimization: if len(uncovered) is large, this is expensive.
                        # But uncovered is usually small.
                        if all(t in c_triples for t in uncovered):
                            valid_candidate = c_idx
                            break
                            
                    if valid_candidate != -1:
                        # Perform swap
                        # Remove b_a, b_b
                        if idx_a > idx_b:
                            cover.pop(idx_a)
                            cover.pop(idx_b)
                        else:
                            cover.pop(idx_b)
                            cover.pop(idx_a)
                        cover_set.discard(b_a)
                        cover_set.discard(b_b)
                        
                        # Add valid_candidate
                        cover.append(valid_candidate)
                        cover_set.add(valid_candidate)
                        
                        # Update counts for new block
                        for t_idx in block_triples[valid_candidate]:
                            cover_count[t_idx] += 1
                            
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Restore counts
                        for t_idx in block_triples[b_a]:
                            cover_count[t_idx] += 1
                        for t_idx in block_triples[b_b]:
                            cover_count[t_idx] += 1

            if found_swap:
                # Restart the while loop to allow further swaps
                continue
            else:
                # No swap found in this pass
                break

        return cover

    best_cover = None
    seed = 0
    num_restarts = 0

    while True:
        if time.time() - start_time > timeout:
            break

        # Run greedy
        cover = greedy_randomized(seed)
        # Reduce single redundancies
        cover = reduce_cover(cover)
        
        # Apply 2-for-1 local search
        # Use a smaller time budget for local search if we have many restarts
        ls_time = 5.0
        if num_restarts < 5:
            ls_time = 10.0
            
        ls_cover = local_search_2for1(cover, seed + 1000, time_budget=ls_time)
        
        if best_cover is None or len(ls_cover) < len(best_cover):
            best_cover = ls_cover
            # If we improved significantly, maybe run another pass of local search on the best
            if time.time() - start_time < timeout - 20:
                # Run one more pass on the best
                ls_cover2 = local_search_2for1(best_cover, seed + 2000, time_budget=10.0)
                if len(ls_cover2) < len(best_cover):
                    best_cover = ls_cover2

        seed += 1
        num_restarts += 1
        
        # Stop if we've done enough restarts or time is running low
        if num_restarts >= 20:
            break
        if time.time() - start_time > timeout - 15:
            break

    # Final polish on the best cover
    if best_cover is not None and time.time() - start_time < timeout - 5:
        final_ls = local_search_2for1(best_cover, seed + 3000, time_budget=5.0)
        if len(final_ls) < len(best_cover):
            best_cover = final_ls

    # Convert indices back to blocks
    if best_cover is None:
        cover = greedy_randomized(0)
        return [list(all_blocks[b_idx]) for b_idx in cover]

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
