import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for final validation/return

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [None] * n_blocks
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # Precompute incidence matrix for fast scoring
    # M[bi, ts_idx] = 1 if block bi covers t-subset ts_idx
    M = np.zeros((n_blocks, n_tsubsets), dtype=np.int32)
    for bi in range(n_blocks):
        for ts_idx in block_tsubs[bi]:
            M[bi, ts_idx] = 1

    # Helper: Greedy Set Cover with vectorized scoring
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=np.int32)
        selected = []
        
        # Initial gains
        gains = M.sum(axis=1)
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            # Find block with max gain
            # We need to consider only blocks not yet selected, but gains for selected blocks 
            # effectively become 0 for new coverage if we track uncovered properly.
            # Actually, standard greedy just picks max gain from all, but if a block is already 
            # picked, its gain should be 0 or we mask it.
            
            # Mask selected blocks to have 0 gain? Or just track gains dynamically.
            # Dynamic update:
            # gain[bi] = number of uncovered t-subsets covered by bi.
            
            best_bi = np.argmax(gains)
            best_gain = gains[best_bi]
            
            if best_gain == 0:
                break
                
            selected.append(best_bi)
            
            # Update gains: subtract 1 from all blocks that cover the t-subsets covered by best_bi
            # covered_tsubs = block_tsubs[best_bi]
            # For each ts in covered_tsubs:
            #   if uncovered[ts]:
            #     for each block bj covering ts:
            #       gains[bj] -= 1
            #     uncovered[ts] = 0
            
            # Vectorized update:
            # We want to subtract 1 from gains[bj] for all bj such that M[bj, ts] == 1 and uncovered[ts] == 1.
            # This is equivalent to: gains -= M[:, covered_tsubs] dot uncovered?
            # No, gains[bj] = sum(M[bj, ts] * uncovered[ts] for ts in range(n_tsubsets))
            # So we can just recompute gains if it's fast enough, or update incrementally.
            # Incremental:
            # Let C be the set of t-subsets covered by best_bi that were previously uncovered.
            # For each ts in C:
            #   gains -= M[:, ts]
            #   uncovered[ts] = 0
            
            # Find which of best_bi's t-subsets are uncovered
            covered_tsubs = block_tsubs[best_bi]
            # Create a mask of uncovered t-subsets in covered_tsubs
            # We can use numpy for this
            # mask = uncovered[covered_tsubs] > 0
            # But updating gains:
            # gains -= M[:, covered_tsubs] @ (uncovered[covered_tsubs] > 0).astype(int)
            # This is O(n_blocks * len(covered_tsubs)) which is fine.
            
            mask = (uncovered[covered_tsubs] > 0).astype(np.int32)
            if np.any(mask):
                # Subtract contribution of these newly covered t-subsets from all blocks
                gains -= M[:, covered_tsubs] @ mask
                uncovered[covered_tsubs] = 0
            else:
                # This shouldn't happen if best_gain > 0, but safe
                pass
            
            # Ensure gains don't go negative (due to floating point or logic errors, though ints here)
            # gains = np.maximum(gains, 0)
            
            # Mark best_bi as selected? 
            # If we pick it again, gain should be 0 because its t-subsets are now uncovered=0.
            # So M[:, covered_tsubs] @ mask handled it for all blocks including best_bi.
            # So gains[best_bi] should now be 0.
            
        return selected

    # Helper: Local Search
    def local_search(blocks, seed):
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)

        for bi in block_set:
            cover_count[block_tsubs[bi]] += 1

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in block_set:
                # A block is redundant if all its t-subsets are covered by at least one other block
                # i.e., cover_count[ts] > 1 for all ts in block_tsubs[bi]
                if np.all(cover_count[block_tsubs[bi]] > 1):
                    to_remove.append(bi)

            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    cover_count[block_tsubs[bi]] -= 1
                improved = True
                continue

            # Phase 2: Try swaps (1-for-1)
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    # Redundant (should have been caught in Phase 1, but safe)
                    block_set.remove(bi)
                    cover_count[block_tsubs[bi]] -= 1
                    improved = True
                    break

                # Find candidates that cover all unique_tsubs
                # Candidates are blocks that cover every ts in unique_tsubs
                # We can find this by intersecting tsub_block_sets for each ts in unique_tsubs
                # But we want to avoid bi itself.
                
                # Efficient way:
                # Start with the set of blocks covering the first unique tsub
                # Intersect with others.
                
                if len(unique_tsubs) == 0:
                    continue
                    
                candidates = set(tsub_blocks[unique_tsubs[0]])
                for ts_idx in unique_tsubs[1:]:
                    candidates &= set(tsub_blocks[ts_idx])
                    if not candidates:
                        break
                
                candidates.discard(bi)
                
                if candidates:
                    # Pick a random candidate
                    new_bi = rng.choice(list(candidates))
                    
                    # Perform swap
                    block_set.remove(bi)
                    block_set.add(new_bi)
                    
                    cover_count[block_tsubs[bi]] -= 1
                    cover_count[block_tsubs[new_bi]] += 1
                    
                    improved = True
                    break # Restart loop after a swap to re-evaluate redundancy

            # Phase 3: Try 2-for-1 exchanges (remove 2, add 1)
            if not improved and len(block_set) >= 2:
                # Try a few random 2-for-1 exchanges
                max_attempts = 20
                for _ in range(max_attempts):
                    if time.time() > deadline - 2:
                        break
                    
                    b1, b2 = rng.sample(list(block_set), 2)
                    
                    # Find t-subsets covered by b1 or b2 that are covered by <= 2 blocks
                    # These are the "critical" t-subsets that would become uncovered if we remove b1 and b2
                    # Actually, we need to find a new block that covers all t-subsets that are
                    # currently covered ONLY by b1 or b2 (i.e., cover_count == 1 and in b1/b2)
                    # OR cover_count == 2 and in both b1 and b2?
                    
                    # Let's define critical_tsubs:
                    # For ts in b1: if cover_count[ts] == 1, it's critical (must be covered by new block)
                    # For ts in b2: if cover_count[ts] == 1, it's critical
                    # For ts in b1 and b2: if cover_count[ts] == 2, it's critical (would become 0 if both removed)
                    
                    critical_tsubs = set()
                    for ts_idx in block_tsubs[b1]:
                        if cover_count[ts_idx] == 1:
                            critical_tsubs.add(ts_idx)
                    for ts_idx in block_tsubs[b2]:
                        if cover_count[ts_idx] == 1:
                            critical_tsubs.add(ts_idx)
                    # Check for ts in both with count 2
                    # This is more complex. Let's stick to the simpler "unique to one of them" first.
                    # If a ts is in both and count is 2, removing both makes it 0.
                    # So we need to add those too.
                    for ts_idx in block_tsubs[b1]:
                        if ts_idx in block_tsubs[b2] and cover_count[ts_idx] == 2:
                            critical_tsubs.add(ts_idx)
                            
                    if not critical_tsubs:
                        continue
                        
                    # Find candidates covering all critical_tsubs
                    candidates = set(tsub_blocks[list(critical_tsubs)[0]])
                    for ts_idx in list(critical_tsubs)[1:]:
                        candidates &= set(tsub_blocks[ts_idx])
                        if not candidates:
                            break
                    
                    candidates.discard(b1)
                    candidates.discard(b2)
                    
                    if candidates:
                        new_bi = rng.choice(list(candidates))
                        
                        # Perform 2-for-1
                        block_set.remove(b1)
                        block_set.remove(b2)
                        block_set.add(new_bi)
                        
                        cover_count[block_tsubs[b1]] -= 1
                        cover_count[block_tsubs[b2]] -= 1
                        cover_count[block_tsubs[new_bi]] += 1
                        
                        improved = True
                        break # Restart loop

            # Phase 4: Perturbation if stuck
            if not improved:
                # Remove a random block and repair greedily
                if block_set:
                    rem_bi = rng.choice(list(block_set))
                    block_set.remove(rem_bi)
                    cover_count[block_tsubs[rem_bi]] -= 1
                    
                    # Repair
                    uncovered = np.where(cover_count == 0)[0]
                    while len(uncovered) > 0 and time.time() < deadline - 2:
                        # Find best block to add
                        gains = M[:, uncovered] @ np.ones(len(uncovered), dtype=np.int32)
                        # Mask out blocks already in set?
                        # For now, just pick max gain. If it's in set, gain might be high but it's already there.
                        # We should only consider blocks not in set.
                        # Create a mask of blocks not in set
                        not_in_set = np.ones(n_blocks, dtype=bool)
                        for bi in block_set:
                            not_in_set[bi] = False
                        gains[~not_in_set] = 0
                        
                        best_bi = np.argmax(gains)
                        if gains[best_bi] == 0:
                            break
                            
                        block_set.add(best_bi)
                        cover_count[block_tsubs[best_bi]] += 1
                        
                        # Update uncovered
                        uncovered = np.where(cover_count == 0)[0]
                        
                    # Remove redundancies again
                    to_remove = []
                    for bi in list(block_set):
                        if np.all(cover_count[block_tsubs[bi]] > 1):
                            to_remove.append(bi)
                    for bi in to_remove:
                        block_set.remove(bi)
                        cover_count[block_tsubs[bi]] -= 1

        return list(block_set)

    # Initial Solution: Greedy
    initial_blocks = greedy_cover(42)
    
    # Local Search
    best_blocks = local_search(initial_blocks, 42)
    best_size = len(best_blocks)
    
    # Try a few more restarts if time permits
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 10:
            break
        
        init = greedy_cover(seed)
        local = local_search(init, seed)
        
        if len(local) < best_size:
            best_blocks = local
            best_size = len(local)
            
        # Also try local search on the current best with different seed
        local2 = local_search(best_blocks, seed + 100)
        if len(local2) < best_size:
            best_blocks = local2
            best_size = len(local2)

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    
    # Final validation
    covered = np.zeros(n_tsubsets, dtype=np.int32)
    for block in final_blocks:
        for sub in itertools.combinations(block, t):
            idx = tsub_idx[sub]
            covered[idx] += 1
            
    if np.any(covered == 0):
        # Fallback to greedy if invalid
        fallback = greedy_cover(0)
        final_blocks = [list(all_blocks[idx]) for idx in fallback]
        
    return final_blocks
