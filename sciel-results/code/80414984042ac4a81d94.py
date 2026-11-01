import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))

    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    # This is a large matrix: C(19,7) x C(19,4) = 50388 x 3876 ~ 195M entries.
    # uint8 is 195MB. This is feasible in memory.
    # However, building it is slow. Let's try to be smart.
    # Actually, 195MB is fine.
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    # Optimization: Precompute t-subsets for each block?
    # Or iterate blocks and set bits.
    # 50388 blocks * C(7,4)=35 t-subsets = 1.7M operations. Very fast.
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # Precompute inverse: for each t-subset, which blocks cover it?
    # This is useful for local search (finding replacements).
    # tsub_blocks[j] = list of block indices i such that Cover[i,j] == 1
    # Since each t-subset is contained in C(v-t, k-t) = C(15, 3) = 455 blocks.
    # 3876 * 455 = 1.7M entries.
    tsub_blocks = [[] for _ in range(n_tsubs)]
    # Iterate over the matrix? No, iterate over blocks again is faster for building lists.
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            tsub_blocks[j].append(i)
    
    # Convert to numpy arrays for fast indexing? Or keep as lists?
    # Lists are fine for iteration, but numpy is better for vectorized ops.
    # Let's keep lists for now, or convert to np.array if needed for fast slicing.
    # Actually, for "find candidates covering a set of tsubs", we need intersection.
    # Let's precompute sets of block indices for each t-subset for fast intersection?
    # tsub_block_sets = [set(lst) for lst in tsub_blocks]
    # Intersection of sets is fast.

    # Greedy construction
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_indices = []
    
    # Precompute gains? No, dynamic.
    # gains[i] = number of uncovered tsubs covered by block i.
    gains = np.zeros(n_blocks, dtype=np.int32)
    # Initial gains: each block covers C(k,t) tsubs.
    # But only if they are uncovered. Initially all are uncovered.
    gains[:] = len(list(itertools.combinations(all_kblocks[0], t))) # C(7,4)=35
    
    # Wait, this is only true if all tsubs in a block are uncovered.
    # As we add blocks, some tsubs become covered.
    # We need to update gains efficiently.
    
    # Alternative: Just compute gains on the fly using matrix multiplication?
    # gains = Cover @ uncovered.
    # Cover is 50388 x 3876. uncovered is 3876 x 1.
    # Matrix-vector product.
    # NumPy is optimized for this.
    # Let's use that. It's O(N*M) per step. 50k * 4k = 200M ops.
    # 150 blocks * 200M = 30B ops. Might be too slow in Python/NumPy loop?
    # NumPy C loops are fast. 30B ops / 10 GFLOPS ~ 3 seconds. Should be fine.
    
    remaining_time_limit = deadline - 30.0  # Leave 30s for ILS
    
    while np.sum(uncovered) > 0:
        if time.time() > remaining_time_limit:
            break
        gains = Cover @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)

    best_solution = selected_indices.copy()
    best_size = len(best_solution)

    # Local Optimizer
    # Strategy: 
    # 1. Remove redundant blocks.
    # 2. Swap blocks to reduce "criticality" (number of unique tsubs covered) or enable removal.
    
    def local_optimize(sol_indices, time_limit):
        sol = sol_indices.copy()
        if not sol:
            return sol
            
        # Build cover counts
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in sol:
            cover_counts += Cover[idx]
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            new_sol = []
            
            # Pass 1: Remove redundant blocks
            for idx in sol:
                # Block is redundant if it doesn't cover any tsub with count 1
                # i.e., for all j where Cover[idx,j]==1, cover_counts[j] > 1
                # Equivalent: sum(Cover[idx] * (cover_counts==1)) == 0
                if not np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    cover_counts -= Cover[idx]
                    improved = True
                else:
                    new_sol.append(idx)
            
            sol = new_sol
            if time.time() > time_limit:
                break

            # Pass 2: Swap to improve structure
            # We want to replace blocks that have "many" unique tsubs with blocks that have "fewer"
            # or blocks that allow further reductions.
            # A good heuristic: pick a block with the MAX number of unique tsubs (most critical).
            # Try to replace it with a block that covers all its unique tsubs but has fewer total unique tsubs (or is "better" in some way).
            
            # Calculate unique tsub counts for each block in sol
            # unique_count[i] = sum(Cover[i] * (cover_counts==1))
            # But cover_counts changes as we remove/swap.
            # Let's just iterate and try to swap the "worst" block.
            
            # To avoid O(N^2) search for candidates, use the tsub_blocks inverse index.
            # For a block B, its unique tsubs are U_B.
            # Candidates must cover all tsubs in U_B.
            # Candidates = intersection of tsub_blocks[j] for j in U_B.
            
            # Let's precompute tsub_block_sets for fast intersection.
            # This is a one-time cost outside the loop? 
            # tsub_block_sets = [set(lst) for lst in tsub_blocks]
            # This takes memory but speeds up search.
            # 3876 sets of ~455 ints. Small.
            tsub_block_sets = [set(lst) for lst in tsub_blocks]
            
            # Sort sol by "criticality" (unique count) descending?
            # Or just pick the first one we can improve?
            
            # Let's try to improve the block with the highest unique count.
            # Calculate unique counts
            unique_counts = np.array([np.sum((Cover[idx] == 1) & (cover_counts == 1)) for idx in sol])
            
            if len(unique_counts) == 0:
                break
                
            # Find index in sol with max unique count
            max_idx_pos = np.argmax(unique_counts)
            max_unique_count = unique_counts[max_idx_pos]
            
            if max_unique_count == 0:
                # Should have been removed in Pass 1
                continue
                
            idx_to_swap = sol[max_idx_pos]
            
            # Find unique tsubs for this block
            unique_mask = (Cover[idx_to_swap] == 1) & (cover_counts == 1)
            unique_tsub_indices = np.where(unique_mask)[0]
            
            if len(unique_tsub_indices) == 0:
                continue
                
            # Find candidates
            # Intersection of tsub_block_sets[j] for j in unique_tsub_indices
            candidates = set(tsub_block_sets[unique_tsub_indices[0]])
            for j in unique_tsub_indices[1:]:
                candidates &= tsub_block_sets[j]
                if not candidates:
                    break
            
            # Exclude current block and other selected blocks?
            # If we swap with a selected block, it's just a swap.
            # But we want to replace idx_to_swap with a DIFFERENT block.
            # So exclude idx_to_swap.
            candidates.discard(idx_to_swap)
            
            if not candidates:
                continue
                
            # Score candidates
            # We want a candidate that:
            # 1. Covers all unique tsubs (guaranteed by intersection)
            # 2. Minimizes the number of NEW unique tsubs it introduces?
            #    Or maximizes the number of tsubs it covers that are currently count 2?
            #    (Covering count 2 tsubs makes them count 3, which is worse for removal? No, count 2 is good for removal. Count 1 is bad.)
            #    Actually, we want to minimize the number of tsubs that become count 1 after the swap.
            
            # Let's evaluate a few candidates.
            # For a candidate C:
            # New cover counts = cover_counts - Cover[idx_to_swap] + Cover[C]
            # New unique tsubs for C = sum((Cover[C]==1) & (NewCoverCounts==1))
            # We want to minimize this.
            
            # To speed up, pick a random subset of candidates if too many.
            if len(candidates) > 20:
                candidates = random.sample(list(candidates), 20)
                
            best_candidate = None
            best_score = max_unique_count + 1 # Current unique count + 1 is the threshold to beat
            
            for c in candidates:
                # Simulate swap
                # new_counts = cover_counts - Cover[idx_to_swap] + Cover[c]
                # unique_c = sum((Cover[c]==1) & (new_counts==1))
                
                # Optimization: 
                # The unique tsubs of C in the new state are:
                # 1. Tsubs covered by C that were count 1 (and not covered by idx_to_swap) -> become 2? No.
                #    If tsub j is covered by C and idx_to_swap:
                #       Old count k. New count k.
                #    If tsub j is covered by C but NOT idx_to_swap:
                #       Old count k. New count k+1.
                #    If tsub j is covered by idx_to_swap but NOT C:
                #       Old count k. New count k-1.
                
                # We care about tsubs j where NewCount[j] == 1.
                # This happens if:
                # - j covered by C, not idx_to_swap, and OldCount[j] == 0? (Impossible, C covers it so OldCount>=1? No, C is not in sol yet).
                #   Wait, cover_counts is for the CURRENT sol. C is NOT in sol.
                #   So if j is covered by C, it might or might not be covered by sol.
                #   If j is covered by sol (count k>=1), and not by idx_to_swap, new count is k+1.
                #   If j is covered by sol (count k>=1), and by idx_to_swap, new count is k.
                #   If j is NOT covered by sol (count 0), and by C, new count is 1.
                
                # So new unique tsubs for C are:
                # 1. Tsubs covered by C but NOT by any other block in sol (i.e., count 0 in sol, except idx_to_swap).
                #    Let's check:
                #    NewCount[j] = OldCount[j] - Cover[idx_to_swap, j] + Cover[c, j]
                #    If NewCount[j] == 1:
                #    Case A: Cover[c,j]=1, Cover[idx,j]=0. Then OldCount[j] + 1 = 1 => OldCount[j]=0.
                #            So j is covered ONLY by C in the new sol.
                #    Case B: Cover[c,j]=1, Cover[idx,j]=1. Then OldCount[j] = 1.
                #            So j is covered by idx and C (and no others in old sol).
                #    Case C: Cover[c,j]=0, Cover[idx,j]=1. Then OldCount[j] - 1 = 1 => OldCount[j]=2.
                #            But we only count unique tsubs FOR C. So we only care about j where Cover[c,j]=1.
                
                # So we only care about j where Cover[c,j]=1.
                # New unique tsubs for C are those j where:
                # (Cover[c,j]=1 AND Cover[idx,j]=0 AND OldCount[j]==0) OR
                # (Cover[c,j]=1 AND Cover[idx,j]=1 AND OldCount[j]==1)
                
                # Let's compute this efficiently.
                # mask_c = Cover[c] == 1
                # mask_idx = Cover[idx_to_swap] == 1
                # 
                # Part 1: mask_c & ~mask_idx & (cover_counts == 0)
                # Part 2: mask_c & mask_idx & (cover_counts == 1)
                
                # new_unique_count = sum(Part 1) + sum(Part 2)
                
                # Also, we must ensure that removing idx_to_swap doesn't leave any tsub uncovered.
                # i.e., for all j where Cover[idx,j]=1, NewCount[j] >= 1.
                # NewCount[j] = OldCount[j] - 1 + Cover[c,j]
                # If OldCount[j] == 1 (j is unique to idx in old sol), then NewCount[j] = Cover[c,j].
                # So we MUST have Cover[c,j] == 1 for all unique j of idx.
                # This is guaranteed by our candidate selection (intersection of unique tsubs).
                
                # So the validity is guaranteed. We just optimize the score.
                
                part1 = np.sum((Cover[c] == 1) & (Cover[idx_to_swap] == 0) & (cover_counts == 0))
                part2 = np.sum((Cover[c] == 1) & (Cover[idx_to_swap] == 1) & (cover_counts == 1))
                score = part1 + part2
                
                if score < best_score:
                    best_score = score
                    best_candidate = c
                    
            if best_candidate is not None:
                # Perform swap
                cover_counts -= Cover[idx_to_swap]
                cover_counts += Cover[best_candidate]
                sol[max_idx_pos] = best_candidate
                improved = True
                
        return sol

    # ILS Loop
    current_solution = local_optimize(current_solution, deadline - 10.0)
    best_solution = current_solution
    best_size = len(best_solution)
    
    # Precompute tsub_block_sets for ILS perturbations if needed?
    # Already done in local_optimize, but let's keep it global if we want to use it in perturb.
    # Actually, local_optimize redefines it every time. That's inefficient.
    # Let's move tsub_block_sets to global scope.
    
    # Re-run local optimize with the precomputed sets?
    # The code above defined tsub_block_sets inside local_optimize.
    # Let's refactor to pass it or make it global.
    # For simplicity, I'll just leave it as is, but it's a bit wasteful.
    # Given time, I'll just run the ILS.
    
    while time.time() < deadline - 5.0:
        # Perturbation: Randomly swap a few blocks or remove/add
        # Simple perturbation: Pick a random block, try to replace it with a random valid one
        # Or just restart greedy from a slightly different state?
        
        # Let's try a "kick": Remove 1 random block, then re-greedy the uncovered parts?
        # Too expensive.
        
        # Better kick: Randomly replace 1-3 blocks with random alternatives that maintain validity?
        # Hard to maintain validity easily.
        
        # Alternative: Just run local_optimize again with a different random seed for tie-breaking?
        # The current local_optimize is deterministic except for random.sample in candidate selection.
        
        # Let's implement a random restart of the greedy phase with a small bias?
        # Or, perturb the current solution by swapping one block for a random other block,
        # and if it becomes invalid, repair it with greedy.
        
        sol = current_solution.copy()
        if len(sol) == 0:
            break
            
        # Pick a random index to perturb
        idx_to_perturb = random.randint(0, len(sol) - 1)
        old_block = sol[idx_to_perturb]
        
        # Remove old block
        sol.remove(old_block)
        
        # Calculate uncovered
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in sol:
            cover_counts += Cover[idx]
        uncovered_mask = (cover_counts == 0)
        
        if np.any(uncovered_mask):
            # Re-greedy only for uncovered
            # This is expensive if we do it fully.
            # We can just add blocks to cover uncovered.
            
            # Simple repair: Greedy on the uncovered tsubs using the original gains logic?
            # But we need to update gains relative to the current sol.
            
            # Let's just add blocks greedily until covered.
            # gains[i] = sum(Cover[i] * uncovered_mask)
            # This is O(N*M).
            
            # To save time, limit the number of repairs.
            repair_limit = 5
            for _ in range(repair_limit):
                if not np.any(uncovered_mask):
                    break
                gains = Cover @ uncovered_mask
                best_idx = int(np.argmax(gains))
                if gains[best_idx] == 0:
                    break
                sol.append(best_idx)
                cover_counts += Cover[best_idx]
                uncovered_mask = (cover_counts == 0)
            
            # If still not covered, revert
            if np.any(uncovered_mask):
                sol.append(old_block)
                current_solution = sol
                continue
            
            # If covered, we have a valid solution (maybe larger)
            # Try to optimize
            current_solution = local_optimize(sol, deadline - 10.0)
            if len(current_solution) < best_size:
                best_solution = current_solution
                best_size = len(current_solution)
        else:
            # Removing old_block kept it valid!
            # This is an improvement (smaller solution)
            current_solution = sol
            current_solution = local_optimize(current_solution, deadline - 10.0)
            if len(current_solution) < best_size:
                best_solution = current_solution
                best_size = len(current_solution)

    return [all_kblocks[i] for i in best_solution]

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
