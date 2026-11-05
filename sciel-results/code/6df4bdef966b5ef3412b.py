import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    # M[i, j] = 1 if block i covers tsub j
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Helper to get block list from indices
    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    # Phase 1: Vectorized Greedy Construction
    # This is fast and provides a strong initial solution
    if time.monotonic() < deadline - 5.0:
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        selected_mask = np.zeros(num_kblocks, dtype=bool)
        selected_indices = []
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 10.0:
                break
            
            # Calculate gain for each block: number of uncovered tsubs it covers
            gains = M @ uncovered
            
            # Find block with max gain
            best_idx = np.argmax(gains)
            
            if gains[best_idx] == 0:
                break
            
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update uncovered and counts
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            counts[covered_indices] += 1
        
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_solution = selected_indices[:]
            best_counts = counts.copy()
            best_selected_mask = selected_mask.copy()

    # Phase 2: Local Search Polish with ILS
    # We operate on the best solution found so far
    if best_solution is not None:
        current_indices = best_solution[:]
        current_counts = best_counts.copy()
        current_mask = best_selected_mask.copy()
        
        improved = True
        stagnation_counter = 0
        max_stagnation = 50 # If no improvement for 50 iterations, perturb
        
        while time.monotonic() < deadline - 2.0:
            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                
            # If stagnating, perform a perturbation (ILS)
            if stagnation_counter >= max_stagnation and len(current_indices) > 0:
                # Remove 5-10 random blocks
                n_remove = min(5 + rng.randint(0, 5), len(current_indices))
                to_remove = rng.sample(current_indices, n_remove)
                for b_idx in to_remove:
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    row = M[b_idx]
                    covered_indices = np.where(row == 1)[0]
                    current_counts[covered_indices] -= 1
                
                # Greedily re-add to cover gaps
                uncovered_mask = current_counts == 0
                while np.any(uncovered_mask):
                    if time.monotonic() > deadline - 5.0:
                        break
                    gains = M @ uncovered_mask
                    best_idx = np.argmax(gains)
                    if gains[best_idx] == 0:
                        break
                    current_mask[best_idx] = True
                    current_indices.append(best_idx)
                    row = M[best_idx]
                    covered_indices = np.where(row == 1)[0]
                    current_counts[covered_indices] += 1
                    uncovered_mask = current_counts == 0
                    
                # Reset stagnation
                stagnation_counter = 0
                improved = True
                continue

            improved = False
            current_indices_set = set(current_indices)
            
            # 1. Remove Redundant Blocks
            # A block is redundant if all tsubs it covers have count > 1
            redundant_found = False
            for b_idx in current_indices:
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                if len(covered_indices) == 0:
                    continue
                min_count = np.min(current_counts[covered_indices])
                if min_count > 1:
                    # Redundant
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    improved = True
                    redundant_found = True
                    break # Restart loop to re-evaluate
            
            if redundant_found:
                continue

            # 2. Try 1-for-1 swaps
            # For each block in the current solution, try replacing it with a block not in the solution
            # if the new block covers all tsubs that the old block uniquely covered, and the new block
            # adds some coverage (or at least doesn't lose coverage), and we can remove the old block.
            # Actually, a valid swap is: remove block A, add block B.
            # Condition: The set of tsubs covered by A but not by any other block in (S - A + B) must be subset of tsubs covered by B.
            # More simply: If we remove A, some tsubs become uncovered (count drops to 0).
            # If we add B, those tsubs must be covered by B.
            
            # To make this efficient, we can iterate through blocks in S.
            # For a candidate A in S:
            #   unique_to_A = tsubs where count == 1 and A covers it.
            #   If unique_to_A is empty, A is redundant (handled above).
            #   So unique_to_A is non-empty.
            #   We need a block B not in S such that B covers all tsubs in unique_to_A.
            #   If such a B exists, we can swap A -> B.
            #   Does this reduce the number of blocks? No, it's a swap.
            #   But it might enable future removals?
            #   Actually, the goal is to reduce the number of blocks.
            #   Swaps alone don't reduce the count.
            #   However, after a swap, the coverage counts change.
            #   Maybe a different block becomes redundant?
            #   
            #   Alternative: Try to find a B that covers unique_to_A.
            #   If we find B, we swap.
            #   Then we check for redundancies again.
            #   This is a standard local search move.
            
            # Optimization:
            # Precompute for each tsub, which blocks cover it? No, that's large.
            # Instead, for a candidate A, compute unique_to_A.
            # Then check if there is any block B in S^c that covers all of unique_to_A.
            # This is a subset query.
            
            # Let's try a simpler heuristic:
            # Try to remove a block A if it's redundant. (Done)
            # If not, try to find a block B that covers the "most" of A's unique tsubs.
            # If B covers ALL of A's unique tsubs, we can swap A->B.
            # This doesn't reduce size, but changes the configuration.
            # If we swap, we mark improved=True to allow further removals.
            
            swap_found = False
            # Limit swaps to avoid long loops
            max_swaps_per_iter = 5
            swaps_done = 0
            
            for b_idx in current_indices:
                if swaps_done >= max_swaps_per_iter:
                    break
                
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                if len(covered_indices) == 0:
                    continue
                
                # Find unique tsubs
                unique_mask = (current_counts == 1) & (row == 1)
                unique_indices = np.where(unique_mask)[0]
                
                if len(unique_indices) == 0:
                    # Redundant, should have been caught
                    continue
                
                # We need a block B not in S that covers all unique_indices.
                # Check all blocks not in S? Too slow (50k blocks).
                # Heuristic: Check a sample of blocks not in S?
                # Or, check if there is a block B that covers the "core" of unique_indices.
                
                # Let's try to find B that maximizes coverage of unique_indices.
                # gains_B = sum(M[B, unique_indices])
                # If max gains_B == len(unique_indices), then B covers all unique tsubs.
                
                # To speed up, we can precompute for each tsub, a list of blocks?
                # Or just do a matrix multiplication with a sparse vector?
                # unique_vec = np.zeros(num_tsubs, dtype=np.uint8)
                # unique_vec[unique_indices] = 1
                # gains = M @ unique_vec
                # candidates = np.where(gains == len(unique_indices))[0]
                # candidates = candidates[~current_mask[candidates]]
                
                # This is O(num_blocks * num_tsubs) per swap check?
                # M is (50388, 3876). unique_vec is sparse.
                # M @ unique_vec is fast in numpy.
                
                unique_vec = np.zeros(num_tsubs, dtype=np.uint8)
                unique_vec[unique_indices] = 1
                gains = M @ unique_vec
                
                # Find blocks that cover ALL unique tsubs
                full_cover_mask = (gains == len(unique_indices))
                # Exclude blocks already in S
                full_cover_mask[current_indices_set] = False # This is slow if current_indices_set is a set
                # Better:
                full_cover_mask &= ~current_mask
                
                candidates = np.where(full_cover_mask)[0]
                
                if len(candidates) > 0:
                    # Pick a candidate B. Maybe the one that covers the most OTHER tsubs?
                    # Or just the first one.
                    # Let's pick the one with highest total gain (covers most tsubs in general)
                    # Or just random from candidates to diversify.
                    best_b = candidates[0] # Or rng.choice(candidates)
                    
                    # Perform swap: Remove b_idx, Add best_b
                    # Remove b_idx
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    
                    # Add best_b
                    current_mask[best_b] = True
                    current_indices.append(best_b)
                    new_row = M[best_b]
                    new_covered = np.where(new_row == 1)[0]
                    current_counts[new_covered] += 1
                    
                    swap_found = True
                    improved = True
                    swaps_done += 1
                    break # Restart loop to re-evaluate counts
                    
            if swap_found:
                continue

            if not improved:
                # No removal, no swap found
                # Stagnation counter will be incremented at the top of the loop
                # But we are at the bottom.
                # Actually, the logic is:
                # if not improved: stagnation_counter += 1
                # So we just break out of the inner loop and let the outer loop handle it.
                # But we need to make sure we don't infinite loop here.
                # The `continue` statements handle the restarts.
                # If we reach here, it means no moves were made.
                pass

    if best_solution is None:
        # Fallback: single greedy
        rng = random.Random(0)
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        while np.any(uncovered):
            if time.monotonic() > deadline - 5.0:
                break
            gains = M @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
        
        best_solution = selected_indices

    return get_blocks(best_solution)
