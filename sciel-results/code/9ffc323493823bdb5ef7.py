import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Safety margin

    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    # Precompute block to triple indices
    # block_t_indices[i] is a list of triple indices covered by block i
    block_t_indices = []
    for block in all_ks:
        subs = list(itertools.combinations(block, t))
        block_t_indices.append([idx_map[sub] for sub in subs])
        
    # Precompute triple to block indices
    # triple_to_blocks[i] is a list of block indices covering triple i
    triple_to_blocks = [[] for _ in range(n_ts)]
    for bi, indices in enumerate(block_t_indices):
        for ti in indices:
            triple_to_blocks[ti].append(bi)

    # Convert to numpy arrays for faster access if needed, but lists of lists are okay for small sizes
    # Actually, let's keep them as lists for flexibility in local search
    
    def greedy(seed):
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        # gain[i] = number of uncovered triples in block i
        gain = np.array([len(indices) for indices in block_t_indices], dtype=np.int32)
        blocks = []
        
        # To speed up finding max gain, we can use a heap or just np.argmax
        # np.argmax is O(N). With N=346k, it's fast.
        # Updating gain is the bottleneck.
        
        while True:
            if time.time() - start_time > time_limit - 5:
                break
            
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Random tie-breaking
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(best_idx)
            
            # Update covered and gain
            subs = block_t_indices[best_idx]
            for ti in subs:
                if not covered[ti]:
                    covered[ti] = True
                    # Decrement gain for all blocks containing this triple
                    # This loop is the main cost
                    for other_bi in triple_to_blocks[ti]:
                        gain[other_bi] -= 1
            
            if np.all(covered):
                break
        
        return blocks

    def local_search(blocks_list, time_budget):
        """
        Perform local search to minimize block count.
        Uses deletion and 1-for-1 replacement.
        """
        current = list(blocks_list)
        # Use a set for O(1) lookup
        current_set = set(current)
        deadline = time.time() + time_budget
        
        while time.time() < deadline:
            if time.time() - start_time > time_limit - 2:
                break
                
            improved = False
            
            # 1. Calculate coverage counts
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1
            
            # 2. Deletion Pass
            new_current_set = set()
            for b_idx in current:
                subs = block_t_indices[b_idx]
                # If all triples in this block are covered by at least 2 blocks, it's redundant
                if np.all(coverage_count[subs] > 1):
                    # It's redundant, so we don't add it to new set
                    improved = True
                else:
                    new_current_set.add(b_idx)
            
            if new_current_set != current_set:
                current = list(new_current_set)
                current_set = new_current_set
                # Restart loop to recalculate coverage
                continue
            
            # 3. 1-for-1 Replacement Pass
            # Try to replace a block with another that covers its unique triples
            # We iterate through blocks in random order
            indices = list(current)
            random.shuffle(indices)
            
            for b_idx in indices:
                if time.time() > deadline:
                    break
                    
                subs = block_t_indices[b_idx]
                # Find unique triples
                unique = [s for s in subs if coverage_count[s] == 1]
                
                if not unique:
                    continue # Handled by deletion or not critical
                
                # Find a block that covers all unique triples
                # Use the first unique triple to get candidates
                first_tri = unique[0]
                candidates = triple_to_blocks[first_tri]
                
                found_replacement = False
                for cb in candidates:
                    if cb == b_idx:
                        continue
                    
                    # Check if cb covers all unique triples
                    # Optimization: Check intersection
                    # unique is small (<= 35)
                    # cb_subs is 35
                    # We can check if set(unique).issubset(set(block_t_indices[cb]))
                    # But creating sets is slow.
                    # Let's use a boolean check.
                    cb_subs = block_t_indices[cb]
                    # Create a set of cb_subs for fast lookup? 
                    # Since we do this many times, precomputing sets for all blocks is memory heavy (346k sets).
                    # Instead, for each candidate, check unique against cb_subs.
                    # unique is very small.
                    all_covered = True
                    # To speed up, we can check if any unique triple is NOT in cb
                    # We can precompute a bitmask for cb? 
                    # 2024 bits is 253 bytes. 346k * 253 bytes ~ 87 MB. Feasible.
                    # But let's try the simple loop first.
                    for s in unique:
                        if s not in cb_subs:
                            all_covered = False
                            break
                    
                    if all_covered:
                        # We can replace b_idx with cb
                        # If cb is already in current_set, this is just a deletion of b_idx
                        # If cb is not in current_set, this is a replacement
                        current_set.remove(b_idx)
                        current_set.add(cb)
                        current = list(current_set)
                        improved = True
                        found_replacement = True
                        break # Restart coverage calculation
                
                if found_replacement:
                    break # Restart outer loop to recalculate coverage
            
            if not improved:
                break # Local optimum reached
                
        return current

    # Main Logic
    best_blocks = None
    best_count = float('inf')
    
    # Run greedy with a few seeds to get diverse starting points
    # Then refine with local search
    num_seeds = 5
    for seed in range(num_seeds):
        if time.time() - start_time > time_limit - 10:
            break
            
        greedy_blocks = greedy(seed)
        
        # Refine
        # Allocate remaining time minus safety margin to local search
        remaining_time = time_limit - (time.time() - start_time) - 5
        if remaining_time < 1.0:
            continue
            
        # Split time among seeds? Or just give each seed a chunk?
        # Let's give each seed a proportional share of the remaining time
        # Or just run local search until no improvement or time runs out
        # Since local search is fast if no improvements, we can just run it.
        
        refined = local_search(greedy_blocks, remaining_time)
        
        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined
            
        # If we hit a very good score, we might stop
        if best_count <= 76:
            break

    if best_blocks is None:
        # Fallback: use the last greedy result
        best_blocks = greedy(0)
        
    # Convert indices to actual blocks
    final_blocks = [list(all_ks[i]) for i in best_blocks]
    
    # Optional: Final verification (expensive, but safe)
    # For v=24, k=7, t=3, it's fast enough to check a few times?
    # No, checking 2024 triples against ~76 blocks is fast.
    # But we trust the local search.
    
    return final_blocks
