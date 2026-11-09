import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Generate all k-subsets and their covered t-subset indices
    k_subs = list(itertools.combinations(range(v), k))
    num_k_subs = len(k_subs)
    
    block_t_subs = []
    for block in k_subs:
        covered_indices = []
        for t_sub in itertools.combinations(block, t):
            covered_indices.append(t_sub_to_idx[t_sub])
        block_t_subs.append(covered_indices)
        
    # 2. Fast Randomized Greedy Initialization
    # We pick blocks greedily but only from a random sample to save time
    # This yields a valid but potentially suboptimal solution
    uncovered_mask = [1] * num_t_subs # 1 if uncovered, 0 if covered
    solution_indices = []
    
    # To ensure we finish fast, we limit the greedy steps or sample size
    # If we sample too few, we might take many steps. 
    # Let's sample 1000 blocks per step.
    sample_size = 1000
    
    while any(uncovered_mask):
        # Sample random blocks
        candidates = random.sample(range(num_k_subs), min(sample_size, num_k_subs))
        
        best_block = -1
        best_gain = 0
        
        for idx in candidates:
            gain = 0
            for t_idx in block_t_subs[idx]:
                if uncovered_mask[t_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = idx
        
        if best_block == -1:
            # Fallback: pick a random block if no gain found (should not happen if uncovered exists and sample is large enough)
            # Actually if uncovered exists, some block must cover it.
            # If sample is too small, we might miss it. 
            # To be safe, if best_gain == 0, do a full scan or increase sample.
            # For robustness, let's just pick a random block from ALL blocks if gain is 0.
            best_block = random.randint(0, num_k_subs - 1)
            
        solution_indices.append(best_block)
        
        # Update uncovered mask
        for t_idx in block_t_subs[best_block]:
            uncovered_mask[t_idx] = 0
            
        # Time check
        if time.time() - start_time > time_limit * 0.2:
            break

    # Ensure we have a valid solution. If the greedy broke early due to time,
    # we might have uncovered subsets. We must complete it.
    # Re-check uncovered
    uncovered_indices = [i for i, val in enumerate(uncovered_mask) if val == 1]
    if uncovered_indices:
        # Complete with a standard greedy (full scan) for the remaining uncovered
        # This might be slow if many are uncovered, but hopefully few.
        while uncovered_indices:
            best_block = -1
            best_gain = 0
            # Full scan is O(N * K). N=346k, K=35. ~12M ops.
            # If we have many uncovered, this could take seconds.
            # Let's try to find blocks covering the specific uncovered ones.
            # For each uncovered t_sub, find blocks containing it.
            # This is faster.
            
            # Map t_sub_idx to list of block indices containing it
            # This map is not built. Building it is O(N*K) = 12M ops.
            # Let's build it once if needed?
            # Actually, we can just do a full scan but it's risky.
            
            # Alternative: Just add random blocks until covered?
            # No, let's do a targeted greedy.
            # Pick a random uncovered t_sub.
            # Find all blocks containing it.
            # Pick the one that covers the most other uncovered t_subs.
            
            target_t_idx = random.choice(uncovered_indices)
            
            # Find blocks containing target_t_idx
            # We didn't precompute reverse map. 
            # Let's just iterate all blocks and check if target_t_idx is in block_t_subs[i].
            # This is O(N) per step.
            
            best_block = -1
            best_gain = 0
            for i in range(num_k_subs):
                if target_t_idx in block_t_subs[i]:
                    gain = 0
                    for t_idx in block_t_subs[i]:
                        if uncovered_mask[t_idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                        
            if best_block == -1:
                # Should not happen
                break
                
            solution_indices.append(best_block)
            for t_idx in block_t_subs[best_block]:
                if uncovered_mask[t_idx]:
                    uncovered_mask[t_idx] = 0
                    uncovered_indices.remove(t_idx) # O(N) removal, slow. Use set?
            
            if time.time() - start_time > time_limit * 0.5:
                # If we are running out of time, just add random blocks to cover remaining
                # This is a fallback for safety
                while uncovered_indices:
                    target_t_idx = uncovered_indices[0]
                    # Find ANY block containing it
                    for i in range(num_k_subs):
                        if target_t_idx in block_t_subs[i]:
                            solution_indices.append(i)
                            for t_idx in block_t_subs[i]:
                                uncovered_mask[t_idx] = 0
                            uncovered_indices.remove(target_t_idx)
                            break
                break

    # 3. Local Search: Prune and Swap
    # Initialize counts
    counts = np.zeros(num_t_subs, dtype=np.int16)
    for idx in solution_indices:
        for t_idx in block_t_subs[idx]:
            counts[t_idx] += 1
            
    # Convert to list for easier manipulation if needed, but numpy is fine
    # We need to check if min(counts) > 0
    
    def prune():
        nonlocal solution_indices, counts
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias
            random.shuffle(solution_indices)
            for i in range(len(solution_indices)):
                block_idx = solution_indices[i]
                # Tentatively remove
                for t_idx in block_t_subs[block_idx]:
                    counts[t_idx] -= 1
                
                if np.min(counts) > 0:
                    # Valid removal
                    solution_indices.pop(i)
                    changed = True
                    break # Restart shuffle/check
                else:
                    # Restore
                    for t_idx in block_t_subs[block_idx]:
                        counts[t_idx] += 1
                        
    # Prune initial solution
    prune()
    
    # Local Search Loop
    while time.time() - start_time < time_limit * 0.9:
        # Try a swap
        if len(solution_indices) > 0:
            # Pick random block from solution
            pos = random.randint(0, len(solution_indices) - 1)
            out_idx = solution_indices[pos]
            
            # Pick random block from all blocks
            in_idx = random.randint(0, num_k_subs - 1)
            
            if out_idx == in_idx:
                continue
                
            # Tentatively swap
            # Remove out_idx
            for t_idx in block_t_subs[out_idx]:
                counts[t_idx] -= 1
            
            # Check if any count is 0
            # If so, in_idx MUST cover those 0s
            zeros = np.where(counts == 0)[0]
            
            valid = True
            if len(zeros) > 0:
                # Check if in_idx covers all zeros
                in_set = set(block_t_subs[in_idx])
                for z in zeros:
                    if z not in in_set:
                        valid = False
                        break
            
            if valid:
                # Add in_idx
                for t_idx in block_t_subs[in_idx]:
                    counts[t_idx] += 1
                
                # Swap
                solution_indices[pos] = in_idx
                
                # Try to prune immediately, as this swap might have made other blocks redundant
                prune()
            else:
                # Restore out_idx
                for t_idx in block_t_subs[out_idx]:
                    counts[t_idx] += 1
        else:
            break
            
    # Final Prune
    prune()
    
    # Convert indices to blocks
    result = [list(k_subs[idx]) for idx in solution_indices]
    
    return result
