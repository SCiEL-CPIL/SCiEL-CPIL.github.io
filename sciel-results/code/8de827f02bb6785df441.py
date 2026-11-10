import itertools
import time
import random
import numpy as np
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # 118s out of 120s limit, leaving 2s safety margin
    
    # Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubsets)
    
    # Map t-subsets to indices
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    # Precompute mask for every possible k-block?
    # C(24, 7) = 346,104. 
    # Each mask has C(7,3)=35 bits.
    # Storing 346k masks of length 2024 as bitsets (numpy arrays) might be memory heavy but doable?
    # 346k * 2024 / 8 bytes ~ 87 MB. Acceptable.
    # However, computing them takes time. 
    # Alternative: Compute mask on the fly using a lookup table for points?
    # No, let's precompute masks for all possible blocks. It's the fastest way to evaluate moves.
    
    # We represent a block by its index in the list of all combinations.
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block tuple to index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage masks for all blocks
    # mask[block_idx] is a numpy array of bools of length num_tsubs
    # Or better, a list of indices of t-subsets covered by the block.
    # Since we need to update counts, a list of indices is faster for iteration if sparse.
    # But for "is covered by block X" checks, bitsets are faster.
    
    # Let's use a list of sets of t-subset indices for each block.
    # This allows fast intersection/union operations?
    # Actually, for local search, we need:
    # 1. How many t-subsets are covered ONLY by block A (and not B)?
    # 2. How many t-subsets are covered by B but not A?
    
    # Let's precompute `block_tsubs[block_idx]` = list of t-subset indices.
    block_tsubs = [None] * num_blocks
    for i, block in enumerate(all_blocks):
        subs = []
        for tsub in itertools.combinations(block, t):
            subs.append(tsub_to_idx[tsub])
        block_tsubs[i] = subs
    
    # Helper to get mask indices for a block
    def get_mask(block):
        return block_tsubs[block_to_idx[block]]
    
    # Greedy Set Cover Construction
    def greedy_construct():
        # uncovered_mask: boolean array of length num_tsubs, True if uncovered
        uncovered_mask = np.ones(num_tsubs, dtype=bool)
        blocks = []
        
        # For speed, we can precompute for each t-subset, which blocks cover it?
        # That's the inverse mapping. 
        # tsub_to_blocks[tsub_idx] = list of block indices.
        # Size: 2024 * (avg blocks per tsub).
        # Each tsub is in C(24-3, 7-3) = C(21,4) = 5985 blocks.
        # 2024 * 5985 ~ 12 million entries. Memory ~ 100MB. Time to build ~ 1s.
        # This makes greedy selection O(1) per candidate? No, we still need to pick the best block.
        
        # Standard greedy:
        # 1. Find block with max new coverage.
        # 2. Add to set.
        # 3. Update uncovered.
        
        # To speed up "max new coverage", we can maintain a count of how many uncovered tsubs each block covers.
        # Initially, count[b] = len(block_tsubs[b]) = 35.
        # When we add a block, we need to decrement counts for all tsubs it covers.
        # This is expensive if done naively.
        
        # Alternative: Randomized Greedy with restarts.
        # Since v=24, k=7, t=3 is small, a simple greedy might be fast enough if optimized.
        
        # Let's try a simpler approach: 
        # Use the precomputed `block_tsubs` and a global `cover_count` array.
        # At each step, iterate all 346k blocks and calculate gain? 
        # 346k * 35 ops = 12M ops per step. 
        # If we need 70 steps, 840M ops. Too slow in Python.
        
        # We need a smarter greedy.
        # Heuristic: Pick a random uncovered t-subset. Pick a random block covering it.
        # This is "Randomized Greedy". It's fast.
        
        # Let's implement Randomized Greedy.
        # 1. Find all uncovered tsubs.
        # 2. Pick one at random.
        # 3. Find all blocks covering it. (Precomputed list)
        # 4. Pick one at random (or one with max total coverage? Max coverage is better).
        
        # Let's precompute tsub_to_blocks.
        tsub_to_blocks = [[] for _ in range(num_tsubs)]
        for b_idx, subs in enumerate(block_tsubs):
            for s in subs:
                tsub_to_blocks[s].append(b_idx)
        
        # Now greedy:
        covered_mask = np.zeros(num_tsubs, dtype=bool)
        current_blocks = []
        
        while not covered_mask.all():
            # Find an uncovered tsub
            # To avoid O(N) scan every time, we can keep a list of uncovered indices and remove from it.
            # But removal is O(N). 
            # For 2024 items, O(N) scan is fine.
            uncovered_indices = np.where(~covered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            
            target_tsub = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            
            # Get blocks covering this tsub
            candidates = tsub_to_blocks[target_tsub]
            
            # Select best candidate
            # Heuristic: Maximize number of *other* uncovered tsubs covered by this block.
            # Or just pick randomly? Random is faster.
            # Let's pick the one with max coverage of uncovered set.
            best_block = -1
            best_gain = -1
            
            # Sample if list is large? 5985 is manageable.
            for b_idx in candidates:
                # Calculate gain
                # Gain = number of uncovered tsubs in block_tsubs[b_idx]
                # This requires iterating 35 items and checking covered_mask.
                # 5985 * 35 = 210k checks. 
                # If we do this 70 times, 15M checks. ~1s in NumPy? Maybe.
                
                # Optimization: Precompute nothing, just loop.
                # Actually, let's just pick a random candidate to be fast, 
                # and rely on local search to fix quality.
                # Or use a simple heuristic: pick the block that covers the most *total* tsubs? 
                # All cover 35. 
                # Pick the one that covers the most *uncovered*?
                
                # Let's try: Pick random candidate.
                b_idx = random.choice(candidates)
                
                # Check if it's already in current_blocks
                if b_idx in [block_to_idx[b] for b in current_blocks]:
                    continue
                
                # Add it
                current_blocks.append(all_blocks[b_idx])
                
                # Update covered_mask
                for s in block_tsubs[b_idx]:
                    covered_mask[s] = True
                    
                break # Only add one block per iteration? 
                # Wait, if we pick a random candidate, we might pick a bad one.
                # Let's pick the best one among a sample.
                
            # Actually, the loop above breaks after picking one. 
            # Let's refine: Pick the best among a random sample of 10 candidates.
            
            # Restart the selection logic properly
            # (The previous logic was flawed in the loop structure)
            
            sample_size = min(10, len(candidates))
            sampled = random.sample(candidates, sample_size)
            
            best_b = -1
            best_g = -1
            for b_idx in sampled:
                gain = 0
                for s in block_tsubs[b_idx]:
                    if not covered_mask[s]:
                        gain += 1
                if gain > best_g:
                    best_g = gain
                    best_b = b_idx
            
            if best_b != -1:
                current_blocks.append(all_blocks[best_b])
                for s in block_tsubs[best_b]:
                    covered_mask[s] = True
            else:
                # Fallback
                b_idx = random.choice(candidates)
                current_blocks.append(all_blocks[b_idx])
                for s in block_tsubs[b_idx]:
                    covered_mask[s] = True
        
        return current_blocks

    # Local Search to reduce block count
    def local_search(initial_blocks):
        # State:
        # blocks: list of block tuples
        # cover_count: array of length num_tsubs, count of blocks covering each tsub
        # block_masks: list of lists of tsub indices for each block
        
        current_blocks = list(initial_blocks)
        N = len(current_blocks)
        
        # Ensure distinct blocks
        unique_blocks = list(set(current_blocks))
        current_blocks = unique_blocks
        N = len(current_blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=int)
        block_masks = []
        
        for b in current_blocks:
            mask = get_mask(b)
            block_masks.append(mask)
            cover_count[mask] += 1
        
        current_uncovered = np.sum(cover_count == 0)
        
        best_blocks = current_blocks[:]
        best_uncovered = current_uncovered
        
        # If already valid, try to reduce
        if current_uncovered == 0:
            # Try removing blocks
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for i in range(len(current_blocks) - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    mask = block_masks[i]
                    # Check if any tsub in mask has count 1
                    if np.any(cover_count[mask] == 1):
                        continue
                    # Can remove
                    cover_count[mask] -= 1
                    del current_blocks[i]
                    del block_masks[i]
                    changed = True
                    best_blocks = current_blocks[:]
                    best_uncovered = 0 # Still valid
                    N -= 1
            
            if len(current_blocks) < len(best_blocks):
                best_blocks = current_blocks[:]
        
        # Simulated Annealing to find better covering (fewer blocks or same blocks but more robust?)
        # Objective: Minimize number of blocks.
        # Since we start with a valid covering, we can try to swap blocks to allow removal of others.
        # Move: Replace block A with block B.
        # If B covers more "critical" tsubs (count 1) than A, we might be able to remove another block C.
        
        # Simpler SA: Minimize uncovered count (if we start with invalid) or Minimize block count (if valid).
        # Let's stick to: Start with greedy (valid). Try to reduce count via SA.
        
        T = 5.0
        cooling = 0.999
        steps = 0
        max_steps = 5000
        
        while T > 0.5 and steps < max_steps and time.time() < deadline - 5:
            # Move: Replace a random block with a new random block
            i = random.randint(0, N - 1)
            old_block = current_blocks[i]
            old_mask = block_masks[i]
            
            # New block: Random
            new_block = random.choice(all_blocks)
            
            # Check if new_block is already in current_blocks
            if new_block in current_blocks:
                steps += 1
                continue
            
            new_mask = get_mask(new_block)
            
            # Calculate delta in uncovered count
            # old_mask -> new_mask
            # tsubs in old_mask but not new_mask: count decreases by 1
            # tsubs in new_mask but not old_mask: count increases by 1
            
            # We want to minimize uncovered.
            # If we replace A with B:
            # - If a tsub was covered only by A (count 1) and is not in B, it becomes uncovered (bad).
            # - If a tsub was uncovered (count 0) and is in B, it becomes covered (good).
            
            # Efficient delta:
            # We need to know counts for tsubs in symmetric difference.
            
            # Convert masks to sets for fast intersection? 
            # Or use numpy?
            # old_mask and new_mask are lists of indices.
            
            # Let's use sets for this step, as N is small (~70) and masks are small (35).
            old_set = set(old_mask)
            new_set = set(new_mask)
            
            only_old = old_set - new_set
            only_new = new_set - old_set
            
            delta_uncovered = 0
            
            for idx in only_old:
                if cover_count[idx] == 1:
                    delta_uncovered += 1
            
            for idx in only_new:
                if cover_count[idx] == 0:
                    delta_uncovered -= 1
            
            # Acceptance
            if delta_uncovered <= 0 or random.random() < np.exp(-delta_uncovered / T):
                # Apply move
                for idx in only_old:
                    cover_count[idx] -= 1
                for idx in only_new:
                    cover_count[idx] += 1
                
                current_blocks[i] = new_block
                block_masks[i] = new_mask
                current_uncovered += delta_uncovered
                
                # Check if we can remove any block now?
                # This is a "repair" step.
                # Try to remove a random block if possible.
                if random.random() < 0.1:
                    for j in range(len(current_blocks) - 1, -1, -1):
                        mask_j = block_masks[j]
                        if np.any(cover_count[mask_j] == 1):
                            continue
                        # Remove
                        cover_count[mask_j] -= 1
                        del current_blocks[j]
                        del block_masks[j]
                        N -= 1
                        current_uncovered = np.sum(cover_count == 0)
                        break # Only one removal per step?
                
                if current_uncovered == 0:
                    if N < len(best_blocks):
                        best_blocks = current_blocks[:]
                        best_uncovered = 0
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = current_blocks[:]
            
            steps += 1
            T *= cooling
            
        return best_blocks

    # Main Execution
    # 1. Run Greedy a few times to get a good initial valid covering
    # 2. Run Local Search on the best one
    
    best_valid_blocks = None
    best_valid_N = float('inf')
    
    # Time allocation
    greedy_time = 10.0
    sa_time = 100.0
    
    # Run greedy
    for i in range(5):
        if time.time() > start_time + greedy_time:
            break
        blocks = greedy_construct()
        # Verify validity
        # (Greedy should be valid by construction, but let's double check)
        # Actually, greedy_construct ensures all tsubs are covered.
        
        N = len(blocks)
        if N < best_valid_N:
            best_valid_N = N
            best_valid_blocks = blocks
    
    if best_valid_blocks is None:
        # Fallback: Random blocks until valid?
        # Should not happen with greedy.
        return []
    
    # Run SA to improve
    # Use remaining time
    while time.time() < deadline - 2:
        # Restart SA with a slightly perturbed version of best_valid_blocks?
        # Or just run SA on best_valid_blocks.
        
        # To avoid getting stuck, we can try SA on a random subset of best_valid_blocks?
        # No, SA needs a valid start or close to it.
        
        # Let's run SA on best_valid_blocks.
        # If it finds a smaller valid covering, update best.
        
        improved = local_search(best_valid_blocks)
        
        if len(improved) < len(best_valid_blocks):
            best_valid_blocks = improved
            best_valid_N = len(improved)
        
        # If we have time, try SA on a different greedy result?
        # Or just keep running SA on the current best.
        
        # To diversify, we can randomly drop a block and run SA to repair?
        # This is a "kick".
        
        if time.time() < deadline - 10:
            # Kick: Remove a random block from best_valid_blocks
            kick_blocks = best_valid_blocks[:]
            if len(kick_blocks) > 1:
                idx_to_remove = random.randint(0, len(kick_blocks)-1)
                del kick_blocks[idx_to_remove]
            
            # Run SA on kick_blocks (it will be invalid, SA must fix it)
            # Note: local_search above assumes valid start? 
            # My local_search above handles invalid starts (minimizes uncovered).
            # If it reaches 0 uncovered, it's valid.
            
            repaired = local_search(kick_blocks)
            
            # Check if repaired is valid
            # local_search returns best_blocks found during SA.
            # We need to verify if it's valid.
            
            # Re-verify validity of repaired
            cover_count_check = np.zeros(num_tsubs, dtype=int)
            for b in repaired:
                cover_count_check[get_mask(b)] += 1
            
            if np.all(cover_count_check > 0):
                if len(repaired) < len(best_valid_blocks):
                    best_valid_blocks = repaired
                    best_valid_N = len(repaired)
    
    return [list(b) for b in best_valid_blocks]
