import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed):
        """
        Efficient greedy set cover using NumPy for gain updates.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Initial gains: number of t-subsets covered by each block
        # Since all blocks cover exactly C(k,t) subsets, and initially all are uncovered,
        # initial gain is constant. But as we pick blocks, gains drop.
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        selected = []
        
        # Optimization: Instead of checking all blocks every time, we could use a heap,
        # but updating a heap is O(log N) and checking all is O(N).
        # N is ~75532. Iterations ~300. 300 * 75k = 22.5M ops. Acceptable in Python? 
        # NumPy argmax is fast.
        
        # We need to handle the case where gains become 0.
        
        for _ in range(n_k): # Safety bound
            if not uncovered.any():
                break
            
            # Add noise for randomization if seed > 0
            if seed > 0:
                noise = rng.random() * 0.1
            else:
                noise = 0.0
            
            # Find best block
            # We only care about blocks with gain > 0
            # To speed up, we could track active blocks, but simple argmax is fine for now.
            # However, scanning 75k elements 300 times is ~22.5M operations. 
            # np.argmax is C-level, so it's fast.
            
            # We need to exclude already selected blocks?
            # If a block is selected, its gain becomes 0 for the uncovered subsets it covers.
            # If it covers no *new* uncovered subsets, gain is 0.
            # So argmax will naturally avoid blocks with 0 gain (unless all are 0, then we break).
            
            # But we must ensure we don't pick the same block twice if it still has gain?
            # Wait, if we pick block B, we mark its t-subsets as covered.
            # The gain of B for *future* steps is the number of *uncovered* subsets it covers.
            # Since we just covered them, gain becomes 0.
            # So B will not be picked again.
            
            scores = gains.copy()
            if seed > 0:
                # Add small noise to break ties
                scores += rng.random() * 0.01
            
            best_idx = int(np.argmax(scores))
            
            if scores[best_idx] == 0:
                break
                
            selected.append(best_idx)
            
            # Update gains
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Decrease gain for all blocks covering this t_subset
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
                        
        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def simulated_annealing(initial_blocks, max_time):
        """
        Simulated Annealing on block selection.
        State: List of block indices.
        Energy: len(blocks) + penalty * num_uncovered
        Moves: 
          1. Remove a block (if valid, always accept; else prob accept)
          2. Swap a block with another that covers its unique subsets (if valid, always accept; else prob accept)
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        current_energy = len(current_blocks) + 1000.0 * np.sum(cover_count == 0)
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        # SA Parameters
        T = 50.0  # Initial temperature
        cooling_rate = 0.99
        min_T = 0.1
        penalty = 1000.0
        
        rng = random.Random(42)
        
        while time.time() < max_time and T > min_T:
            # --- Move 1: Try to Remove ---
            if len(current_blocks) > 0:
                # Pick random block
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                # Check if removal is valid
                # It's valid if all t-subsets covered by b_idx are covered by at least 1 other block
                # i.e., cover_count[t] > 1 for all t in k_sub_t_idxs[b_idx]
                valid_removal = True
                new_uncovered = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        valid_removal = False
                        new_uncovered += 1
                    # If cover_count[t_idx] > 1, it remains covered.
                    # If cover_count[t_idx] == 1, it becomes uncovered.
                
                if valid_removal:
                    # Always accept
                    # Perform removal
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    # Energy change: +1 (size) + penalty * new_uncovered
                    delta_E = 1.0 + penalty * new_uncovered
                    # Accept with probability exp(-delta_E / T)
                    if rng.random() < np.exp(-delta_E / T):
                        # Accept invalid move
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E

            # --- Move 2: Try to Swap ---
            # Only try swap if we have blocks
            if len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                
                # Find unique subsets covered by old_b
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if not unique_ts:
                    # No unique subsets. We can swap with any other block.
                    # This allows "shuffling" the set without breaking coverage.
                    # Pick a random other block
                    # To avoid picking the same block, we can pick a random index from all_k_subs
                    # and check if it's in current_blocks
                    # This is O(N) check. 
                    # Better: pick a random block from all_k_subs that is NOT in current_blocks.
                    # Since current_blocks is small (~300) and all_k_subs is large (~75k),
                    # random sampling is likely to pick a new block.
                    
                    new_b = rng.randint(0, n_k - 1)
                    if new_b in current_blocks:
                        continue # Skip this move, try again next iteration
                    
                    # Perform swap
                    # Remove old_b
                    for t_idx in k_sub_t_idxs[old_b]:
                        cover_count[t_idx] -= 1
                    # Add new_b
                    for t_idx in k_sub_t_idxs[new_b]:
                        cover_count[t_idx] += 1
                    
                    # Check if this swap caused any new uncovered subsets
                    # We need to check if any t_idx in k_sub_t_idxs[new_b] became uncovered?
                    # No, adding a block only increases cover count.
                    # The risk is if old_b was the *only* cover for some t, and new_b doesn't cover it.
                    # But we established unique_ts is empty, so old_b is not the only cover for any t.
                    # So cover_count[t] >= 2 for all t in old_b.
                    # After removal, cover_count[t] >= 1.
                    # So no new uncovered subsets.
                    # Energy change: 0 (size same, coverage same).
                    # Always accept.
                    
                    current_blocks[i] = new_b
                    # current_energy unchanged
                    
                else:
                    # We have unique subsets. New block MUST cover all of them.
                    # Find candidates
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    
                    # Remove old_b and current blocks from candidates
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                    
                    if not candidates:
                        continue # No valid swap candidates
                    
                    # Pick a random candidate
                    new_b = rng.choice(list(candidates))
                    
                    # Perform swap
                    for t_idx in k_sub_t_idxs[old_b]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[new_b]:
                        cover_count[t_idx] += 1
                    
                    # Check for new uncovered subsets
                    # Since new_b covers all unique_ts, those remain covered.
                    # Other subsets covered by old_b: cover_count was >= 2, now >= 1.
                    # Subsets covered by new_b: cover_count increases.
                    # So no new uncovered subsets.
                    # Energy change: 0.
                    # Always accept.
                    
                    current_blocks[i] = new_b

            # --- Refinement: Try to Remove Redundant Blocks ---
            # This is crucial to actually reduce the size.
            # We do a simple pass to remove any block that is now redundant.
            # This is essentially a "greedy deletion" step.
            changed = True
            while changed:
                changed = False
                # Shuffle to avoid bias
                rng.shuffle(current_blocks)
                for j in range(len(current_blocks)):
                    b_idx = current_blocks[j]
                    is_essential = False
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1:
                            is_essential = True
                            break
                    if not is_essential:
                        # Remove
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(j)
                        current_energy -= 1.0
                        changed = True
                        break # Restart loop as indices changed

            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)

            # Cool down
            T *= cooling_rate

        return best_blocks

    # Run initial greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Try randomized greedy if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 20.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        for seed in seeds:
            if time.time() >= deadline - 10.0:
                break
            b_idx = greedy_cover(seed)
            b_idx = reduce_blocks(b_idx)
            if b_idx and len(b_idx) < best_len:
                best_blocks = b_idx
                best_len = len(b_idx)

    # Run Simulated Annealing on the best solution found
    if best_blocks and time.time() < deadline - 5.0:
        sa_result = simulated_annealing(best_blocks, deadline - 5.0)
        if len(sa_result) < best_len:
            best_blocks = sa_result
            best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
