import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # 1. Precompute all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # 3. For each k-subset, compute the list of t-subset indices it covers
    # k_sub_t_idxs[i] is a list of t-subset indices covered by k-subset i
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)

    # 4. Build reverse mapping: for each t-subset, which k-subsets cover it
    # t_sub_to_k_idxs[j] is a list of k-subset indices covering t-subset j
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
            
            scores = gains.astype(np.float64).copy()
            if seed > 0:
                # Add small noise to break ties
                scores += rng.random() * 0.01
            
            best_idx = int(np.argmax(scores))
            
            if scores[best_idx] < 0.5: # Gain is integer, so < 0.5 means 0
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
                T *= cooling_rate
            else:
                break
            
            # Check if we have a better solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)
                
            # Break if we've reached a perfect solution (no uncovered, minimal size)
            # We can't know minimal size, but if no uncovered, we just try to remove more.
            if np.sum(cover_count == 0) == 0:
                 # We have a valid covering. We can continue to try to reduce size.
                 pass
            else:
                 # If we have uncovered, we might want to add blocks? 
                 # The problem requires a valid covering. 
                 # If we accept invalid moves, we must ensure we return a valid one.
                 # So we track the best *valid* solution separately?
                 # Or we just ensure that if we have uncovered, we don't update best_blocks.
                 if np.sum(cover_count == 0) == 0:
                     if current_energy < best_energy:
                         best_energy = current_energy
                         best_blocks = list(current_blocks)

        return best_blocks

    # --- Main Logic ---
    
    # 1. Get an initial greedy solution
    initial_blocks = greedy_cover(0)
    
    # 2. Reduce redundant blocks
    reduced_blocks = reduce_blocks(initial_blocks)
    
    # 3. Try to improve with Simulated Annealing
    # We have a time budget. Let's reserve 10s for the final return.
    # So we have ~125s for SA.
    sa_end_time = start_time + 125.0
    improved_blocks = simulated_annealing(reduced_blocks, sa_end_time)
    
    # 4. Final reduction
    final_blocks = reduce_blocks(improved_blocks)
    
    # 5. Convert indices to blocks
    result_blocks = [list(all_k_subs[i]) for i in final_blocks]
    
    return result_blocks
