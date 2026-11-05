import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precomputation specific to v=19, k=7, t=4
    if v == 19 and k == 7 and t == 4:
        # Map 4-subsets to indices
        subsets = list(itertools.combinations(range(v), t))
        subset_to_idx = {s: i for i, s in enumerate(subsets)}
        N_SUBSETS = len(subsets)
        
        # Generate all 7-subsets (blocks)
        all_blocks = list(itertools.combinations(range(v), k))
        N_BLOCKS = len(all_blocks)
        
        # Precompute for each block:
        # 1. The list of subset indices it covers
        # 2. A bitmask (optional, but list is faster for counting)
        block_subs = []
        subset_frequencies = np.zeros(N_SUBSETS, dtype=np.int32)
        
        for block in all_blocks:
            subs = []
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                subs.append(idx)
                subset_frequencies[idx] += 1
            block_subs.append(subs)
        
        # Weights for weighted greedy: 1/frequency
        weights = np.zeros(N_SUBSETS, dtype=np.float64)
        # Avoid division by zero (should not happen in this specific problem)
        mask = subset_frequencies > 0
        weights[mask] = 1.0 / subset_frequencies[mask]
        
        def fast_greedy(seed):
            """
            Weighted Greedy Set Cover.
            1. Maintain count of how many times each subset is covered.
            2. Maintain 'uncovered' status.
            3. At each step, pick the block that maximizes the weighted gain 
               (sum of weights of currently uncovered subsets it covers).
            4. Update counts.
            """
            random.seed(seed)
            counts = np.zeros(N_SUBSETS, dtype=np.int32)
            selected_indices = []
            # Track which subsets are still uncovered
            # We can just check counts == 0
            
            # Optimization: Pre-compute block gains dynamically is expensive.
            # Instead, iterate through all blocks and calculate gain.
            # With N_BLOCKS ~ 50388 and N_SUBSETS ~ 3876, and 35 subsets per block,
            # calculating gain for all blocks is 50388 * 35 = ~1.7M ops per step.
            # 150 steps * 1.7M = 255M ops. In Python/NumPy mix, this might be too slow 
            # if done naively in a loop.
            
            # Let's use a vectorized approach for gain calculation if possible,
            # or a heuristic to limit candidates.
            
            # Alternative: Standard Greedy with "most uncovered" heuristic.
            # Pick a random uncovered subset, then find the block covering it 
            # that covers the most *other* uncovered subsets.
            
            uncovered = set(range(N_SUBSETS))
            
            while uncovered:
                # Pick a random uncovered subset to anchor the search
                # This breaks symmetry and speeds up finding a "good" block
                target_subset = random.choice(list(uncovered))
                
                # Find all blocks that contain this target subset
                # We need a reverse index: subset -> list of blocks
                # Let's build it on the fly or precompute?
                # Precomputing reverse index for 3876 subsets * ~500 blocks each is ~2M entries.
                # This is feasible.
                
                # Actually, let's just scan all blocks. It's O(N_BLOCKS * 35).
                # To speed up, we can precompute `blocks_containing_subset`.
                
                # Let's assume we precomputed `blocks_containing_subset` outside.
                # Wait, I didn't in this scope. Let's do it.
                # It's better to precompute outside the loop.
                
                # For now, let's implement the scan.
                best_block_idx = -1
                best_gain = -1
                
                # Heuristic: Only consider blocks that contain the target subset?
                # No, the block MUST contain the target subset to cover it.
                # So we only need to check blocks containing `target_subset`.
                
                # Let's precompute `blocks_containing_subset` in the outer scope.
                # I will add that below.
                
                # For this draft, I'll use the precomputed list.
                candidates = blocks_containing_subset[target_subset]
                
                for b_idx in candidates:
                    # Calculate gain: number of uncovered subsets in this block
                    gain = 0
                    for sub_idx in block_subs[b_idx]:
                        if sub_idx in uncovered:
                            gain += 1
                            # Optional: use weights
                            # gain += weights[sub_idx]
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
                        
                if best_block_idx == -1:
                    break # Should not happen if uncovered is not empty and we have valid blocks
                
                # Add the block
                selected_indices.append(best_block_idx)
                
                # Update uncovered
                for sub_idx in block_subs[best_block_idx]:
                    uncovered.discard(sub_idx)
                    
            return selected_indices
        
        # Precompute reverse index: subset -> list of blocks
        blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
        for b_idx, subs in enumerate(block_subs):
            for sub_idx in subs:
                blocks_containing_subset[sub_idx].append(b_idx)
        
        def local_search(block_indices, max_steps=500):
            """
            Try to remove redundant blocks or swap blocks to reduce count.
            1. Check for redundant blocks (can be removed without uncovering anything).
            2. If none, try to swap: remove 1 block, add 1 block that covers the "lost" subsets 
               and potentially allows removing another? 
               Simpler: Just try to remove blocks.
            """
            counts = np.zeros(N_SUBSETS, dtype=np.int32)
            for b_idx in block_indices:
                for sub_idx in block_subs[b_idx]:
                    counts[sub_idx] += 1
            
            current_set = set(block_indices)
            improved = True
            steps = 0
            
            while improved and steps < max_steps:
                improved = False
                steps += 1
                
                # Try removing each block
                # To speed up, only try blocks that have low "essentiality"
                # A block is redundant if all its subsets have count > 1.
                
                # Shuffle to avoid bias
                indices_to_check = list(current_set)
                random.shuffle(indices_to_check)
                
                for b_idx in indices_to_check:
                    if b_idx not in current_set:
                        continue
                    
                    # Check if redundant
                    is_redundant = True
                    for sub_idx in block_subs[b_idx]:
                        if counts[sub_idx] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        # Remove it
                        for sub_idx in block_subs[b_idx]:
                            counts[sub_idx] -= 1
                        current_set.remove(b_idx)
                        improved = True
            
            return list(current_set)
        
        def hybrid_search(seed):
            """
            1. Greedy to get a solution.
            2. Local search to prune.
            """
            blocks = fast_greedy(seed)
            blocks = local_search(blocks)
            return blocks

        best_blocks = None
        best_count = float('inf')
        
        trial = 0
        while time.time() < deadline:
            blocks = hybrid_search(trial)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # Optional: If we hit a very low number, we might want to spend more time on it?
                # But for now, just keep trying.
            
            trial += 1
            
            # If we are getting close to deadline, ensure we have a valid result
            if best_blocks is None and time.time() > deadline - 5:
                 # Fallback
                 blocks = fast_greedy(0)
                 return [list(all_blocks[i]) for i in blocks]

        if best_blocks is None:
            # Fallback
            blocks = fast_greedy(0)
            return [list(all_blocks[i]) for i in blocks]
            
        return [list(all_blocks[i]) for i in best_blocks]

    else:
        # Fallback for other parameters
        # Simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        
        # Ensure coverage? The above greedy might not cover everything if we run out of subsets?
        # No, we iterate through ALL subsets. So it will cover everything.
        return blocks
