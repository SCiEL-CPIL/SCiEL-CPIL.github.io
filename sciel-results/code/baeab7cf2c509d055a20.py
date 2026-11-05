import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute the triplets covered by each block (indices into all_triplets)
    # Store as list of lists of Python ints to avoid numpy scalar overflow in bit shifts
    block_triplets = []
    for block in all_blocks:
        tps = []
        for triplet in itertools.combinations(block, t):
            tps.append(triplet_to_id[triplet])
        block_triplets.append(tps)
        
    # Precompute masks for greedy
    # Use Python ints for masks. Cast tid to int to ensure 1 << tid works correctly
    block_masks = [0] * n_blocks
    for b_idx in range(n_blocks):
        mask = 0
        for tid in block_triplets[b_idx]:
            mask |= (1 << int(tid))
        block_masks[b_idx] = mask
        
    # Full mask of all triplets
    all_mask = (1 << n_triplets) - 1

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_blocks = []
        solution_indices = []
        
        while uncovered:
            if time.time() > deadline - 20:
                break
                
            best_gain = -1
            best_idx = -1
            candidates = []
            
            # Find block with max gain
            # Optimization: If we have many candidates, we can break early if gain is max possible?
            # Max gain is len(block_triplets[b_idx]) = 35.
            # If best_gain == 35, we can break? No, there might be multiple 35s.
            # But if we find a 35, it's optimal for this step.
            
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    candidates = [b_idx]
                    # If gain is maximum possible (35), we can't do better, but we should still collect candidates for randomness?
                    # Or just pick one? Randomness helps diversify.
                    if gain == 35:
                        # We can break early if we just want *a* best block, but for diversity we might want all.
                        # However, collecting all 35s is fast.
                        pass
                elif gain == best_gain and gain > 0:
                    candidates.append(b_idx)
                    
            if best_idx == -1 or best_gain == 0:
                break
                
            # Randomly pick from candidates to diversify
            chosen_idx = rng.choice(candidates)
            
            solution_blocks.append(all_blocks[chosen_idx])
            solution_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        return solution_blocks

    def refine_solution(blocks):
        # Convert to list of lists for mutability
        blocks = [list(b) for b in blocks]
        
        # Helper to calculate coverage counts using precomputed block_triplets
        def get_counts(blk_list):
            counts = np.zeros(n_triplets, dtype=np.int32)
            for b in blk_list:
                # Find index of b in all_blocks? 
                # It's faster to just recompute or map.
                # Since blocks are distinct, we can use a set of tuples for O(1) lookup?
                # Or just iterate block_triplets? 
                # We don't have the index.
                # Let's create a map from tuple(block) to index in all_blocks?
                # This is expensive to build.
                # Alternative: Just recompute triplets for b.
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    counts[tid] += 1
            return counts

        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            counts = get_counts(blocks)
            new_blocks = []
            for b in blocks:
                redundant = True
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(b)
                else:
                    changed = True
            blocks = new_blocks
            
            if not blocks:
                break

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds
    num_seeds = 10 # Reduced to fit time
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        blocks = greedy_set_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = greedy_set_cover(0)
        
    # Local search polish
    best_blocks = refine_solution(best_blocks)
    
    return best_blocks

if __name__ == "__main__":
    pass
