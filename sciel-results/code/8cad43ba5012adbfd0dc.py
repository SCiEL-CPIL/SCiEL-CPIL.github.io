import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 110

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute t-subsets for a block? No, we compute on the fly as it's small (56)
    
    # Precompute combinations of points for completion
    # We need to choose k-t points from v-t points
    # We will generate these on the fly for a specific seed
    
    def get_block_tsub_indices(block):
        """Return list of t-subset indices covered by this block"""
        idxs = []
        # Optimization: only check if tsub is in tsub_to_idx (it always is)
        for ts in itertools.combinations(block, t):
            idxs.append(tsub_to_idx[ts])
        return idxs

    def greedy_seed_completion(uncovered, rng, deadline):
        """
        Greedy algorithm that picks an uncovered t-subset as a seed,
        then chooses the best k-block containing that seed to cover the most uncovered t-subsets.
        """
        blocks = []
        uncovered_set = set(uncovered)
        
        # Precompute list of all points for random selection if needed
        points = list(range(v))
        
        while uncovered_set and time.time() < deadline - 5:
            # 1. Pick a seed t-subset from uncovered
            # Heuristic: Pick one that is "hardest" to cover? 
            # For simplicity and speed, pick random to allow diversity across restarts.
            # Or pick the one with minimum coverage? We don't track coverage count easily here.
            # Let's pick random.
            seed_list = list(uncovered_set)
            if not seed_list:
                break
            seed_idx = rng.choice(seed_list)
            seed = all_tsubs[seed_idx]
            seed_set = set(seed)
            
            # 2. Generate all completions
            # We need k-t more points from v-t available points
            available_points = [p for p in range(v) if p not in seed_set]
            k_rem = k - t
            
            best_block = None
            best_gain = -1
            best_tsubs = None
            
            # If k_rem is small, iterate all combinations.
            # For v=19, k=8, t=5, k_rem=3, available=14. C(14,3)=364.
            # This is small enough to iterate all.
            for combo in itertools.combinations(available_points, k_rem):
                block = list(seed) + list(combo)
                block.sort()
                
                # Calculate gain: number of uncovered t-subsets in this block
                # We need to check the 56 t-subsets of this block against uncovered_set
                gain = 0
                tsubs_in_block = []
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx[ts]
                    tsubs_in_block.append(idx)
                    if idx in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_tsubs = tsubs_in_block
            
            if best_block is None:
                break
            
            # 3. Add best block
            blocks.append(best_block)
            
            # 4. Update uncovered
            for idx in best_tsubs:
                uncovered_set.discard(idx)
                
        return blocks

    def remove_redundant(blocks, deadline):
        """Remove blocks that don't cover any unique t-subset"""
        if not blocks:
            return blocks

        # Map each block to its t-subset indices
        block_tsub_indices = [get_block_tsub_indices(b) for b in blocks]
        
        # Count coverage
        cover_count = defaultdict(int)
        for idxs in block_tsub_indices:
            for idx in idxs:
                cover_count[idx] += 1

        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                is_redundant = True
                for idx in block_tsub_indices[i]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_tsub_indices[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsub_indices.pop(i)
                    changed = True
                    break

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts
    # The seed-completion greedy is deterministic if seed selection is deterministic.
    # Randomizing the seed selection allows us to find different coverings.
    max_restarts = 50 # Limit restarts to ensure time safety
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(42 + restart)
        # Start with all t-subsets uncovered
        initial_uncovered = set(range(n_tsubs))
        
        blocks = greedy_seed_completion(initial_uncovered, rng, deadline)
        
        if not blocks:
            continue
            
        # Verify coverage
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        
        if len(covered) != n_tsubs:
            continue
            
        # Post-processing
        blocks = remove_redundant(blocks, deadline)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 330: # Early exit if we beat known best significantly
                break

    if best_blocks is None:
        # Fallback: Simple deterministic greedy (seed = min uncovered)
        rng = random.Random(0)
        initial_uncovered = set(range(n_tsubs))
        best_blocks = greedy_seed_completion(initial_uncovered, rng, deadline)
        best_blocks = remove_redundant(best_blocks, deadline)

    return best_blocks
