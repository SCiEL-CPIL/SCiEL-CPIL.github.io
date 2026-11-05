"""Covering design C(v,k,t) solver with numpy-vectorized greedy and local search."""
import time
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # For each point, precompute which t-subsets contain it
    # point_tsubs[p] = list of indices of t-subsets containing p
    point_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_tsubs):
        for p in sub:
            point_tsubs[p].append(i)
    
    def greedy_cover(uncovered_mask, block_seed=None, max_blocks=None):
        """Greedy covering algorithm using numpy for speed."""
        blocks = []
        uncovered = uncovered_mask.copy()
        
        if max_blocks is not None and max_blocks <= 0:
            return blocks
        
        while np.any(uncovered):
            if max_blocks is not None and len(blocks) >= max_blocks:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Find an uncovered t-subset to anchor on
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick the first uncovered subset (or a random one for diversity)
            if block_seed is not None and len(blocks) == 0:
                anchor_idx = block_seed % len(uncovered_indices)
            else:
                anchor_idx = uncovered_indices[0]
            
            anchor_sub = all_tsubs[anchor_idx]
            block = list(anchor_sub)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                block_set = set(block)
                best_point = -1
                best_gain = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    # Count how many uncovered t-subsets this point would cover
                    # A t-subset is covered by adding p if it contains p and all other t-1 points are in block
                    gain = 0
                    for tsub_idx in point_tsubs[p]:
                        if not uncovered[tsub_idx]:
                            continue
                        tsub = all_tsubs[tsub_idx]
                        # Check if all other points in tsub are in block
                        if all(q in block_set for q in tsub if q != p):
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                uncovered[idx] = False
        
        return blocks
    
    # Run greedy with different seeds to find diverse solutions
    best_blocks = None
    best_len = float('inf')
    
    # First, run a standard greedy
    initial_uncovered = np.ones(n_tsubs, dtype=bool)
    
    for seed in range(20):
        if time.time() > deadline - 10.0:
            break
        
        if seed == 0:
            blocks = greedy_cover(initial_uncovered, block_seed=0)
        else:
            # For subsequent runs, add some randomness
            blocks = greedy_cover(initial_uncovered, block_seed=seed * 7919)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # Local search: try to improve by swapping points
    if best_blocks is not None and time.time() < deadline - 15.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try removing each point from each block and see if coverage is maintained
            for i, block in enumerate(best_blocks):
                if time.time() > deadline - 10.0:
                    break
                for j in range(len(block)):
                    # Try removing block[j]
                    new_block = block[:j] + block[j+1:]
                    if len(new_block) < t:
                        continue
                    
                    # Check if removing this point breaks any coverage
                    # We need to verify that all t-subsets of new_block are still covered
                    # by other blocks
                    removed_point = block[j]
                    # Find t-subsets that contain removed_point and are in this block
                    # These might become uncovered if not covered by other blocks
                    
                    # Quick check: count how many times each t-subset in the modified block appears
                    # Actually, let's just verify full coverage after modification
                    # This is expensive, so let's do a cheaper check first
                    
                    # Check if the removed point was essential for any t-subset
                    # A t-subset is essential if it's only covered by this block
                    # and contains the removed point
                    
                    # For efficiency, just try the swap and verify
                    candidate_blocks = best_blocks[:i] + [new_block] + best_blocks[i+1:]
                    
                    # Verify coverage
                    covered = np.zeros(n_tsubs, dtype=bool)
                    for b in candidate_blocks:
                        for tsub in itertools.combinations(b, t):
                            covered[tsub_to_idx[tsub]] = True
                    
                    if np.all(covered):
                        best_blocks = candidate_blocks
                        improved = True
                        break
                if improved:
                    break
    
    return best_blocks
