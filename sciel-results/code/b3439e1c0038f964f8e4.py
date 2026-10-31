"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Track which t-subsets are covered by how many blocks
    covered_count = [0] * n_tsubs
    
    def get_uncovered_tsubs():
        """Return list of indices of uncovered t-subsets."""
        return [i for i in range(n_tsubs) if covered_count[i] == 0]
    
    def is_covered(tsub):
        return covered_count[tsub_to_idx[tsub]] > 0
    
    def add_block(block):
        """Add a block and update coverage counts."""
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            covered_count[idx] += 1
    
    def remove_block(block):
        """Remove a block and update coverage counts."""
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            covered_count[idx] -= 1
    
    def count_uncovered():
        return sum(1 for c in covered_count if c == 0)
    
    # Phase 1: Greedy construction with improved gain calculation
    # Precompute for each point x and each (t-1)-subset of points, the t-subset index
    # This allows O(1) lookup for gain calculation
    
    blocks = []
    
    while True:
        if time.time() > deadline:
            break
            
        # Find all uncovered t-subsets
        uncovered_indices = [i for i in range(n_tsubs) if covered_count[i] == 0]
        if not uncovered_indices:
            break
        
        # Select target: pick uncovered t-subset with highest "difficulty"
        # Difficulty heuristic: number of points that could extend it
        # For simplicity, just pick the first uncovered one (or use frequency)
        target_idx = uncovered_indices[0]
        target = all_tsubs[target_idx]
        
        # Initialize block with the target elements
        block = list(target)
        block_set = set(block)
        
        # Fill the block to size k using greedy gain
        while len(block) < k:
            best_x = -1
            best_gain = -1
            
            # Candidates are points not in block
            for x in range(v):
                if x in block_set:
                    continue
                
                # Gain: number of uncovered t-subsets S such that x in S and S \ {x} ⊆ block
                # S = x ∪ combo where combo is a (t-1)-subset of block
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    cand = tuple(sorted(combo + (x,)))
                    idx = tsub_to_idx.get(cand)
                    if idx is not None and covered_count[idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                # Fallback: pick any available point
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        add_block(block)
        
        # Check if we're out of time
        if time.time() > deadline:
            break
    
    # Phase 2: Local search - try to remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    if time.time() < deadline - 10:
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                # Check if block i is redundant
                redundant = True
                for sub in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[sub]
                    if covered_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(blocks[i])
                    blocks.pop(i)
                    changed = True
    
    # Phase 3: Local search - try to improve blocks by swapping elements
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for bi in range(len(blocks)):
                if time.time() > deadline - 2:
                    break
                block = blocks[bi]
                block_set = set(block)
                
                # Try removing each element and replacing with another
                for ri in range(k):
                    if time.time() > deadline - 2:
                        break
                    old_elem = block[ri]
                    # Try each replacement
                    best_new_elem = -1
                    best_gain = -1
                    
                    for x in range(v):
                        if x in block_set:
                            continue
                        
                        new_block_set = block_set - {old_elem} | {x}
                        
                        # Calculate how many uncovered t-subsets would be covered
                        # by this new block that aren't covered by others
                        gain = 0
                        for sub in itertools.combinations(sorted(list(new_block_set)), t):
                            idx = tsub_to_idx[sub]
                            if covered_count[idx] <= 1:  # Only covered by this block
                                gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_new_elem = x
                    
                    # If this swap doesn't make any t-subset lose coverage, do it
                    # Actually, we need to be careful: we want to maintain coverage
                    # A swap is valid if all t-subsets of the new block are covered
                    # by at least one block (possibly the modified block itself)
                    
                    # For now, skip complex swaps; the redundancy removal is key
    
    return blocks
# EVOLVE-BLOCK-END
