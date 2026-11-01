"""Covering design C(19,8,5) - bitmask-accelerated randomized greedy with multi-restart."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    total_tsubsets = sum(1 for _ in itertools.combinations(range(v), t))
    
    # Precompute all t-subsets as bitmasks
    all_tsubsets = [0] * total_tsubsets
    idx = 0
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_tsubsets[idx] = mask
        idx += 1
    
    # Map bitmask to index for fast lookup
    mask_to_idx = {m: i for i, m in enumerate(all_tsubsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        # Bitmask for uncovered t-subsets (1 = uncovered)
        uncovered_mask = (1 << total_tsubsets) - 1
        uncovered_count = total_tsubsets
        blocks = []
        
        while uncovered_count > 0:
            if time.time() >= deadline:
                break
            
            # Find an uncovered t-subset to start a new block
            # Pick a random uncovered one for diversification
            while True:
                r = random.getrandbits(32)
                if r >= total_tsubsets:
                    r = r % total_tsubsets
                if (uncovered_mask >> r) & 1:
                    break
            
            start_tsub = all_tsubsets[r]
            
            # Greedily build a block: start with points from start_tsub, then add best-gain points
            block = list(range(v))
            random.shuffle(block)
            
            # Initialize with the t points from start_tsub
            start_points = [i for i in range(v) if (start_tsub >> i) & 1]
            remaining = [x for x in block if x not in start_points]
            current_block = list(start_points)
            
            # Precompute which t-subsets are covered by current_block
            # We'll track this incrementally
            
            while len(current_block) < k:
                if time.time() >= deadline:
                    break
                
                # For each candidate point, compute how many new t-subsets it covers
                best_gain = -1
                best_point = None
                
                # Compute t-1 subsets of current_block for gain calculation
                t_minus_1_combos = list(itertools.combinations(current_block, t - 1))
                
                for x in remaining:
                    gain = 0
                    for rest in t_minus_1_combos:
                        # Form the t-subset from rest + x
                        mask = 0
                        for y in rest:
                            mask |= (1 << y)
                        mask |= (1 << x)
                        # Check if this t-subset is uncovered
                        sub_idx = mask_to_idx.get(mask, -1)
                        if sub_idx != -1 and (uncovered_mask >> sub_idx) & 1:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None or best_gain == 0:
                    # Pick a random remaining point
                    best_point = random.choice(remaining)
                
                current_block.append(best_point)
                remaining.remove(best_point)
            
            # Now current_block has k points. Compute all t-subsets it covers and mark them covered.
            block_mask = 0
            for x in current_block:
                block_mask |= (1 << x)
            
            newly_covered = 0
            for comb in itertools.combinations(current_block, t):
                mask = 0
                for y in comb:
                    mask |= (1 << y)
                sub_idx = mask_to_idx.get(mask, -1)
                if sub_idx != -1 and (uncovered_mask >> sub_idx) & 1:
                    uncovered_mask ^= (1 << sub_idx)
                    newly_covered += 1
            
            uncovered_count -= newly_covered
            blocks.append(sorted(current_block))
        
        if uncovered_count == 0:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
        
        if best_blocks is not None:
            # Check if we can improve; if we've found a good solution, keep searching
            # but with a tighter time budget
            if len(blocks) <= best_count + 2:
                # Close to best, keep searching
                pass
            else:
                # This restart was bad, continue
                pass
    
    if best_blocks is None:
        # Fallback: run deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
