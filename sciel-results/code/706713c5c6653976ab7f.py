# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses a greedy algorithm with multiple random restarts and local improvement.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Precompute all t-subsets for fast lookup
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast array operations
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    def greedy_cover(seed_offset=0, max_blocks=None):
        """
        Greedy covering with random tie-breaking.
        Returns (blocks, time_elapsed)
        """
        t0 = time.time()
        rng = random.Random(42 + seed_offset)
        
        # Track uncovered subsets using a boolean numpy array
        uncovered_mask = np.ones(total_t_subsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered_mask):
            if time.time() > deadline:
                return None, time.time() - t0
            
            # Pick a random uncovered subset to start the block
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
            start_subset = list(all_t_subsets[rand_idx])
            block = list(start_subset)
            
            # Add points one by one to maximize coverage of uncovered subsets
            while len(block) < k:
                if time.time() > deadline:
                    return None, time.time() - t0
                
                best_x = None
                best_gain = -1
                best_gain_secondary = -1
                
                # Precompute (t-1)-subsets of the current block
                block_subsets = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate gain: how many uncovered t-subsets would be covered
                    # A t-subset is covered if it's a subset of block + {x}
                    # i.e., it contains t-1 elements from block and x
                    indices_to_check = []
                    for rest in block_subsets:
                        candidate = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx.get(candidate, -1)
                        if idx != -1:
                            indices_to_check.append(idx)
                    
                    if indices_to_check:
                        # Vectorized sum of uncovered status for these indices
                        gain = int(np.sum(uncovered_mask[indices_to_check]))
                    else:
                        gain = 0
                    
                    # For tie-breaking, use secondary criterion: 
                    # prefer points that appear in fewer existing blocks (to encourage diversity)
                    # or just use random tie-breaking
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_gain_secondary = rng.random()
                    elif gain == best_gain and gain > 0:
                        # Random tie-breaking
                        if rng.random() < 0.5:
                            best_x = x
                            best_gain_secondary = rng.random()
                
                if best_x is None:
                    # Should not happen if k > t and v > k
                    break
                
                block.append(best_x)
            
            if len(block) < k:
                break
            
            block.sort()
            
            # Remove covered subsets from uncovered
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx.get(sub, -1)
                if idx != -1:
                    uncovered_mask[idx] = False
            
            blocks.append(block)
            
            if max_blocks is not None and len(blocks) >= max_blocks:
                break
        
        return blocks, time.time() - t0
    
    # Try multiple restarts and keep the best
    best_blocks = None
    best_len = float('inf')
    
    # Estimate number of restarts based on time
    # Each restart takes some time. We'll just loop until deadline.
    restart_count = 0
    while time.time() < deadline - 5:
        blocks, elapsed = greedy_cover(seed_offset=restart_count)
        restart_count += 1
        
        if blocks is not None:
            # Verify the covering is valid (should be, but just in case)
            # Note: _verify_covering is slow, so we only do it if we have time
            if time.time() < deadline - 2:
                if _verify_covering(blocks, v, k, t):
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = blocks
                        # If we found a good solution, we can try to improve it
                        # with local search in remaining time
                        if time.time() < deadline - 10:
                            improved = _local_improve(blocks, v, k, t, deadline)
                            if improved is not None and len(improved) < len(best_blocks):
                                best_blocks = improved
                                best_len = len(best_blocks)
            else:
                # Assume valid if we ran out of time for verification
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = blocks

    if best_blocks is None:
        # Fallback: use the first greedy result even if not fully verified
        blocks, _ = greedy_cover(seed_offset=0)
        if blocks is not None:
            return blocks
        # Last resort: return a trivial (invalid) result
        return []
    
    return best_blocks


def _verify_covering(blocks, v, k, t):
    """Verify that every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            covered.add(sub)
    
    # Check if all t-subsets are covered
    for sub in itertools.combinations(range(v), t):
        if sub not in covered:
            return False
    return True


def _local_improve(blocks, v, k, t, deadline):
    """
    Try to improve the covering by removing redundant blocks.
    """
    if not blocks:
        return blocks
    
    # First, try to remove redundant blocks
    # A block is redundant if removing it doesn't uncover any t-subset
    blocks_copy = list(blocks)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(blocks_copy)):
            if time.time() > deadline - 5:
                break
            
            # Check if block i is redundant
            without_i = blocks_copy[:i] + blocks_copy[i+1:]
            if _verify_covering(without_i, v, k, t):
                blocks_copy = without_i
                changed = True
                break
    
    return blocks_copy
# EVOLVE-BLOCK-END
