"""
Covering design C(19,8,5): find few 8-subsets of 19 points covering all 5-subsets.
Strategy: greedy set cover with efficient gain computation, followed by local search
to reduce block count.
"""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = 145  # seconds, leave 5s margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    
    # Map each t-subset to an index
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute for each k-subset (block), the list of t-subset indices it covers
    # This is too large to precompute fully (C(19,8) = 75582 blocks), so we compute on demand.
    
    def get_covered_t_indices(block):
        """Return list of t-subset indices covered by this block."""
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(t_sub_index[ts])
        return covered
    
    # Greedy set cover
    def greedy_cover():
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            best_gain = -1
            best_block = None
            # Try adding each point to build a block greedily
            # Start with the first uncovered t-subset as seed
            seed = list(min(uncovered))
            block = list(seed)
            while len(block) < k:
                best_x = None
                best_x_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in t_sub_index:
                            idx = t_sub_index[ts]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_x_gain:
                        best_x_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            # Remove covered t-subsets
            covered = get_covered_t_indices(block)
            uncovered.difference_update(covered)
            if time.time() - start_time > DEADLINE:
                break
        return blocks, uncovered
    
    # Run greedy
    blocks, uncovered = greedy_cover()
    
    if not uncovered:
        return blocks
    
    # Local search: try to remove or replace blocks to reduce count
    # We'll use a simple approach: try to merge/replace blocks
    
    def count_uncovered(blocks_list):
        """Count uncovered t-subsets."""
        covered = set()
        for b in blocks_list:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        return len(all_t_subs) - len(covered)
    
    def is_valid(blocks_list):
        covered = set()
        for b in blocks_list:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        return len(covered) == n_t
    
    current_blocks = list(blocks)
    current_count = len(current_blocks)
    
    # Local search: try removing a block and re-covering with minimal additions
    def try_remove_and_recover(blks, deadline):
        """Try to remove one block and see if we can recover coverage with fewer net blocks."""
        n = len(blks)
        if n <= 1:
            return blks
        # Compute coverage count for each t-subset
        cov_count = [0] * n_t
        for b in blks:
            for idx in get_covered_t_indices(b):
                cov_count[idx] += 1
        
        for i in range(n):
            if time.time() - start_time > deadline:
                break
            # Check if removing block i leaves any t-subset uncovered
            removed = get_covered_t_indices(blks[i])
            uncovered_after = [idx for idx in removed if cov_count[idx] == 1]
            if not uncovered_after:
                # Block i is redundant
                new_blks = blks[:i] + blks[i+1:]
                return new_blks
            # Try to find a block that covers all uncovered_after
            # This is expensive; skip for now
        return blks
    
    # Iterate local search
    for iteration in range(200):
        if time.time() - start_time > DEADLINE:
            break
        old_len = len(current_blocks)
        current_blocks = try_remove_and_recover(current_blocks, DEADLINE)
        if len(current_blocks) < old_len:
            continue
        # Try swapping: remove one block, add a different one that helps
        break  # Simple version: just do removal
    
    # Verify and return
    if is_valid(current_blocks):
        return current_blocks
    else:
        # Fallback to original greedy result if it was valid
        if is_valid(blocks):
            return blocks
        # If even that's invalid (shouldn't happen), return what we have
        return current_blocks
