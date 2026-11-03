"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a randomized greedy construction
    followed by local search (simulated annealing style) to minimize block count.
    
    Strategy:
    1. Build an initial covering with randomized greedy (multiple restarts, keep best).
    2. Local search: try to remove a block if all its t-subsets are still covered.
       Try to swap a block with a better one (fewer unique t-subsets, or enabling removals).
    3. Time-bounded: return best valid covering found before deadline.
    """
    deadline = time.time() + 110  # safety margin before 120s kill
    
    # Precompute all t-subsets for fast lookup
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_set = set(all_tsubs)
    
    # Helper: check if a block covers certain t-subsets
    def block_tsubs(block):
        return set(itertools.combinations(block, t))
    
    # Helper: count how many t-subsets a block uniquely covers given current coverage
    def unique_count(block, coverage):
        """Return number of t-subsets in block that are ONLY covered by this block."""
        cnt = 0
        for ts in itertools.combinations(block, t):
            if coverage.get(ts, 0) == 1:
                cnt += 1
        return cnt
    
    # Randomized greedy construction
    def greedy_randomized(seed):
        rng = random.Random(seed)
        uncovered = set(all_tsubs)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_ts = rng.choice(list(uncovered))
            block = list(start_ts)
            
            while len(block) < k:
                # For each candidate point, compute how many new t-subsets it would cover
                best_gain = -1
                best_point = None
                # Sample candidates if v is large, else use all
                candidates = [x for x in range(v) if x not in block]
                if len(candidates) > 50:
                    candidates = rng.sample(candidates, 50)
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    best_point = rng.choice(candidates)
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        
        return blocks
    
    # Build initial covering with a few random seeds, keep best
    best_blocks = None
    best_len = float('inf')
    
    # Try a few seeds quickly
    for seed in range(5):
        if time.time() > deadline - 60:
            break
        blocks = greedy_randomized(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        blocks = greedy_randomized(0)
        best_blocks = blocks
        best_len = len(blocks)
    
    # Local search: iteratively try to remove or swap blocks
    # Maintain coverage counts
    def build_coverage(blocks):
        coverage = {}
        for block in blocks:
            for ts in itertools.combinations(block, t):
                coverage[ts] = coverage.get(ts, 0) + 1
        return coverage
    
    def is_valid(coverage):
        return all(c >= 1 for c in coverage.values()) and len(coverage) == len(all_tsubs)
    
    # Local search phase
    current_blocks = list(best_blocks)
    current_coverage = build_coverage(current_blocks)
    best_found = list(current_blocks)
    best_found_len = len(current_blocks)
    
    rng = random.Random(42)
    
    # Iterative improvement: try removing blocks that have no unique t-subsets
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        # Try to remove each block if it has no unique t-subsets
        for i in range(len(current_blocks)):
            if time.time() > deadline - 30:
                break
            block = current_blocks[i]
            # Check if all t-subsets in this block are covered by other blocks
            can_remove = True
            for ts in itertools.combinations(block, t):
                if current_coverage.get(ts, 0) <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove it
                for ts in itertools.combinations(block, t):
                    current_coverage[ts] -= 1
                current_blocks.pop(i)
                improved = True
                if len(current_blocks) < best_found_len:
                    best_found_len = len(current_blocks)
                    best_found = list(current_blocks)
                break  # Restart the loop since indices shifted
    
    # Simulated annealing style: try swapping blocks
    # A swap: replace block i with a new block that covers some unique t-subsets of i
    # and doesn't leave any t-subset uncovered
    for iteration in range(200):
        if time.time() > deadline - 15:
            break
        
        # Try random swaps
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            if len(current_blocks) < 2:
                break
            
            i = rng.randrange(len(current_blocks))
            old_block = current_blocks[i]
            
            # Generate a candidate new block
            # Start from a random t-subset that is uniquely covered by old_block
            unique_tsubs = []
            for ts in itertools.combinations(old_block, t):
                if current_coverage.get(ts, 0) == 1:
                    unique_tsubs.append(ts)
            
            if not unique_tsubs:
                # No unique t-subsets, try to remove
                can_remove = True
                for ts in itertools.combinations(old_block, t):
                    if current_coverage.get(ts, 0) <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in itertools.combinations(old_block, t):
                        current_coverage[ts] -= 1
                    current_blocks.pop(i)
                    if len(current_blocks) < best_found_len:
                        best_found_len = len(current_blocks)
                        best_found = list(current_blocks)
                continue
            
            # Pick a random unique t-subset to preserve
            target_ts = rng.choice(unique_tsubs)
            
            # Build a new block containing target_ts
            new_block = list(target_ts)
            while len(new_block) < k:
                # Add points that help cover other unique t-subsets of old_block
                candidates = [x for x in range(v) if x not in new_block]
                if len(candidates) > 30:
                    candidates = rng.sample(candidates, 30)
                
                best_gain = -1
                best_point = None
                for x in candidates:
                    gain = 0
                    for ts2 in itertools.combinations(new_block, t - 1):
                        ts_tuple = tuple(sorted(ts2 + (x,)))
                        # Count if this helps cover a unique t-subset of old_block
                        if current_coverage.get(ts_tuple, 0) == 1:
                            gain += 2  # Priority for covering unique ones
                        elif current_coverage.get(ts_tuple, 0) == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    best_point = rng.choice(candidates)
                new_block.append(best_point)
            
            new_block.sort()
            
            # Check if swapping is valid: all t-subsets of old_block must still be covered
            # after removing old_block and adding new_block
            # First, tentatively remove old_block
            for ts in itertools.combinations(old_block, t):
                current_coverage[ts] -= 1
            
            # Check validity
            valid_swap = True
            for ts in itertools.combinations(old_block, t):
                if current_coverage.get(ts, 0) < 1:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Apply the new block
                for ts in itertools.combinations(new_block, t):
                    current_coverage[ts] = current_coverage.get(ts, 0) + 1
                current_blocks[i] = new_block
                
                # Check if we can now remove the new block (if it has no unique t-subsets)
                # This is a secondary improvement
                can_remove_new = True
                for ts in itertools.combinations(new_block, t):
                    if current_coverage.get(ts, 0) <= 1:
                        can_remove_new = False
                        break
                if can_remove_new:
                    for ts in itertools.combinations(new_block, t):
                        current_coverage[ts] -= 1
                    current_blocks.pop(i)
                    if len(current_blocks) < best_found_len:
                        best_found_len = len(current_blocks)
                        best_found = list(current_blocks)
                else:
                    if len(current_blocks) < best_found_len:
                        best_found_len = len(current_blocks)
                        best_found = list(current_blocks)
            else:
                # Revert
                for ts in itertools.combinations(old_block, t):
                    current_coverage[ts] += 1
    
    # Final cleanup: try to remove any blocks with no unique t-subsets
    current_coverage = build_coverage(current_blocks)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(current_blocks)):
            if time.time() > deadline - 5:
                break
            block = current_blocks[i]
            can_remove = True
            for ts in itertools.combinations(block, t):
                if current_coverage.get(ts, 0) <= 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in itertools.combinations(block, t):
                    current_coverage[ts] -= 1
                current_blocks.pop(i)
                improved = True
                if len(current_blocks) < best_found_len:
                    best_found_len = len(current_blocks)
                    best_found = list(current_blocks)
                break
    
    # Return the best found
    if len(current_blocks) < best_found_len:
        best_found = list(current_blocks)
    
    return best_found

# EVOLVE-BLOCK-END
