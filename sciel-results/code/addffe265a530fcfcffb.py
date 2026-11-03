"""Improved covering design C(24,7,3) using optimized greedy with redundancy removal."""
import itertools
import time
from collections import defaultdict


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using greedy + redundancy removal.
    
    Strategy:
    1. Greedy: repeatedly pick the block that covers the most uncovered t-subsets.
       To speed up, we use a two-phase block construction:
       - Start from an uncovered t-subset
       - Grow the block by adding points that maximize new coverage
    2. After greedy completion, remove redundant blocks.
    3. Use time budget to do multiple restarts with different random seeds if time permits.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets for quick lookup
    all_tsubsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    num_tsubsets = len(all_tsubsets)
    
    # We'll track coverage as a bitset-like array for speed
    # covered[i] = number of blocks covering t-subset i
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple random restarts
    import random
    rng = random.Random(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Greedy construction with slight randomization
        uncovered_count = [1] * num_tsubsets  # count of uncovered (starts at 1 meaning uncovered)
        # Actually, let's track which t-subsets are covered
        covered = [0] * num_tsubsets  # coverage count
        blocks = []
        
        # For efficiency, track uncovered t-subsets per point pair etc.
        # Use a different approach: maintain list of uncovered t-subset indices
        uncovered = set(range(num_tsubsets))
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered t-subset to start (or the first one for determinism)
            if len(uncovered) > 1000:
                # Pick randomly to add diversity across restarts
                start_idx = rng.choice(list(uncovered))
            else:
                start_idx = min(uncovered)
            
            start_ts = all_tsubsets[start_idx]
            block = set(start_ts)
            
            # Grow the block greedily
            while len(block) < k and time.time() < deadline:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    # A t-subset is covered if it contains t-1 elements from block and x
                    gain = 0
                    # For each (t-1)-subset of block, check if (subset + x) is uncovered
                    for rest in itertools.combinations(sorted(block), t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        idx = tset_to_idx.get(candidate, -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and gain > 0:
                        # Tie-break randomly for diversity
                        if rng.random() < 0.3:
                            best_point = x
                if best_point == -1 or best_gain <= 0:
                    # No point adds coverage; just pick a random remaining point
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        best_point = rng.choice(remaining)
                    else:
                        break
                block.add(best_point)
            
            if len(block) < k:
                break
            
            block_sorted = sorted(block)
            blocks.append(block_sorted)
            
            # Update coverage
            for ts in itertools.combinations(block_sorted, t):
                idx = tset_to_idx[ts]
                if idx in uncovered:
                    uncovered.discard(idx)
                    covered[idx] = 1
                else:
                    covered[idx] += 1
        
        if uncovered:
            # Incomplete; skip this attempt
            continue
        
        # Redundancy removal
        # Remove blocks that don't exclusively cover any t-subset
        # A block is redundant if for every t-subset it covers, another block also covers it
        # Efficient approach: for each block, check if all its t-subsets are covered by other blocks
        blocks_set = set(tuple(b) for b in blocks)
        n_blocks = len(blocks)
        keep = [True] * n_blocks
        
        # For each block, count how many t-subsets are uniquely covered by it
        # This is expensive, so do it smartly
        # Build: for each t-subset index, which blocks cover it
        # But that's O(n_blocks * C(k,t)) which is large
        
        # Simpler: iterate and try to remove each block
        # For block i, it's redundant if for every t-subset in block i,
        # there exists another kept block j != i that also covers that t-subset
        
        # Optimize: for each t-subset, maintain a count of covering blocks
        # If we remove block i, a t-subset becomes uncovered only if count was 1 and block i was the only one
        
        # Let's do iterative removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Recompute coverage counts
            cov_count = [0] * num_tsubsets
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    idx = tset_to_idx[ts]
                    cov_count[idx] += 1
            
            # Try to remove each block
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                b = blocks[i]
                # Check if any t-subset in b has cov_count == 1 (only this block covers it)
                is_redundant = True
                for ts in itertools.combinations(b, t):
                    idx = tset_to_idx[ts]
                    if cov_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    blocks.pop(i)
                    changed = True
                    break  # Restart the loop since indices changed
        
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        
        # If we've found a good solution, we can stop early
        # Known best is 76, lower bound 69
        if best_count <= 76:
            break
        
        # Limit number of attempts to avoid excessive time
        if attempts >= 20:
            break
    
    if best_blocks is None:
        # Fallback: should not happen, but return something
        return []
    
    return best_blocks
