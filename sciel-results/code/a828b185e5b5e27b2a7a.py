"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def _build_incidence(v, k, t):
    """
    Build helper structures for fast gain computation.
    For each (t-1)-subset, we track which t-subsets it extends to.
    """
    # Precompute all (t-1)-subsets and the t-subsets they extend to
    # For a partial block P (size m < k), the gain of adding x is the number of
    # uncovered t-subsets that contain x and (t-1) elements from P.
    # i.e., for each (t-1)-subset S of P, and x not in P, the t-subset S ∪ {x}
    # is covered if it's in uncovered.
    
    # We'll use a bitset-like approach with a set of uncovered t-subsets.
    # For speed, we precompute for each point x, the list of (t-1)-subsets containing x? No.
    
    # Actually, for a given partial block P, gain(x) = |{S in C(P, t-1) : S ∪ {x} in uncovered}|
    # This is what the original does. We can speed it up by precomputing, for each point x,
    # a mapping from (t-1)-subset to whether it's in uncovered. But that's the same.
    
    # Alternative: represent uncovered as a set of tuples. The original approach is O(C(m, t-1)) per candidate x.
    # With m up to k-1=6, t-1=3, C(6,3)=20. And v-k candidates ~13. So ~260 lookups per step. That's fast.
    
    pass


def _greedy_cover(v, k, t, seed=0, time_limit=100.0):
    """
    Greedy covering with randomized tie-breaking and local improvement.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() - start_time > time_limit:
            # Return what we have if it's a valid cover, otherwise continue
            # We need to ensure validity. If we exit early, we might not have a valid cover.
            # So we should only exit if uncovered is empty.
            # Actually, we can't exit early unless we have a valid cover.
            # Let's just break and return the partial result only if valid, else continue.
            # For safety, let's not break early in the main loop.
            pass
        
        # Pick a random uncovered t-subset to start
        # The original picks min(uncovered). Let's pick randomly for diversity.
        target = rng.choice(list(uncovered))
        block = list(target)
        
        # Complete the block greedily
        while len(block) < k:
            # Compute gain for each candidate point
            best_gain = -1
            best_candidates = []
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                # Gain = number of uncovered t-subsets that contain x and (t-1) points from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate_t = tuple(sorted(rest + (x,)))
                    if candidate_t in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            
            # Randomly pick among best candidates
            block.append(rng.choice(best_candidates))
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
        
        if not uncovered:
            break
    
    return blocks


def _local_search(blocks, v, k, t, max_iter=10000, time_limit=30.0):
    """
    Local search: try to swap a point in a block with a point not in it,
    if it reduces the total number of uncovered t-subsets (or maintains coverage while allowing removal).
    
    Actually, a simpler local search: try 1-for-1 swaps to see if we can remove a redundant block.
    Or try to improve by swapping points to cover more.
    
    Let's do: for each block, try swapping each point with each other point.
    If the new set of blocks still covers everything and has the same size, accept.
    This doesn't reduce size. 
    
    Better: try to find a block that is redundant (all its t-subsets are covered by other blocks).
    If so, remove it.
    
    And try swaps that might make a block redundant.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage count for each t-subset
    def compute_coverage(blocks):
        cov_count = defaultdict(int)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov_count[ts] += 1
        return cov_count
    
    cov_count = compute_coverage(blocks)
    
    # Check if valid
    all_ts = set(itertools.combinations(range(v), t))
    covered = set(ts for ts, c in cov_count.items() if c > 0)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try to remove a redundant block
        removed = False
        for i in range(len(blocks)):
            # Check if block i is redundant
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                if cov_count[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                del blocks[i]
                # Update coverage
                for ts in itertools.combinations(blocks[i], t):
                    cov_count[ts] -= 1
                    if cov_count[ts] == 0:
                        del cov_count[ts]
                removed = True
                break
        
        if removed:
            continue
        
        # Try 1-for-1 swaps
        swapped = False
        for i in range(len(blocks)):
            for j in range(len(blocks[i])):
                old_point = blocks[i][j]
                for new_point in range(v):
                    if new_point == old_point or new_point in blocks[i]:
                        continue
                    
                    # Simulate swap
                    new_block = blocks[i][:]
                    new_block[j] = new_point
                    new_block.sort()
                    
                    # Check if this swap maintains validity
                    # A swap is valid if all t-subsets of the old block that were only covered by this block
                    # are still covered by the new block or other blocks.
                    
                    # This is complex. Let's just do a full validity check for small v.
                    # For v=19, k=7, t=4, this might be slow.
                    
                    # Alternative: only accept if the new block covers at least as many
                    # uniquely-covered t-subsets as the old block.
                    
                    pass
        
        # Simplified: just try random swaps and check validity
        if len(blocks) > 0:
            i = random.randrange(len(blocks))
            j = random.randrange(len(blocks[i]))
            old_point = blocks[i][j]
            candidates = [x for x in range(v) if x not in blocks[i]]
            if candidates:
                new_point = random.choice(candidates)
                old_block = blocks[i][:]
                blocks[i][j] = new_point
                blocks[i].sort()
                
                # Check validity
                cov_count_new = compute_coverage(blocks)
                valid = all(cov_count_new.get(ts, 0) > 0 for ts in all_ts)
                
                if valid:
                    cov_count = cov_count_new
                    swapped = True
                else:
                    # Revert
                    blocks[i] = old_block
                    blocks[i].sort()
        
        if not swapped and not removed:
            # No improvement, break
            break
    
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different seeds
    for seed in range(20):
        if time.time() > deadline - 30:  # Leave time for local search
            break
        
        blocks = _greedy_cover(v, k, t, seed=seed, time_limit=deadline - time.time() - 10)
        
        if len(blocks) < best_size:
            best_blocks = blocks
            best_size = len(blocks)
    
    # Local search on the best solution
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, 
                                     max_iter=5000, 
                                     time_limit=deadline - time.time() - 2)
    
    return best_blocks if best_blocks is not None else []
# EVOLVE-BLOCK-END
