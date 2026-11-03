import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(range(v), t))
    total_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each k-block, which t-subsets it covers
    # For efficiency, we'll precompute the list of t-subset indices covered by each block
    # But there are C(19,8) = 75582 blocks, each covering C(8,5)=56 t-subsets
    # Total: 75582 * 56 = 4.2M entries, manageable
    
    # Actually, let's be smarter. For the greedy, we need to know for a given
    # partial block, how many NEW t-subsets it would cover.
    # We'll track uncovered t-subsets as a boolean array.
    
    # Precompute: for each pair of points, which t-subsets contain both
    # This helps in incremental gain calculation
    # Actually, for t=5, adding a point p to a block with (t-1)=4 existing points
    # covers t-subsets that contain p and exactly 4 of the existing points
    # That's C(4,4)=1 per 4-subset of existing points... wait no.
    
    # If we have a partial block B of size s < k, and we add point p,
    # the new t-subsets covered are those that contain p and (t-1) other points from B.
    # So we need to count how many (t-1)-subsets of B form t-subsets with p that are uncovered.
    
    # For efficiency, let's precompute for each point p and each (t-1)-subset S,
    # whether the t-subset S ∪ {p} is uncovered. We'll maintain this dynamically.
    
    # Let's use a different approach: maintain uncovered as a set of frozensets
    # and for gain calculation, check membership.
    
    uncovered = set(all_tsubsets)
    blocks = []
    point_usage = [0] * v
    
    # Precompute for each point, the list of (t-1)-subsets that with this point form a t-subset
    # And map t-subset to index for fast uncovered check
    # For point p, the (t-1)-subsets S such that S ∪ {p} is a t-subset
    # There are C(v-1, t-1) such subsets for each point
    
    # For faster gain calculation, let's precompute for each point p,
    # a dictionary mapping (t-1)-subset (as tuple of sorted points excluding p) to the t-subset
    point_to_tsubsets_with = {}
    for p in range(v):
        point_to_tsubsets_with[p] = {}
        for rest in itertools.combinations([x for x in range(v) if x != p], t - 1):
            ts = tuple(sorted(rest + (p,)))
            point_to_tsubsets_with[p][tuple(sorted(rest))] = ts
    
    # For even faster access, let's create a set of uncovered t-subsets for O(1) lookup
    # We'll use frozensets for the uncovered set
    
    def greedy_step(v, k, t, uncovered, point_usage, rng, urg_w=1.0, rar_w=0.3):
        """One greedy step: pick the best block to add."""
        # Pick a random uncovered t-subset as a seed
        if not uncovered:
            return None
        
        if len(uncovered) > 200:
            seed_ts = rng.choice(tuple(uncovered))
        else:
            seed_ts = min(uncovered)
        
        block_points = set(seed_ts)
        
        # Greedily add points to maximize new coverage
        while len(block_points) < k:
            best_gain = -1
            best_point = None
            
            for p in range(v):
                if p in block_points:
                    continue
                
                # Calculate gain: how many new t-subsets does adding p cover?
                gain = 0
                # t-subsets containing p and (t-1) points from block_points
                bp_list = list(block_points)
                for rest in itertools.combinations(bp_list, t - 1):
                    candidate = tuple(sorted(rest + (p,)))
                    if candidate in uncovered:
                        gain += 1
                
                # Rarity bonus
                rarity_bonus = (v - point_usage[p]) * rar_w / v
                total_score = gain * urg_w + rarity_bonus
                total_score += rng.random() * 0.001
                
                if total_score > best_gain:
                    best_gain = total_score
                    best_point = p
            
            if best_point is None:
                break
            block_points.add(best_point)
        
        block = sorted(block_points)
        return block
    
    # Main greedy loop
    rng = random.Random(42)
    
    while uncovered and time.time() < deadline:
        block = greedy_step(v, k, t, uncovered, point_usage, rng)
        if block is None:
            break
        
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            uncovered.discard(tuple(sorted(ts)))
        
        # Update point usage
        for p in block:
            point_usage[p] += 1
    
    if not uncovered:
        # Try to prune
        blocks = prune_blocks_fast(blocks, v, k, t, time.time(), deadline)
        return blocks
    
    # If we didn't finish, return what we have (it may not be valid)
    # Check if it's valid
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(tuple(sorted(ts)))
    
    if covered == set(all_tsubsets):
        blocks = prune_blocks_fast(blocks, v, k, t, time.time(), deadline)
        return blocks
    else:
        # Fallback: simple greedy that always works
        return simple_greedy(v, k, t)


def simple_greedy(v, k, t):
    """Simple greedy that always produces a valid covering."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        # Pick the t-subset with the fewest possible extensions
        # (i.e., the one that appears in the fewest k-blocks)
        # For simplicity, just pick the first uncovered
        seed_ts = min(uncovered)
        
        # Find a k-block containing seed_ts that covers the most new t-subsets
        remaining_points = [p for p in range(v) if p not in seed_ts]
        needed = k - t
        
        best_block = None
        best_gain = -1
        
        for ext in itertools.combinations(remaining_points, needed):
            candidate = tuple(sorted(seed_ts + ext))
            gain = 0
            for ts in itertools.combinations(candidate, t):
                if ts in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = candidate
        
        if best_block is None:
            # Should not happen
            break
        
        blocks.append(list(best_block))
        
        for ts in itertools.combinations(best_block, t):
            uncovered.discard(tuple(sorted(ts)))
    
    return blocks


def prune_blocks_fast(blocks, v, k, t, start_time, deadline):
    """Fast pruning of redundant blocks."""
    if not blocks:
        return blocks
    
    n = len(blocks)
    if n < 2:
        return blocks
    
    # For each t-subset, find which blocks cover it
    tsub_to_blocks = defaultdict(list)
    for i, block in enumerate(blocks):
        for ts in itertools.combinations(block, t):
            tsub_to_blocks[tuple(sorted(ts))].append(i)
    
    # A block is redundant if every t-subset it covers is also covered by at least one other block
    result = list(blocks)
    removed = set()
    
    # Try to remove blocks, checking if they're redundant
    # Process in random order to break symmetry
    indices = list(range(n))
    random.shuffle(indices)
    
    for i in indices:
        if i in removed:
            continue
        if time.time() > deadline - 10:
            break
        
        block = blocks[i]
        redundant = True
        for ts in itertools.combinations(block, t):
            ts_tuple = tuple(sorted(ts))
            covering_blocks = tsub_to_blocks.get(ts_tuple, [])
            # Check if any OTHER block covers this t-subset
            other_cover = False
            for j in covering_blocks:
                if j != i and j not in removed:
                    other_cover = True
                    break
            if not other_cover:
                redundant = False
                break
        
        if redundant:
            removed.add(i)
    
    if removed:
        result = [b for i, b in enumerate(blocks) if i not in removed]
    
    return result
