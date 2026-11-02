"""Improved covering design C(19,8,5) with deadline guard and better greedy + local search."""
import itertools
import time
import random
import sys

# Deadline: evaluator kills at 150s; use 140s as safety margin
DEADLINE = 140.0


def _all_t_subsets(v, t):
    """Generate all t-subsets of {0,...,v-1} as tuples."""
    return list(itertools.combinations(range(v), t))


def _blocks_covering_t_subsets(blocks, t):
    """For each block, precompute the set of t-subsets it covers."""
    result = []
    for b in blocks:
        result.append(set(itertools.combinations(b, t)))
    return result


def _uncovered_count(blocks_cover, all_ts):
    """Count uncovered t-subsets given precomputed block covers."""
    covered = set()
    for bc in blocks_cover:
        covered.update(bc)
    return len(all_ts) - len(covered)


def _greedy_construct(v, k, t, deadline, seed=None):
    """
    Greedy covering: repeatedly pick the k-subset that covers the most uncovered t-subsets.
    Uses a smarter approach: maintain uncovered set, for each candidate point set compute gain.
    """
    if seed is not None:
        random.seed(seed)
    
    all_ts = _all_t_subsets(v, t)
    uncovered = set(all_ts)
    blocks = []
    
    # Precompute: for each point, list of t-subsets containing it
    point_to_ts = [[] for _ in range(v)]
    for ts in all_ts:
        for p in ts:
            point_to_ts[p].append(ts)
    
    start_time = time.time()
    
    while uncovered:
        if time.time() - start_time > deadline:
            break
        
        # Greedy: build block one point at a time
        # Start with the point that appears in the most uncovered t-subsets
        # Actually, let's use a different strategy: 
        # Pick a random uncovered t-subset to "anchor" the block, then greedily add points.
        
        # Better: for the first point, pick the one with max uncovered coverage
        # But that's expensive. Let's use: pick an uncovered t-subset, include all its points,
        # then greedily add remaining points to maximize new coverage.
        
        # Pick a random uncovered t-subset as seed
        anchor = random.choice(list(uncovered))
        block = list(anchor)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            if time.time() - start_time > deadline:
                break
            
            best_point = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                # Compute how many new uncovered t-subsets would be covered by adding x
                # A t-subset is covered by block+x if it contains x and (t-1) points from block
                # and is currently uncovered
                gain = 0
                # For efficiency, check t-subsets containing x that have t-1 elements in block
                # This is expensive; approximate by checking all uncovered t-subsets containing x
                for ts in point_to_ts[x]:
                    if ts in uncovered:
                        # Check if ts has at least t-1 elements in block
                        # Actually we need: ts contains x and the other t-1 elements are in block
                        # Since x is in ts, we need the other t-1 elements to be in block
                        rest = [y for y in ts if y != x]
                        if all(y in block for y in rest):
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    return blocks


def _local_search_improve(blocks, v, k, t, deadline, seed=None):
    """
    Try to improve the covering by removing blocks that are redundant,
    or swapping blocks to reduce count.
    """
    if seed is not None:
        random.seed(seed)
    
    all_ts = _all_t_subsets(v, t)
    blocks = list(blocks)
    
    # Precompute coverage for each block
    block_covers = _blocks_covering_t_subsets(blocks, t)
    
    # Build coverage count for each t-subset
    from collections import Counter
    cover_count = Counter()
    for bc in block_covers:
        for ts in bc:
            cover_count[ts] += 1
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        # Try removing each block if it doesn't leave any t-subset uncovered
        for i in range(len(blocks)):
            if time.time() - start_time > deadline:
                break
            # Check if block i is the only cover for any t-subset
            redundant = True
            for ts in block_covers[i]:
                if cover_count[ts] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                blocks.pop(i)
                block_covers.pop(i)
                for ts in block_covers[i] if i < len(block_covers) else []:
                    pass
                # Recompute cover counts
                cover_count = Counter()
                for bc in block_covers:
                    for ts in bc:
                        cover_count[ts] += 1
                improved = True
                break
    
    return blocks


def _random_greedy_with_lookahead(v, k, t, deadline, seed=None):
    """
    Random greedy with lookahead: at each step, try multiple candidate blocks
    and pick the one that seems to lead to fewer total blocks.
    """
    if seed is not None:
        random.seed(seed)
    
    all_ts = _all_t_subsets(v, t)
    uncovered = set(all_ts)
    blocks = []
    
    start_time = time.time()
    
    while uncovered and time.time() - start_time < deadline:
        # Try several candidate blocks, pick the best
        best_block = None
        best_score = float('inf')
        
        # Generate candidates: random k-subsets biased toward uncovered t-subsets
        candidates = []
        
        # Strategy 1: anchor on random uncovered t-subset
        for _ in range(3):
            if not uncovered:
                break
            anchor = random.choice(list(uncovered))
            block = list(anchor)
            remaining = [x for x in range(v) if x not in block]
            random.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
            block.sort()
            candidates.append(block)
        
        # Strategy 2: pure random
        for _ in range(2):
            block = sorted(random.sample(range(v), k))
            candidates.append(block)
        
        # Score each candidate by how many uncovered t-subsets it covers
        for block in candidates:
            if time.time() - start_time > deadline:
                break
            gain = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
            # Simple score: prefer higher gain
            if gain > best_score:
                best_score = gain
                best_block = block
        
        if best_block is None:
            # Fallback
            if not uncovered:
                break
            anchor = random.choice(list(uncovered))
            block = list(anchor)
            remaining = [x for x in range(v) if x not in block]
            random.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
            best_block = sorted(block)
        
        blocks.append(best_block)
        for ts in itertools.combinations(best_block, t):
            uncovered.discard(ts)
    
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    
    # Run multiple strategies and keep the best
    best_blocks = None
    best_len = float('inf')
    
    # Strategy 1: Random greedy with lookahead (multiple seeds)
    for seed in [42, 123, 7, 999, 555]:
        if time.time() - start_time > DEADLINE - 10:
            break
        blocks = _random_greedy_with_lookahead(v, k, t, DEADLINE - 5, seed=seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Strategy 2: Standard greedy with different anchors
    for seed in [1, 2, 3]:
        if time.time() - start_time > DEADLINE - 5:
            break
        blocks = _greedy_construct(v, k, t, DEADLINE - 3, seed=seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Local search improvement
    if best_blocks is not None and time.time() - start_time < DEADLINE - 2:
        improved = _local_search_improve(best_blocks, v, k, t, DEADLINE - 1, seed=42)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved
    
    # Final safety: if we ran out of time and best_blocks is None, do a quick fallback
    if best_blocks is None:
        # Very quick greedy
        all_ts = _all_t_subsets(v, t)
        uncovered = set(all_ts)
        blocks = []
        while uncovered:
            anchor = min(uncovered)
            block = list(anchor)
            remaining = [x for x in range(v) if x not in block]
            random.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        best_blocks = blocks
    
    # Ensure distinct blocks
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
