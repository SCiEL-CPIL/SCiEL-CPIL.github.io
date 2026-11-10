"""Covering design C(v, k, t) – global set-cover greedy with repair and randomized restarts."""
import itertools
import time
import random
from collections import defaultdict

_DEADLINE_S = 140.0  # Safety margin from 150s evaluator limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _build_point_freq(blocks, v):
    """Count how many times each point appears across all blocks."""
    freq = [0] * v
    for block in blocks:
        for p in block:
            freq[p] += 1
    return freq


def _score_block(block, uncovered, v, k, t):
    """Score a block by number of uncovered t-subsets it covers."""
    score = 0
    for sub in itertools.combinations(block, t):
        if sub in uncovered:
            score += 1
    return score


def _greedy_set_cover(uncovered, v, k, t, rng, max_blocks=None):
    """
    Global set-cover greedy: at each step, evaluate a pool of candidate k-subsets
    and pick the one covering the most uncovered t-subsets.
    Uses randomized candidate generation for diversity.
    """
    blocks = []
    block_set = set()
    point_freq = [0] * v
    max_blocks = max_blocks or 200
    
    # Precompute all t-subsets for quick membership
    # We'll work with the uncovered set directly
    
    for step in range(max_blocks):
        if not uncovered:
            break
        if _time_left() < 1.0:
            break
        
        # Generate candidate blocks: mix of random and "informed" candidates
        candidates = []
        
        # Informed candidates: build around uncovered subsets
        # Sample some uncovered subsets and extend them
        sample_size = min(30, len(uncovered))
        if sample_size > 0:
            sample = rng.sample(list(uncovered), sample_size)
            for target in sample:
                # Extend target to a k-subset by adding points with low frequency
                block = list(target)
                block_set_local = set(block)
                # Add points in order of low frequency (more "available")
                remaining = [x for x in range(v) if x not in block_set_local]
                remaining.sort(key=lambda x: (point_freq[x], rng.random()))
                for x in remaining:
                    if len(block) >= k:
                        break
                    block.append(x)
                    block_set_local.add(x)
                if len(block) == k:
                    candidates.append(tuple(sorted(block)))
        
        # Random candidates
        for _ in range(20):
            block = tuple(sorted(rng.sample(range(v), k)))
            candidates.append(block)
        
        # Deduplicate
        candidates = list(set(candidates))
        
        if not candidates:
            break
        
        # Score each candidate
        best_block = None
        best_score = -1
        for cand in candidates:
            score = 0
            for sub in itertools.combinations(cand, t):
                if sub in uncovered:
                    score += 1
            if score > best_score:
                best_score = score
                best_block = cand
        
        if best_block is None or best_score <= 0:
            # No improvement possible from candidates; add a random block
            best_block = tuple(sorted(rng.sample(range(v), k)))
        
        bkey = best_block
        if bkey in block_set:
            continue
        
        block_set.add(bkey)
        blocks.append(list(bkey))
        
        # Update uncovered
        for sub in itertools.combinations(bkey, t):
            uncovered.discard(sub)
        
        # Update point frequencies
        for p in bkey:
            point_freq[p] += 1
    
    return blocks


def _repair(blocks, uncovered, v, k, t, rng):
    """
    Repair: iteratively add blocks that cover the most uncovered t-subsets.
    Uses a smarter candidate generation based on uncovered subset structure.
    """
    if not uncovered:
        return blocks
    
    block_set = set(tuple(b) for b in blocks)
    point_freq = _build_point_freq(blocks, v)
    
    # Build a map: for each point, how many uncovered t-subsets contain it
    point_uncovered_count = [0] * v
    for sub in uncovered:
        for p in sub:
            point_uncovered_count[p] += 1
    
    max_iter = 500
    for _ in range(max_iter):
        if not uncovered:
            break
        if _time_left() < 1.0:
            break
        
        # Strategy: pick the most "urgent" uncovered subset (one whose points
        # have the highest total uncovered count) and build a block around it.
        # Also consider random uncovered subsets for diversity.
        
        # Score uncovered subsets by urgency
        # Use a sample for efficiency
        if len(uncovered) > 100:
            sample = rng.sample(list(uncovered), 100)
        else:
            sample = list(uncovered)
        
        # Score each sample: sum of point_uncovered_count for its points
        scored = []
        for sub in sample:
            urgency = sum(point_uncovered_count[p] for p in sub)
            scored.append((urgency + rng.random() * 0.1, sub))
        scored.sort(reverse=True)
        
        # Pick from top candidates with randomness
        top_n = min(5, len(scored))
        target = scored[rng.randint(0, top_n - 1)][1]
        
        # Build a block around target
        block = list(target)
        block_set_local = set(block)
        remaining = [x for x in range(v) if x not in block_set_local]
        # Sort by point_uncovered_count (high = covers more uncovered)
        remaining.sort(key=lambda x: (-point_uncovered_count[x], point_freq[x], rng.random()))
        for x in remaining:
            if len(block) >= k:
                break
            block.append(x)
            block_set_local.add(x)
        
        if len(block) < k:
            # Fill with random points
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block_set_local:
                    block.append(x)
                    block_set_local.add(x)
        
        block = tuple(sorted(block))
        if block in block_set:
            continue
        
        # Check if this block actually covers any uncovered subsets
        covers_any = False
        for sub in itertools.combinations(block, t):
            if sub in uncovered:
                covers_any = True
                break
        
        if not covers_any:
            continue
        
        block_set.add(block)
        blocks.append(list(block))
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
        
        # Update point frequencies and uncovered counts
        for p in block:
            point_freq[p] += 1
        for sub in itertools.combinations(block, t):
            # This sub was just covered, so decrement counts for its points
            for p in sub:
                point_uncovered_count[p] -= 1
    
    return blocks


def _local_search(blocks, uncovered, v, k, t, rng, max_moves=200):
    """
    Local search: try to replace a block with a better one that covers
    more uncovered t-subsets (or enables removing another block).
    """
    if not uncovered:
        return blocks
    
    block_set = set(tuple(b) for b in blocks)
    point_freq = _build_point_freq(blocks, v)
    
    for _ in range(max_moves):
        if not uncovered:
            break
        if _time_left() < 1.0:
            break
        
        # Pick a random block to potentially replace
        idx = rng.randint(0, len(blocks) - 1)
        old_block = blocks[idx]
        
        # Try to find a replacement block that covers more uncovered subsets
        best_replacement = None
        best_gain = 0
        
        # Generate candidates by perturbing the old block
        old_set = set(old_block)
        for _ in range(10):
            # Remove 1-2 points, add 1-2 new points
            remove_count = rng.randint(1, 2)
            remove_points = rng.sample(list(old_set), remove_count)
            new_set = old_set - set(remove_points)
            remaining = [x for x in range(v) if x not in new_set]
            add_count = rng.randint(1, 2)
            add_points = rng.sample(remaining, add_count)
            new_set.update(add_points)
            
            if len(new_set) != k:
                continue
            
            cand = tuple(sorted(new_set))
            if cand in block_set:
                continue
            
            # Score: how many uncovered t-subsets does cand cover that old_block doesn't?
            old_covers = set()
            for sub in itertools.combinations(old_block, t):
                old_covers.add(sub)
            
            new_covers = set()
            for sub in itertools.combinations(cand, t):
                new_covers.add(sub)
            
            # Gain: uncovered subsets covered by new but not old
            gain = len((new_covers - old_covers) & uncovered)
            # Penalty: uncovered subsets covered by old but not new
            loss = len((old_covers - new_covers) & uncovered)
            
            net_gain = gain - loss
            if net_gain > best_gain:
                best_gain = net_gain
                best_replacement = cand
        
        if best_replacement is not None and best_gain > 0:
            # Accept the replacement
            block_set.remove(tuple(old_block))
            blocks[idx] = list(best_replacement)
            block_set.add(best_replacement)
            
            # Update uncovered
            # Remove coverage from old block
            for sub in itertools.combinations(old_block, t):
                # Check if any other block still covers sub
                still_covered = False
                for other in blocks:
                    if other == blocks[idx]:
                        continue
                    if sub in set(itertools.combinations(other, t)):
                        still_covered = True
                        break
                if not still_covered:
                    uncovered.add(sub)
            
            # Add coverage from new block
            for sub in itertools.combinations(best_replacement, t):
                uncovered.discard(sub)
            
            # Update point frequencies
            for p in old_block:
                point_freq[p] -= 1
            for p in best_replacement:
                point_freq[p] += 1
    
    return blocks


def construct(v, k, t):
    random.seed(12345)
    all_tsubs = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts with different strategies
    max_restarts = 30
    for r in range(max_restarts):
        if _time_left() < 3.0:
            break
        
        rng = random.Random(12345 + r * 7919)
        uncovered = set(all_tsubs)
        
        # Phase 1: Global set-cover greedy
        blocks = _greedy_set_cover(uncovered, v, k, t, rng, max_blocks=120)
        
        # Phase 2: Repair
        blocks = _repair(blocks, uncovered, v, k, t, rng)
        
        # Phase 3: Local search
        if _time_left() > 5.0:
            blocks = _local_search(blocks, uncovered, v, k, t, rng, max_moves=50)
        
        # Check validity
        if not uncovered:
            if len(blocks) < best_len:
                best_blocks = blocks
                best_len = len(blocks)
                if best_len <= 151:
                    break
        else:
            # Try to finish with a simple greedy on remaining
            if _time_left() > 2.0:
                while uncovered and _time_left() > 0.5:
                    target = min(uncovered)
                    block = list(target)
                    block_set_local = set(block)
                    remaining = [x for x in range(v) if x not in block_set_local]
                    remaining.sort(key=lambda x: rng.random())
                    for x in remaining:
                        if len(block) >= k:
                            break
                        block.append(x)
                        block_set_local.add(x)
                    block = tuple(sorted(block))
                    if block not in set(tuple(b) for b in blocks):
                        blocks.append(list(block))
                        for sub in itertools.combinations(block, t):
                            uncovered.discard(sub)
                
                if not uncovered and len(blocks) < best_len:
                    best_blocks = blocks
                    best_len = len(blocks)
    
    # Fallback
    if best_blocks is None:
        uncovered = set(all_tsubs)
        blocks = []
        rng = random.Random(99999)
        while uncovered and _time_left() > 0.1:
            target = min(uncovered)
            block = list(target)
            block_set_local = set(block)
            remaining = [x for x in range(v) if x not in block_set_local]
            remaining.sort(key=lambda x: rng.random())
            for x in remaining:
                if len(block) >= k:
                    break
                block.append(x)
                block_set_local.add(x)
            block = tuple(sorted(block))
            if block not in set(tuple(b) for b in blocks):
                blocks.append(list(block))
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
        best_blocks = blocks
    
    return best_blocks
