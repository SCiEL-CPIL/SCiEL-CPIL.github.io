# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def _precompute(v, t):
    """Precompute all t-subsets and per-point t-subset lists."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_set = set(all_tsubs)
    # For each point, list of t-subsets containing it (as tuples)
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        for x in ts:
            point_tsubs[x].append(ts)
    return all_tsubs, tsub_set, point_tsubs


def _greedy_build(v, k, t, rng, all_tsubs, tsub_set, point_tsubs):
    """
    Greedy covering: repeatedly pick the block that covers the most uncovered t-subsets.
    Uses randomized tie-breaking.
    """
    uncovered = set(all_tsubs)
    blocks = []
    n_uncovered = len(uncovered)

    while uncovered:
        # For each candidate block (we can't enumerate all C(v,k)), use a heuristic:
        # Start from a random uncovered t-subset, then greedily extend.
        # But a better approach: score each point by how many uncovered t-subsets it's in,
        # then build a block from top-scoring points.
        
        # Compute point scores: number of uncovered t-subsets containing each point
        # We can do this incrementally, but for simplicity recompute each iteration
        # Actually, let's use a smarter approach:
        
        # Pick a random uncovered t-subset as a seed
        seed_sub = rng.choice(list(uncovered))
        
        # Build block starting with seed_sub, then greedily add points
        block = list(seed_sub)
        block_set = set(block)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Gain = number of uncovered t-subsets that contain x and t-1 elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain == 0:
                # Pick random point
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_x = rng.choice(candidates)
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    
    return blocks


def _greedy_build_v2(v, k, t, rng, all_tsubs, tsub_set, point_tsubs):
    """
    Improved greedy: use point scoring to build blocks.
    At each step, pick the block that maximizes new coverage.
    Approximate by scoring points and building from top points.
    """
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        # Score each point by number of uncovered t-subsets it appears in
        point_scores = [0] * v
        for sub in uncovered:
            for x in sub:
                point_scores[x] += 1
        
        # Sort points by score descending, with random tie-breaking
        indexed = list(range(v))
        indexed.sort(key=lambda x: (-point_scores[x], rng.random()))
        
        # Take top k points as the block
        block = sorted(indexed[:k])
        blocks.append(block)
        
        # Remove covered t-subsets
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    
    return blocks


def _local_search_remove(v, k, t, blocks, deadline, rng):
    """
    Try to remove blocks that are redundant (all their t-subsets are covered by other blocks).
    """
    if not blocks:
        return blocks
    
    n_blocks = len(blocks)
    # Build cover index: for each t-subset, which blocks cover it
    cover = defaultdict(list)
    for bi, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            cover[sub].append(bi)
    
    active = set(range(n_blocks))
    removed = set()
    
    # Iterate: try to remove each block
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        # Shuffle order
        order = list(active)
        rng.shuffle(order)
        for bi in order:
            if bi in removed:
                continue
            if time.time() > deadline - 5.0:
                break
            # Check if block bi is redundant: all its t-subsets are covered by some other active block
            redundant = True
            for sub in itertools.combinations(blocks[bi], t):
                covs = cover.get(sub, [])
                has_other = False
                for c in covs:
                    if c != bi and c in active:
                        has_other = True
                        break
                if not has_other:
                    redundant = False
                    break
            if redundant:
                removed.add(bi)
                active.discard(bi)
                changed = True
    
    result = [blocks[i] for i in range(n_blocks) if i not in removed]
    return result


def _local_search_replace(v, k, t, blocks, deadline, rng):
    """
    Try to replace blocks with better ones. For each block, try to find a replacement
    that covers all the unique t-subsets of the original and possibly more.
    """
    if not blocks:
        return blocks
    
    n_blocks = len(blocks)
    cover = defaultdict(list)
    for bi, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            cover[sub].append(bi)
    
    active = list(range(n_blocks))
    removed = set()
    added = []  # list of new blocks added
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        order = list(active)
        rng.shuffle(order)
        for bi in order:
            if bi in removed:
                continue
            if time.time() > deadline - 5.0:
                break
            
            # Find t-subsets uniquely covered by block bi
            unique_subs = []
            for sub in itertools.combinations(blocks[bi], t):
                covs = cover.get(sub, [])
                active_covs = [c for c in covs if c not in removed]
                if len(active_covs) == 1 and active_covs[0] == bi:
                    unique_subs.append(sub)
            
            if not unique_subs:
                # Can safely remove
                removed.add(bi)
                improved = True
                continue
            
            # Find required points
            required = set()
            for sub in unique_subs:
                required.update(sub)
            if len(required) > k:
                continue
            
            need_more = k - len(required)
            remaining_points = [x for x in range(v) if x not in required]
            
            # Try to find a good completion
            found = False
            for _attempt in range(10):
                if need_more == 0:
                    new_block = sorted(required)
                else:
                    extra = rng.sample(remaining_points, need_more)
                    new_block = sorted(required | set(extra))
                
                # Check if this new block actually covers all unique_subs
                # It should by construction, but let's verify
                block_set = set(new_block)
                ok = True
                for sub in unique_subs:
                    if not all(x in block_set for x in sub):
                        ok = False
                        break
                if not ok:
                    continue
                
                # Add new block, remove old
                new_idx = len(blocks) + len(added)
                added.append(new_block)
                for sub in itertools.combinations(new_block, t):
                    cover[sub].append(new_idx)
                removed.add(bi)
                active.append(new_idx)
                improved = True
                found = True
                break
    
    # Rebuild
    result = [blocks[i] for i in range(n_blocks) if i not in removed]
    result.extend(added)
    return result


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Precompute
    all_tsubs, tsub_set, point_tsubs = _precompute(v, t)
    
    best = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy strategies with different seeds
    strategies = [
        ('greedy_v1', _greedy_build),
        ('greedy_v2', _greedy_build_v2),
    ]
    
    seed = 42
    for strat_name, strat_fn in strategies:
        if time.time() > deadline - 20.0:
            break
        for _ in range(3):  # 3 seeds per strategy
            if time.time() > deadline - 15.0:
                break
            rng = random.Random(seed)
            blocks = strat_fn(v, k, t, rng, all_tsubs, tsub_set, point_tsubs)
            
            # Local search
            if time.time() < deadline - 10.0:
                blocks = _local_search_remove(v, k, t, blocks, deadline, rng)
            if time.time() < deadline - 5.0:
                blocks = _local_search_replace(v, k, t, blocks, deadline, rng)
            
            if len(blocks) < best_len:
                best = blocks
                best_len = len(blocks)
            seed += 1
    
    if best is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best = _greedy_build(v, k, t, rng, all_tsubs, tsub_set, point_tsubs)
        best = _local_search_remove(v, k, t, best, deadline, rng)
    
    return best
# EVOLVE-BLOCK-END
