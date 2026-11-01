"""Covering design C(24,7,3) via greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def _greedy_cover(v, k, t, seed=0):
    """Greedy covering with frozenset-based uncovered tracking."""
    rng = random.Random(seed)
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsets)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-set to start the block
        start_tset = rng.choice(list(uncovered))
        block = list(start_tset)
        
        while len(block) < k:
            # For each candidate point, count how many uncovered t-sets it would cover
            best_point = None
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_tset = frozenset(rest) | {x}
                    if new_tset in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        # Remove covered t-sets
        for c in itertools.combinations(block, t):
            uncovered.discard(frozenset(c))
    
    return blocks


def _local_search(v, k, t, blocks, deadline, seed=0):
    """Try to reduce block count by replacing blocks or removing redundant ones."""
    rng = random.Random(seed)
    blocks = [list(b) for b in blocks]
    
    # Build coverage count: which t-set is covered by which block
    # Map from frozenset -> list of block indices
    cover_map = defaultdict(list)
    for i, block in enumerate(blocks):
        for c in itertools.combinations(block, t):
            cover_map[frozenset(c)].append(i)
    
    n_blocks = len(blocks)
    
    # Try removing redundant blocks
    improved = True
    while improved:
        improved = False
        # Shuffle order for diversity
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                return blocks
            # Check if block idx is redundant
            redundant = True
            for c in itertools.combinations(blocks[idx], t):
                fs = frozenset(c)
                # Check if any other block covers this t-set
                for other_idx in cover_map[fs]:
                    if other_idx != idx:
                        redundant = False
                        break
                if not redundant:
                    break
            if redundant:
                # Remove block idx
                del blocks[idx]
                # Rebuild cover_map
                cover_map = defaultdict(list)
                for i, block in enumerate(blocks):
                    for c in itertools.combinations(block, t):
                        cover_map[frozenset(c)].append(i)
                improved = True
                break
    
    # Try replacing a block with a better one
    # For each block, try to find a new block that covers more uncovered t-sets
    # (or at least the same, allowing removal of redundancy)
    attempts = 0
    max_attempts = 200
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        # Pick a random block to try replacing
        i = rng.randrange(len(blocks))
        old_block = blocks[i]
        
        # Find t-sets that are ONLY covered by this block
        exclusive = []
        for c in itertools.combinations(old_block, t):
            fs = frozenset(c)
            if len(cover_map[fs]) == 1 and cover_map[fs][0] == i:
                exclusive.append(fs)
        
        if not exclusive:
            continue
        
        # Try to construct a new block that covers all exclusive t-sets
        # The new block must contain the union of all exclusive t-sets
        union = set()
        for fs in exclusive:
            union.update(fs)
        
        if len(union) > k:
            continue
        
        # Start with the union, fill in remaining points greedily
        new_block = list(union)
        new_block_set = set(new_block)
        
        # Count how many t-sets the new block would cover
        # and how many of those are currently covered only by block i
        # We want to maximize coverage of t-sets not covered by other blocks
        
        # Try different random completions
        best_new_block = None
        best_score = -1
        
        for _ in range(5):
            trial = list(new_block)
            trial_set = set(trial)
            while len(trial) < k:
                candidates = [x for x in range(v) if x not in trial_set]
                rng.shuffle(candidates)
                # Pick point that maximizes coverage of t-sets not exclusively covered by i
                best_p = None
                best_g = -1
                for x in candidates[:10]:  # Limit candidates for speed
                    g = 0
                    for rest in itertools.combinations(trial, t - 1):
                        fs = frozenset(rest) | {x}
                        if fs in cover_map:
                            # Count how many other blocks cover this
                            others = [j for j in cover_map[fs] if j != i]
                            if not others:
                                g += 2  # Double weight for exclusive
                            else:
                                g += 1
                        else:
                            g += 3  # New coverage
                    if g > best_g:
                        best_g = g
                        best_p = x
                if best_p is not None:
                    trial.append(best_p)
                    trial_set.add(best_p)
            
            trial.sort()
            # Score: number of t-sets covered that were exclusively in old block
            score = 0
            for c in itertools.combinations(trial, t):
                fs = frozenset(c)
                if fs in exclusive:
                    score += 1
            if score > best_score:
                best_score = score
                best_new_block = trial
        
        if best_new_block is not None and best_score >= len(exclusive):
            # Replace block i
            blocks[i] = best_new_block
            # Rebuild cover_map
            cover_map = defaultdict(list)
            for i2, block in enumerate(blocks):
                for c in itertools.combinations(block, t):
                    cover_map[frozenset(c)].append(i2)
    
    return blocks


def construct(v, k, t):
    deadline = time.time() + 105  # Safety margin before 120s limit
    
    # Try multiple greedy seeds and keep the best
    best_blocks = None
    best_count = float('inf')
    
    for seed in range(10):
        if time.time() > deadline:
            break
        blocks = _greedy_cover(v, k, t, seed=seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        if best_count <= 76:
            break  # Can't do better than known best
    
    if best_blocks is None:
        best_blocks = _greedy_cover(v, k, t, seed=0)
    
    # Local search to try to reduce further
    if time.time() < deadline:
        best_blocks = _local_search(v, k, t, best_blocks, deadline, seed=42)
    
    return best_blocks
# EVOLVE-BLOCK-END
