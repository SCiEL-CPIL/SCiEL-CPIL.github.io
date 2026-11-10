"""Stronger covering design C(19,7,4) solver.

Strategy:
1. Fast greedy construction with hard-subset prioritization and incremental coverage counting.
2. Redundant block removal.
3. Local search: try to remove blocks, and replace blocks to enable further removals.
4. Deadline guard to return best result before timeout.
"""

import itertools
import time
import random
from collections import defaultdict, Counter


def _all_t_subsets(v, t):
    """Generate all t-subsets of {0,...,v-1}."""
    return set(itertools.combinations(range(v), t))


def _block_subs(block, t):
    """Return the set of t-subsets covered by a block."""
    return set(itertools.combinations(block, t))


def _greedy_construct(v, k, t, deadline):
    """Greedy covering algorithm with hard-subset prioritization and fast gain computation."""
    all_subs = _all_t_subsets(v, t)
    uncovered = set(all_subs)
    blocks = []
    
    # Track point frequency to identify "hard" subsets
    point_freq = [0] * v
    
    while uncovered and time.time() < deadline:
        # Select the "hardest" uncovered subset: prefer subsets with rarely-used points
        # Score = sum of (1/freq) for points in subset, higher = harder
        best_sub = None
        best_score = -1
        # Sample a few candidates for speed, but include the min for determinism
        candidates = [min(uncovered)]
        if len(uncovered) > 10:
            sample_size = min(20, len(uncovered))
            candidates.extend(random.sample(list(uncovered), sample_size))
        
        for sub in candidates:
            score = 0.0
            for p in sub:
                score += 1.0 / (1 + point_freq[p])
            if score > best_score:
                best_score = score
                best_sub = sub
        
        if best_sub is None:
            best_sub = min(uncovered)
        
        block = list(best_sub)
        block_set = set(block)
        
        # Greedily add points with tiebreaking by point frequency (prefer rare points)
        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1
            best_freq = float('inf')
            
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: number of uncovered t-subsets containing x and (t-1) from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        gain += 1
                # Tiebreak: prefer points with lower frequency (harder to cover)
                if gain > best_gain or (gain == best_gain and point_freq[x] < best_freq):
                    best_gain = gain
                    best_freq = point_freq[x]
                    best_x = x
            
            if best_x == -1:
                break
            block.append(best_x)
            block_set.add(best_x)
            point_freq[best_x] += 1
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered and point frequencies
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
        
    return blocks, uncovered


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks

    # Count how many blocks cover each t-subset
    sub_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1

    # A block is redundant if every t-subset it covers is also covered by another block
    kept = []
    for block in blocks:
        redundant = True
        for sub in itertools.combinations(block, t):
            if sub_count[sub] == 1:
                redundant = False
                break
        if not redundant:
            kept.append(block)

    return kept


def _local_search(blocks, v, k, t, deadline):
    """Try to improve the covering by removing or replacing blocks."""
    all_subs = _all_t_subsets(v, t)
    total_subs = len(all_subs)

    # Count coverage
    sub_count = Counter()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1

    # First, remove any redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            needed = False
            for sub in itertools.combinations(block, t):
                if sub_count[sub] == 1:
                    needed = True
                    break
            if not needed:
                for sub in itertools.combinations(block, t):
                    sub_count[sub] -= 1
                blocks.pop(i)
                changed = True
                break

    # Now try to replace blocks to enable further removals
    random.seed(42)
    improved = True
    max_replacements = 2000
    replacements = 0

    while improved and time.time() < deadline and replacements < max_replacements:
        improved = False
        replacements += 1

        # Try removing each block and see if coverage is still complete
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            block = blocks[i]
            needed = False
            for sub in itertools.combinations(block, t):
                if sub_count[sub] == 1:
                    needed = True
                    break
            if not needed:
                for sub in itertools.combinations(block, t):
                    sub_count[sub] -= 1
                blocks.pop(i)
                improved = True
                break

        if improved:
            continue

        # Try replacing a block with a better one
        if len(blocks) > 0:
            # Pick a random block to try replacing
            idx = random.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            old_subs = _block_subs(old_block, t)

            # Count how many of old_block's subsets are uniquely covered by it
            unique_count = 0
            for sub in old_subs:
                if sub_count[sub] == 1:
                    unique_count += 1

            # Generate several random candidate blocks
            best_candidate = None
            best_score = -1

            for _ in range(20):
                new_block = sorted(random.sample(range(v), k))
                new_subs = _block_subs(new_block, t)

                # Compute score: how many new subsets does this cover that aren't already covered?
                # And how many unique subsets would we lose?
                gained = 0
                for sub in new_subs:
                    if sub_count[sub] == 0:
                        gained += 1

                lost = 0
                for sub in old_subs:
                    if sub_count[sub] == 1 and sub not in new_subs:
                        lost += 1

                score = gained - lost * 10  # Heavily penalize losing unique coverage

                if score > best_score:
                    best_score = score
                    best_candidate = new_block

            if best_candidate is not None and best_score > 0:
                # Apply the replacement
                for sub in old_subs:
                    sub_count[sub] -= 1
                for sub in _block_subs(best_candidate, t):
                    sub_count[sub] += 1
                blocks[idx] = best_candidate
                improved = True

    return blocks


def _pairwise_swap_search(blocks, v, k, t, deadline):
    """Try to improve the covering by swapping pairs of blocks."""
    all_subs = _all_t_subsets(v, t)
    total_subs = len(all_subs)

    # Count coverage
    sub_count = Counter()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1

    # First, remove any redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            needed = False
            for sub in itertools.combinations(block, t):
                if sub_count[sub] == 1:
                    needed = True
                    break
            if not needed:
                for sub in itertools.combinations(block, t):
                    sub_count[sub] -= 1
                blocks.pop(i)
                changed = True
                break

    # Now try pairwise swaps
    random.seed(123)
    improved = True
    max_swaps = 1000
    swaps = 0

    while improved and time.time() < deadline and swaps < max_swaps:
        improved = False
        swaps += 1

        # Try removing each block and see if coverage is still complete
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            block = blocks[i]
            needed = False
            for sub in itertools.combinations(block, t):
                if sub_count[sub] == 1:
                    needed = True
                    break
            if not needed:
                for sub in itertools.combinations(block, t):
                    sub_count[sub] -= 1
                blocks.pop(i)
                improved = True
                break

        if improved:
            continue

        # Try pairwise swaps
        if len(blocks) > 1:
            i = random.randint(0, len(blocks) - 1)
            j = random.randint(0, len(blocks) - 1)
            if i == j:
                continue

            old_block_i = blocks[i]
            old_block_j = blocks[j]
            old_subs_i = _block_subs(old_block_i, t)
            old_subs_j = _block_subs(old_block_j, t)

            # Generate random candidate blocks for both positions
            best_candidate_i = None
            best_candidate_j = None
            best_score = -1

            for _ in range(10):
                new_block_i = sorted(random.sample(range(v), k))
                new_block_j = sorted(random.sample(range(v), k))
                new_subs_i = _block_subs(new_block_i, t)
                new_subs_j = _block_subs(new_block_j, t)

                # Compute score: how many new subsets does this cover that aren't already covered?
                # And how many unique subsets would we lose?
                gained = 0
                for sub in new_subs_i | new_subs_j:
                    if sub_count[sub] == 0:
                        gained += 1

                lost = 0
                for sub in old_subs_i:
                    if sub_count[sub] == 1 and sub not in new_subs_i and sub not in new_subs_j:
                        lost += 1
                for sub in old_subs_j:
                    if sub_count[sub] == 1 and sub not in new_subs_i and sub not in new_subs_j:
                        lost += 1

                score = gained - lost * 10  # Heavily penalize losing unique coverage

                if score > best_score:
                    best_score = score
                    best_candidate_i = new_block_i
                    best_candidate_j = new_block_j

            if best_candidate_i is not None and best_candidate_j is not None and best_score > 0:
                # Apply the replacement
                for sub in old_subs_i:
                    sub_count[sub] -= 1
                for sub in old_subs_j:
                    sub_count[sub] -= 1
                for sub in _block_subs(best_candidate_i, t):
                    sub_count[sub] += 1
                for sub in _block_subs(best_candidate_j, t):
                    sub_count[sub] += 1
                blocks[i] = best_candidate_i
                blocks[j] = best_candidate_j
                improved = True

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    # Seed random for reproducibility
    random.seed(42)

    # Phase 1: Greedy construction with hard-subset prioritization
    blocks, uncovered = _greedy_construct(v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 3: Local search - try to find better blocks
    blocks = _local_search(blocks, v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 4: Pairwise swap search
    blocks = _pairwise_swap_search(blocks, v, k, t, deadline)

    return blocks
