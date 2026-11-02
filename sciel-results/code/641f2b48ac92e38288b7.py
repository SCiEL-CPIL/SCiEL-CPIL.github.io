"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter


def _greedy_cover(v, k, t, deadline, rng=None):
    """Greedy covering with random tie-breaking. Returns list of blocks or None if timeout."""
    if rng is None:
        rng = random.Random()
    points = list(range(v))
    # Use a set of frozensets for uncovered t-subsets
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        if time.time() > deadline:
            return None
        # Pick a random uncovered t-subset to seed the block
        seed = rng.choice(list(uncovered))
        block = list(seed)
        # Greedily add points
        while len(block) < k:
            if time.time() > deadline:
                return None
            block_set = set(block)
            best_gain = -1
            best_candidates = []
            for x in points:
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    if (tuple(sorted(rest + (x,))) in uncovered):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(x)
            if best_gain <= 0:
                # No point helps; pick randomly from remaining
                remaining = [x for x in points if x not in block_set]
                if remaining:
                    block.append(rng.choice(remaining))
                else:
                    break
            else:
                block.append(rng.choice(best_candidates))
        block.sort()
        blocks.append(block)
        # Remove covered t-subsets
        covered = set(itertools.combinations(block, t))
        uncovered.difference_update(covered)
    return blocks


def _delete_redundant_blocks(blocks, v, t, deadline):
    """Remove blocks whose removal still leaves all t-subsets covered."""
    if not blocks:
        return blocks
    # Build coverage: for each t-subset, which block indices cover it
    # To save memory/time, track counts per block
    # First, compute how many t-subsets each block uniquely covers
    # Simpler: try removing each block one at a time and check if still valid
    
    # Build a mapping: t-subset -> set of block indices that contain it
    # This can be expensive; instead, use a counting approach
    # Count coverage of each t-subset by all blocks
    cover_count = {}
    for bi, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            cover_count[sub] = cover_count.get(sub, 0) + 1
    
    # A block can be removed if every t-subset it covers is also covered by another block
    # i.e., for every t-subset in this block, cover_count[sub] > 1
    # But we need to be careful: if we remove multiple blocks, counts change.
    # Do iterative removal: repeatedly find a removable block and remove it.
    
    # For efficiency, track which t-subsets have count > 1
    removable_indices = []
    for bi, block in enumerate(blocks):
        removable = True
        for sub in itertools.combinations(block, t):
            if cover_count.get(sub, 0) <= 1:
                removable = False
                break
        if removable:
            removable_indices.append(bi)
    
    if not removable_indices:
        return blocks
    
    # Remove the first removable block, update counts, repeat
    removed = set()
    while True:
        if time.time() > deadline:
            break
        found = False
        for bi in removable_indices:
            if bi in removed:
                continue
            block = blocks[bi]
            removable = True
            for sub in itertools.combinations(block, t):
                if cover_count.get(sub, 0) <= 1:
                    removable = False
                    break
            if removable:
                # Remove this block
                for sub in itertools.combinations(block, t):
                    cover_count[sub] -= 1
                removed.add(bi)
                found = True
                break
        if not found:
            break
    
    if removed:
        blocks = [b for i, b in enumerate(blocks) if i not in removed]
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    # Safety margin: stop 5 seconds before the 150s limit
    deadline = start_time + 145.0
    
    # Phase 1: Run multiple greedy restarts, keep the best
    best_blocks = None
    best_len = float('inf')
    
    # Number of restarts depends on time available; do at least 1, more if time permits
    # Estimate: first greedy run gives a sense of timing
    rng = random.Random(42)
    
    # First run: deterministic-ish (seeded)
    result = _greedy_cover(v, k, t, deadline, rng)
    if result is not None:
        best_blocks = result
        best_len = len(result)
    
    # Try additional restarts if time permits
    # Estimate time per restart
    elapsed = time.time() - start_time
    if best_blocks is not None and elapsed > 0:
        # Try to do more restarts within remaining time
        remaining = deadline - time.time()
        est_per_restart = elapsed / 1.0  # one restart done so far
        # Leave some time for deletion phase
        deletion_budget = max(5.0, est_per_restart * 0.5)
        restart_budget = max(0, remaining - deletion_budget)
        max_restarts = int(restart_budget / est_per_restart) if est_per_restart > 0 else 0
        max_restarts = min(max_restarts, 20)  # cap to avoid too many
    else:
        max_restarts = 0
    
    for i in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        rng2 = random.Random(42 + i + 1)
        result = _greedy_cover(v, k, t, deadline - 5.0, rng2)
        if result is not None:
            if len(result) < best_len:
                best_blocks = result
                best_len = len(result)
    
    # Phase 2: Deletion phase to remove redundant blocks
    if best_blocks is not None:
        if time.time() < deadline - 2.0:
            best_blocks = _delete_redundant_blocks(best_blocks, v, t, deadline - 2.0)
    
    # Safety: if we somehow have no blocks (shouldn't happen), fall back to a quick greedy
    if best_blocks is None:
        # Last resort: simple greedy with tight deadline
        best_blocks = _greedy_cover(v, k, t, deadline - 1.0, random.Random(0))
        if best_blocks is None:
            # Extremely unlikely; return empty (invalid but avoids crash)
            return []
    
    return best_blocks
# EVOLVE-BLOCK-END
