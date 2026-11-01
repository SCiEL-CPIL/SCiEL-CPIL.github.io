"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _greedy_cover(v, k, t, rng=None):
    """Fast greedy set-cover using bitmask-based gain computation."""
    if rng is None:
        rng = random.Random()
    # Precompute all t-subsets as frozensets for O(1) lookup
    all_tsubs = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            best_x = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets are completed by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = frozenset(rest + (x,))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block
        for ts in itertools.combinations(block, t):
            uncovered.discard(frozenset(ts))
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Local search: removal + swap to reduce block count."""
    blocks = [list(b) for b in blocks]
    
    def build_coverage(blocks):
        coverage = {}
        for bi, block in enumerate(blocks):
            for ts in itertools.combinations(block, t):
                ts_key = tuple(sorted(ts))
                if ts_key not in coverage:
                    coverage[ts_key] = set()
                coverage[ts_key].add(bi)
        return coverage

    coverage = build_coverage(blocks)
    
    def try_remove(blocks, coverage, bi, deadline):
        """Try to remove block bi. Returns (success, new_blocks, new_coverage)."""
        newly_uncovered = []
        for ts in itertools.combinations(blocks[bi], t):
            ts_key = tuple(sorted(ts))
            cov = coverage.get(ts_key, set())
            if len(cov) == 1 and bi in cov:
                newly_uncovered.append(ts_key)
        
        if not newly_uncovered:
            # Safe removal
            removed = blocks.pop(bi)
            for ts in itertools.combinations(removed, t):
                ts_key = tuple(sorted(ts))
                if ts_key in coverage:
                    coverage[ts_key].discard(bi)
                    if not coverage[ts_key]:
                        del coverage[ts_key]
            # Re-index
            for ts_key in coverage:
                new_set = set()
                for idx in coverage[ts_key]:
                    if idx < bi:
                        new_set.add(idx)
                    else:
                        new_set.add(idx - 1)
                coverage[ts_key] = new_set
            return True, blocks, coverage
        return False, blocks, coverage

    def try_swap(blocks, coverage, bi, v, k, t, deadline, rng):
        """Try to replace block bi with a better block. Returns (success, new_blocks, new_coverage)."""
        # Find t-subsets uniquely covered by block bi
        unique_to_bi = []
        for ts in itertools.combinations(blocks[bi], t):
            ts_key = tuple(sorted(ts))
            cov = coverage.get(ts_key, set())
            if len(cov) == 1 and bi in cov:
                unique_to_bi.append(ts_key)
        
        if not unique_to_bi:
            # Block is redundant, just remove it
            return try_remove(blocks, coverage, bi, deadline)
        
        # Try to find a replacement block that covers all unique_to_bi
        # and potentially more
        # Use greedy: seed with a unique t-subset, build block to cover as many unique as possible
        best_new_block = None
        best_new_gain = 0
        
        for seed_ts in unique_to_bi[:min(5, len(unique_to_bi))]:
            candidate = list(seed_ts)
            cand_set = set(candidate)
            while len(candidate) < k:
                best_g = -1
                best_x = -1
                for x in range(v):
                    if x in cand_set:
                        continue
                    g = 0
                    for rest in itertools.combinations(candidate, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in unique_to_bi:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x == -1:
                    break
                candidate.append(best_x)
                cand_set.add(best_x)
            
            if len(candidate) == k:
                # Count how many unique_to_bi this covers
                covered_count = 0
                for ts in unique_to_bi:
                    if set(ts).issubset(set(candidate)):
                        covered_count += 1
                if covered_count > best_new_gain:
                    best_new_gain = covered_count
                    best_new_block = sorted(candidate)
        
        if best_new_block is None or best_new_gain < len(unique_to_bi):
            return False, blocks, coverage
        
        # Check if the new block covers all unique subsets
        new_block_set = set(best_new_block)
        all_covered = all(set(ts).issubset(new_block_set) for ts in unique_to_bi)
        if not all_covered:
            return False, blocks, coverage
        
        # Replace block bi with best_new_block
        old_block = blocks[bi]
        blocks[bi] = best_new_block
        
        # Update coverage
        for ts in itertools.combinations(old_block, t):
            ts_key = tuple(sorted(ts))
            if ts_key in coverage:
                coverage[ts_key].discard(bi)
                if not coverage[ts_key]:
                    del coverage[ts_key]
        
        for ts in itertools.combinations(best_new_block, t):
            ts_key = tuple(sorted(ts))
            if ts_key not in coverage:
                coverage[ts_key] = set()
            coverage[ts_key].add(bi)
        
        return True, blocks, coverage

    improved = True
    rng = random.Random(123)
    while improved and time.time() < deadline:
        improved = False
        # Try removals first
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            success, blocks, coverage = try_remove(blocks, coverage, bi, deadline)
            if success:
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            success, blocks, coverage = try_swap(blocks, coverage, bi, v, k, t, deadline, rng)
            if success:
                improved = True
                break
    
    return blocks


def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover with randomization, multiple restarts
    best_blocks = None
    best_count = float('inf')
    
    # First greedy run
    blocks = _greedy_cover(v, k, t, rng)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Try more greedy restarts if time permits
    restarts = 0
    while time.time() < deadline - 40 and restarts < 8:
        blocks = _greedy_cover(v, k, t, rng)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Local search to reduce block count
    if time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, deadline - 2)

    return best_blocks
# EVOLVE-BLOCK-END
