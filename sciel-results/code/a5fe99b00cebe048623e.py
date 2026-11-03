"""Improved covering design C(19,7,4) using bitset-accelerated greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter

# Precompute all 4-subsets of 19 points as bitmasks for fast coverage tracking
# There are C(19,4) = 3876 such subsets
_ALL_TSUBS = list(itertools.combinations(range(19), 4))
_TSUB_TO_IDX = {ts: i for i, ts in enumerate(_ALL_TSUBS)}
_NUM_TSUBS = len(_ALL_TSUBS)

# Precompute for each 7-subset, the list of t-subset indices it covers
# C(19,7) = 50388 blocks, each covers C(7,4)=35 t-subsets
# We'll build this lazily or precompute for speed

def _build_block_tsub_maps(v, k, t):
    """Map each k-subset (as frozenset) to list of t-subset indices it covers."""
    tsub_indices = {}
    all_ksubs = list(itertools.combinations(range(v), k))
    for b in all_ksubs:
        idxs = []
        for ts in itertools.combinations(b, t):
            idxs.append(_TSUB_TO_IDX[tuple(ts)])
        tsub_indices[frozenset(b)] = idxs
    return tsub_indices

def _greedy_cover(v, k, t, tsub_to_block_idx, rng=None, max_blocks=None, deadline=None):
    """
    Greedy covering: repeatedly pick the block that covers the most uncovered t-subsets.
    Uses precomputed tsub_to_block_idx: for each t-subset index, list of block frozensets covering it.
    """
    if rng is None:
        rng = random.Random()
    
    # uncovered: set of t-subset indices not yet covered
    uncovered = set(range(_NUM_TSUBS))
    blocks = []
    block_set = set()
    
    # Precompute: for each t-subset index, which blocks cover it
    # tsub_to_block_idx[tidx] = list of frozenset blocks
    
    # For efficiency, maintain a count of how many uncovered t-subsets each candidate block covers
    # But that's expensive to update. Instead, use a scoring approach:
    # For each candidate block, score = number of uncovered t-subsets it covers.
    # We need to compute this efficiently.
    
    # Alternative: for each uncovered t-subset, add +1 to score of each block that covers it.
    # Maintain a score array indexed by block index.
    
    all_ksubs = list(itertools.combinations(range(v), k))
    block_idx = {b: i for i, b in enumerate(all_ksubs)}
    num_blocks = len(all_ksubs)
    
    # For each t-subset, list of block indices covering it
    tsub_covering_blocks = [[] for _ in range(_NUM_TSUBS)]
    for b in all_ksubs:
        bi = block_idx[b]
        for ts in itertools.combinations(b, t):
            tsub_covering_blocks[_TSUB_TO_IDX[tuple(ts)]].append(bi)
    
    scores = [0] * num_blocks
    for tidx in uncovered:
        for bi in tsub_covering_blocks[tidx]:
            scores[bi] += 1
    
    while uncovered:
        if deadline is not None and time.time() > deadline:
            break
        if max_blocks is not None and len(blocks) >= max_blocks:
            break
        
        # Find block with max score among those not already used
        best_score = -1
        best_bi = -1
        for bi in range(num_blocks):
            if bi not in block_set and scores[bi] > best_score:
                best_score = scores[bi]
                best_bi = bi
        
        if best_bi == -1:
            break
        
        block = list(all_ksubs[best_bi])
        block.sort()
        blocks.append(block)
        block_set.add(best_bi)
        
        # Update uncovered and scores
        for tidx in tsub_covering_blocks[best_bi]:
            pass  # We need to know which t-subsets this block covers
        
        # Actually, let's recompute: for the chosen block, find all t-subsets it covers
        for ts in itertools.combinations(block, t):
            tidx = _TSUB_TO_IDX[tuple(ts)]
            if tidx in uncovered:
                uncovered.discard(tidx)
        
        # Update scores: for each t-subset newly covered, decrement scores of all blocks covering it
        # This is expensive. Let's use a different approach.
        
    return blocks


def _greedy_cover_v2(v, k, t, deadline=None, rng=None):
    """
    Faster greedy: maintain uncovered as a set of t-subset indices.
    For scoring, use the fact that a block's gain = number of its t-subsets that are in uncovered.
    """
    if rng is None:
        rng = random.Random()
    
    uncovered = set(range(_NUM_TSUBS))
    blocks = []
    
    all_ksubs = list(itertools.combinations(range(v), k))
    num_blocks = len(all_ksubs)
    
    # Precompute: for each block index, list of t-subset indices it covers
    block_tsubs = []
    for b in all_ksubs:
        idxs = [_TSUB_TO_IDX[tuple(ts)] for ts in itertools.combinations(b, t)]
        block_tsubs.append(idxs)
    
    # Precompute: for each t-subset index, list of block indices that cover it
    tsub_blocks = [[] for _ in range(_NUM_TSUBS)]
    for bi, idxs in enumerate(block_tsubs):
        for tidx in idxs:
            tsub_blocks[tidx].append(bi)
    
    # Score: number of uncovered t-subsets covered by each block
    scores = [0] * num_blocks
    for tidx in uncovered:
        for bi in tsub_blocks[tidx]:
            scores[bi] += 1
    
    used = [False] * num_blocks
    
    while uncovered:
        if deadline is not None and time.time() > deadline:
            break
        
        # Find best block
        best_score = -1
        best_bi = -1
        for bi in range(num_blocks):
            if not used[bi] and scores[bi] > best_score:
                best_score = scores[bi]
                best_bi = bi
        
        if best_bi == -1 or best_score == 0:
            break
        
        block = list(all_ksubs[best_bi])
        block.sort()
        blocks.append(block)
        used[best_bi] = True
        
        # Update: for each t-subset covered by this block that is in uncovered, remove it
        for tidx in block_tsubs[best_bi]:
            if tidx in uncovered:
                uncovered.discard(tidx)
                # Decrement scores of all blocks covering this t-subset
                for bi2 in tsub_blocks[tidx]:
                    if not used[bi2]:
                        scores[bi2] -= 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline=None):
    """Remove blocks that are redundant (all their t-subsets are covered by other blocks)."""
    if not blocks:
        return blocks
    
    n = len(blocks)
    # For each t-subset, count how many blocks cover it
    # A block is redundant if every t-subset it covers is covered by at least one other block
    
    # Map t-subset -> list of block indices
    tsub_to_blocks = {}
    for bi, block in enumerate(blocks):
        for ts in itertools.combinations(block, t):
            tsub_to_blocks.setdefault(tuple(ts), []).append(bi)
    
    # Check each block for redundancy
    kept = []
    for bi, block in enumerate(blocks):
        if deadline is not None and time.time() > deadline:
            kept.append(block)
            continue
        
        redundant = True
        for ts in itertools.combinations(block, t):
            covering = tsub_to_blocks[tuple(ts)]
            if len(covering) == 1:  # Only this block covers it
                redundant = False
                break
        if not redundant:
            kept.append(block)
    
    return kept


def _local_search(blocks, v, k, t, deadline=None, rng=None):
    """
    Local search: try to swap a block with another that covers more uncovered t-subsets.
    Actually, since we already have a valid cover, we try to reduce block count.
    Strategy: iteratively try removing a block; if the cover becomes invalid, try to fix by
    adding a block that covers the most newly-uncovered t-subsets. If the added block is
    different and we end up with same or fewer blocks, accept.
    """
    if not blocks or rng is None:
        return blocks
    
    # Current coverage: for each t-subset, count of blocks covering it
    def compute_coverage(blks):
        cov = Counter()
        for block in blks:
            for ts in itertools.combinations(block, t):
                cov[tuple(ts)] += 1
        return cov
    
    cov = compute_coverage(blocks)
    
    # Try to remove each block; if removal causes some t-subset to have count 0,
    # we need to add a block that covers those t-subsets.
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(len(blocks)):
            if deadline is not None and time.time() > deadline:
                break
            
            # Temporarily remove block i
            block_i = blocks[i]
            temp_blocks = blocks[:i] + blocks[i+1:]
            
            # Find t-subsets that would become uncovered
            newly_uncovered = []
            for ts in itertools.combinations(block_i, t):
                if cov[tuple(ts)] == 1:  # Only block_i covers it
                    newly_uncovered.append(tuple(ts))
            
            if not newly_uncovered:
                # Block is redundant
                blocks = temp_blocks
                cov = compute_coverage(blocks)
                improved = True
                break
            
            # Try to find a block that covers all newly_uncovered t-subsets
            # or at least covers the most of them
            best_block = None
            best_covered = -1
            for candidate in itertools.combinations(range(v), k):
                cset = set(candidate)
                covered = sum(1 for ts in newly_uncovered if set(ts).issubset(cset))
                if covered > best_covered:
                    best_covered = covered
                    best_block = candidate
                if covered == len(newly_uncovered):
                    break
            
            if best_block is not None and best_covered == len(newly_uncovered):
                # We can replace block_i with best_block
                if tuple(sorted(best_block)) != tuple(sorted(block_i)):
                    blocks = temp_blocks + [sorted(best_block)]
                    cov = compute_coverage(blocks)
                    improved = True
                    break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute t-subset to index mapping for this specific v, t
    # Note: _TSUB_TO_IDX and _ALL_TSUBS are for v=19, t=4.
    # For generality, we should handle any v, k, t. But the task is specifically C(19,7,4).
    # Let's check if v=19, k=7, t=4 and use precomputed, else fallback.
    
    if v != 19 or k != 7 or t != 4:
        # Fallback to simple greedy for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered and time.time() < deadline:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t-1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                block.append(max(candidates, key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks
    
    # For v=19, k=7, t=4: use bitset-accelerated greedy
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Run greedy multiple times with different random seeds / strategies
    # Actually, the greedy is deterministic. Let's just run it once, then do local search.
    
    blocks = _greedy_cover_v2(v, k, t, deadline=deadline, rng=rng)
    
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = blocks[:]
    
    # Local search to try to reduce further
    if time.time() < deadline:
        blocks_ls = _local_search(blocks, v, k, t, deadline=deadline, rng=rng)
        if len(blocks_ls) < best_count:
            best_count = len(blocks_ls)
            best_blocks = blocks_ls[:]
    
    # Try a few more greedy runs with slight variations (e.g., random tie-breaking)
    # For now, just return the best we have
    
    if best_blocks is None:
        # Fallback: shouldn't happen
        return []
    
    return best_blocks
# EVOLVE-BLOCK-END
