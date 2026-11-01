"""
Improved greedy covering for C(v, k, t) using NumPy-accelerated set cover
with randomized tie-breaking, redundancy removal, and a deadline guard.
"""
import time
import itertools
import random
import numpy as np

# Global deadline
_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds to spare


def _remaining_time():
    if _DEADLINE is None:
        return float('inf')
    return max(0.0, _DEADLINE - time.time())


def _combo_index(combo, v, t):
    """Map a t-subset (tuple of sorted ints) to a unique integer index."""
    # Use combinatorial number system
    idx = 0
    for i in range(t):
        # C(combo[i], i+1) contribution
        # Precompute not done here for simplicity; use a hash-based approach
        # Actually, let's use a simpler unique encoding:
        # Since v <= 24 and t <= 7, we can use a base-v encoding
        idx = idx * v + combo[i]
    return idx


def _build_index_map(v, t):
    """Build a mapping from t-subset to index, and back."""
    # Use itertools.combinations and assign sequential indices
    combos = list(itertools.combinations(range(v), t))
    n = len(combos)
    combo_to_idx = {}
    for i, c in enumerate(combos):
        combo_to_idx[c] = i
    return combos, combo_to_idx, n


def _block_covers(block, v, t, combo_to_idx):
    """Return the set of indices of t-subsets covered by this block."""
    covered = set()
    for sub in itertools.combinations(block, t):
        idx = combo_to_idx[sub]
        covered.add(idx)
    return covered


def _greedy_cover_fast(v, k, t, seed=0, deadline=None):
    """
    NumPy-accelerated greedy set cover with randomized tie-breaking.
    Returns a list of blocks (each a sorted list of k ints).
    """
    global _DEADLINE
    if deadline is not None:
        _DEADLINE = deadline
    else:
        _DEADLINE = time.time() + 140.0  # safety margin

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Build index mapping
    combos, combo_to_idx, n_total = _build_index_map(v, t)
    n_total = len(combos)

    # uncovered: boolean array, True = still uncovered
    uncovered = np.ones(n_total, dtype=bool)
    remaining = n_total

    blocks = []
    all_blocks_set = set()

    # Precompute for each point, which t-subsets contain it? 
    # Actually, we need for each candidate point x and current partial block,
    # how many new t-subsets would be covered.
    
    # Strategy: build each block by adding one point at a time.
    # For each candidate point, compute the number of currently-uncovered
    # t-subsets that would be covered if we add this point to the current block.
    
    # To speed up: for a block B of size b, adding point x covers all t-subsets
    # that contain x and (t-1) elements from B.
    
    # Precompute: for each point x, a list of t-subset indices that contain x.
    # And for each (t-1)-subset S, the list of points x such that S ∪ {x} is a t-subset.
    
    # Alternative: use the fact that for a block B, the t-subsets covered are
    # all combinations of t elements from B.
    
    # For greedy: at each step, we have a partial block B (size b < k).
    # We want to pick x not in B that maximizes the number of uncovered t-subsets
    # that are subsets of B ∪ {x}.
    
    # A t-subset T is covered by B ∪ {x} iff T ⊆ B ∪ {x}, i.e., T ∩ (V \ (B ∪ {x})) = ∅.
    # Equivalently, T ⊆ B or (x ∈ T and T \ {x} ⊆ B).
    
    # So the new t-subsets covered by adding x are those T such that x ∈ T and T \ {x} ⊆ B.
    # That is, for each (t-1)-subset S of B, and for each x not in B, the t-subset S ∪ {x}
    # (if valid, i.e., x not in S) would be covered.
    
    # Let's precompute for each (t-1)-subset S (as a tuple), the list of points x such that
    # S ∪ {x} is a valid t-subset (i.e., x not in S, and S ∪ {x} sorted is in combo_to_idx).
    
    # Actually, simpler: for each (t-1)-subset S, the points x that can extend it to a t-subset
    # are all x not in S. The t-subset is sorted(S + (x,)).
    
    # Let's precompute a mapping: for each (t-1)-tuple S, a dict from x to index of S∪{x}
    # This might be large. v=24, t=3: (t-1)=2, C(24,2)=276, each has 22 possible x's. Total ~6000. Fine.
    # For t=7: (t-1)=6, C(24,6)=134596, each has 18 possible x's. Total ~2.4M. Might be ok.
    
    # Let's do it lazily or use a different approach.
    
    # Alternative approach: for each candidate point x, compute the gain by checking
    # all (t-1)-subsets of the current block B.
    
    # For a block B of size b, there are C(b, t-1) (t-1)-subsets. For each such subset S,
    # the number of uncovered t-subsets that are S ∪ {x} for some x... wait, for a fixed x,
    # the t-subsets covered by adding x are exactly {S ∪ {x} : S ⊆ B, |S| = t-1, x ∉ S}.
    
    # So gain(x) = sum over S ⊆ B, |S|=t-1, x∉S of [uncovered[index(S ∪ {x})]]
    
    # To compute this fast, we can precompute for each x a bitset or count of uncovered
    # t-subsets that contain x and have the other t-1 elements in B.
    
    # Let's use a different strategy: maintain for each point x, a count of how many
    # uncovered t-subsets contain x. But that's not enough because we need the other
    # t-1 elements to be in B.
    
    # For simplicity and correctness, let's use the direct computation but optimize with
    # early termination and caching.
    
    # Actually, let's use a simpler but effective approach:
    # 1. Use the standard greedy: pick the block (k-subset) that covers the most uncovered t-subsets.
    # 2. But enumerating all C(v,k) blocks is expensive.
    
    # Compromise: build blocks one point at a time (as in the original), but use NumPy
    # for the gain computation and add randomization.
    
    while remaining > 0 and _remaining_time() > _DEADLINE_MARGIN:
        # Start a new block
        # Pick an uncovered t-subset to start with
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Randomly pick one uncovered t-subset
        start_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        start_combo = combos[start_idx]
        block = list(start_combo)
        
        # Extend the block to size k
        while len(block) < k and _remaining_time() > _DEADLINE_MARGIN:
            block_set = set(block)
            candidates = [x for x in range(v) if x not in block_set]
            
            if not candidates:
                break
            
            # Compute gain for each candidate
            # gain(x) = number of uncovered t-subsets T such that T ⊆ B ∪ {x} and x ∈ T
            # = number of (t-1)-subsets S of B such that S ∪ {x} is uncovered
            
            # Get all (t-1)-subsets of current block
            block_tuples = [tuple(b) for b in itertools.combinations(block, t - 1)]
            
            gains = {}
            for x in candidates:
                gain = 0
                for S in block_tuples:
                    # S ∪ {x}
                    new_t = tuple(sorted(S + (x,)))
                    if new_t in combo_to_idx:
                        idx = combo_to_idx[new_t]
                        if uncovered[idx]:
                            gain += 1
                gains[x] = gain
            
            # Pick the candidate with max gain; break ties randomly
            max_gain = max(gains.values())
            best_candidates = [x for x, g in gains.items() if g == max_gain]
            x = rng.choice(best_candidates)
            block.append(x)
        
        block.sort()
        
        # Add the block
        if block not in all_blocks_set:
            blocks.append(block)
            all_blocks_set.add(tuple(block))
            
            # Update uncovered
            covered_idx = _block_covers(block, v, t, combo_to_idx)
            for idx in covered_idx:
                if uncovered[idx]:
                    uncovered[idx] = False
                    remaining -= 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't contribute to covering any t-subset uniquely."""
    if not blocks:
        return blocks
    
    # For each block, check if removing it still leaves all t-subsets covered
    # This is expensive, so do it greedily: for each block, if there's another block
    # that covers all the same t-subsets, remove it.
    
    # Simpler: for each block, check if there exists a t-subset that is ONLY covered
    # by this block. If not, the block is redundant.
    
    # Build a map: t-subset -> list of block indices that cover it
    combo_to_idx = {}
    all_combos = list(itertools.combinations(range(v), t))
    for i, c in enumerate(all_combos):
        combo_to_idx[c] = i
    
    n_combos = len(all_combos)
    # cover_count[c] = number of blocks covering t-subset c
    # Actually, we need to know if any t-subset is covered by exactly one block.
    
    # For each block, find the set of t-subsets it covers.
    block_cover_sets = []
    for block in blocks:
        covered = set()
        for sub in itertools.combinations(block, t):
            covered.add(combo_to_idx[sub])
        block_cover_sets.append(covered)
    
    # For each t-subset, count how many blocks cover it
    from collections import defaultdict
    cover_count = defaultdict(int)
    for covered in block_cover_sets:
        for idx in covered:
            cover_count[idx] += 1
    
    # A block is redundant if all t-subsets it covers are also covered by at least one other block
    # i.e., for all idx in block_cover_sets[i], cover_count[idx] >= 2
    keep = [True] * len(blocks)
    for i, covered in enumerate(block_cover_sets):
        redundant = True
        for idx in covered:
            if cover_count[idx] == 1:
                redundant = False
                break
        if redundant:
            keep[i] = False
            # Update cover_count
            for idx in covered:
                cover_count[idx] -= 1
    
    return [blocks[i] for i in range(len(blocks)) if keep[i]]


def construct(v, k, t):
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 140.0  # 10s safety margin
    
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple random seeds
    for seed in range(20):
        if _remaining_time() < _DEADLINE_MARGIN + 5:
            break
        
        try:
            blocks = _greedy_cover_fast(v, k, t, seed=seed, deadline=_DEADLINE)
        except Exception:
            continue
        
        if not blocks:
            continue
        
        # Verify coverage
        all_combos = set(itertools.combinations(range(v), t))
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)
        
        if covered != all_combos:
            # Incomplete coverage, skip
            continue
        
        # Try redundancy removal
        try:
            reduced = _remove_redundant_blocks(blocks, v, t)
            if len(reduced) < len(blocks):
                # Verify reduced still covers
                covered2 = set()
                for block in reduced:
                    for sub in itertools.combinations(block, t):
                        covered2.add(sub)
                if covered2 == all_combos:
                    blocks = reduced
        except Exception:
            pass
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        if _remaining_time() < _DEADLINE_MARGIN + 10:
            break
    
    if best_blocks is None:
        # Fallback: simple greedy
        _DEADLINE = time.time() + 10.0
        best_blocks = _greedy_cover_fast(v, k, t, seed=0, deadline=_DEADLINE)
    
    return best_blocks
