"""Covering design C(19,7,4) via randomized greedy with deletion phase."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

# Precompute all 4-subsets as indices for vectorized operations
# C(19,4) = 3876 subsets
# C(19,7) = 50388 possible blocks

def _precompute(v, k, t):
    """Precompute block-to-subset incidence structure."""
    all_subsets = list(combinations(range(v), t))
    n_subsets = len(all_subsets)  # 3876
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(combinations(range(v), k))
    n_blocks = len(all_blocks)  # 50388
    
    # For each block, precompute which t-subsets it covers
    # block_subsets[i] = list of subset indices covered by block i
    block_subsets = []
    for block in all_blocks:
        covered = []
        for sub in combinations(block, t):
            covered.append(subset_to_idx[sub])
        block_subsets.append(covered)
    
    return all_blocks, block_subsets, n_subsets, subset_to_idx

def _greedy_cover(v, k, t, all_blocks, block_subsets, n_subsets, rng, deadline):
    """Single greedy pass with randomized tie-breaking."""
    uncovered = np.ones(n_subsets, dtype=np.int32)  # count of uncovered subsets
    blocks = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        
        # Find block that covers most uncovered subsets
        # Vectorized: sum of uncovered over subsets in each block
        # This is expensive for 50388 blocks, so use a smarter approach
        
        # For each candidate block, count how many of its t-subsets are uncovered
        # We need an efficient way to do this
        
        # Alternative: for the greedy step, pick the uncovered subset with highest "pressure"
        # then find the best block covering it
        
        # Pick first uncovered subset (or random one)
        unc_idx = np.where(uncovered > 0)[0]
        if len(unc_idx) == 0:
            break
        # Random selection among uncovered for diversity
        target = unc_idx[rng.integers(len(unc_idx))]
        
        # Find all blocks that cover this subset
        # block_subsets is a list of lists; we need blocks containing subset `target`
        # Precompute: for each subset, which blocks cover it
        # This is expensive to do on the fly, so let's precompute reverse mapping
        
        # Actually, let's use a different approach:
        # For each block, compute coverage of currently uncovered subsets
        # Use numpy for speed
        
        # Build a coverage matrix approach won't work (50388 x 3876 is ~195M entries)
        # Instead, iterate over blocks and compute gain
        
        best_block = None
        best_gain = -1
        
        # For efficiency, precompute which blocks cover each subset
        # We'll do this once outside the loop - but it's large
        # 3876 subsets, each covered by C(19-4, 7-4) = C(15,3) = 455 blocks
        # Total: 3876 * 455 = 1.76M entries - manageable
        
        # Let's use a different greedy: pick block maximizing coverage
        # But computing this for all 50388 blocks each iteration is too slow
        
        # Better approach: use the precomputed reverse mapping
        # For subset `target`, get all blocks covering it, then score them
        
        pass  # This approach needs the reverse mapping
    
    return blocks


def _greedy_cover_fast(v, k, t, all_blocks, block_subsets, n_subsets, 
                       subset_to_blocks, rng, deadline):
    """Greedy cover using reverse mapping for efficiency."""
    uncovered = np.ones(n_subsets, dtype=np.int32)
    total_uncovered = n_subsets
    blocks = []
    
    while total_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Pick a random uncovered subset as target
        unc_mask = uncovered > 0
        unc_idx = np.where(unc_mask)[0]
        if len(unc_idx) == 0:
            break
        target = unc_idx[rng.integers(len(unc_idx))]
        
        # Get all blocks covering this subset
        candidate_blocks = subset_to_blocks[target]
        
        # Score each candidate: how many uncovered subsets it covers
        best_block = None
        best_gain = -1
        best_gain_val = -1
        
        for bi in candidate_blocks:
            # Count uncovered subsets in this block
            subsets_in_block = block_subsets[bi]
            gain = uncovered[subsets_in_block].sum()
            if gain > best_gain:
                best_gain = gain
                best_block = bi
        
        if best_block is None or best_gain <= 0:
            break
        
        # Add this block
        blocks.append(all_blocks[best_block])
        # Update uncovered
        uncovered[block_subsets[best_block]] -= 1
        uncovered = np.maximum(uncovered, 0)
        total_uncovered = uncovered.sum()
    
    return blocks


def _delete_redundant(blocks, v, k, t, deadline):
    """Remove redundant blocks (those whose coverage is subset of others)."""
    if not blocks:
        return blocks
    
    n = len(blocks)
    # For each block, compute the set of t-subsets it uniquely covers
    # A block is redundant if removing it still leaves all subsets covered
    
    # Compute coverage sets
    block_sets = []
    for b in blocks:
        s = frozenset(combinations(b, t))
        block_sets.append(s)
    
    # For each subset, count how many blocks cover it
    from collections import defaultdict
    subset_count = defaultdict(int)
    for s in block_sets:
        for sub in s:
            subset_count[sub] += 1
    
    # A block is redundant if all its subsets are covered by other blocks
    # i.e., for every subset in the block, count > 1
    keep = []
    for i, b in enumerate(blocks):
        if time.time() > deadline:
            keep.append(b)
            continue
        redundant = True
        for sub in block_sets[i]:
            if subset_count[sub] == 1:
                redundant = False
                break
        if not redundant:
            keep.append(b)
    
    return keep


def construct(v, k, t):
    """Construct covering design C(v,k,t) using randomized greedy with deletion."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute structure
    all_blocks, block_subsets, n_subsets, subset_to_idx = _precompute(v, k, t)
    
    # Build reverse mapping: for each subset, which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for bi, subs in enumerate(block_subsets):
        for si in subs:
            subset_to_blocks[si].append(bi)
    
    # Run multiple randomized greedy passes and keep the best
    best_blocks = None
    best_count = float('inf')
    
    n_restarts = 30  # Will be limited by deadline
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:  # Leave time for deletion
            break
        
        rng = np.random.default_rng(seed=42 + restart)
        
        # Greedy pass
        blocks = _greedy_cover_fast(v, k, t, all_blocks, block_subsets, n_subsets,
                                     subset_to_blocks, rng, deadline - 20.0)
        
        if not blocks:
            continue
        
        # Deletion phase
        if time.time() < deadline - 5.0:
            blocks = _delete_redundant(blocks, v, k, t, deadline - 5.0)
        
        # Verify coverage
        if _verify_coverage(blocks, v, k, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
        
        # Early stop if we hit the known best
        if best_count <= 151:
            break
    
    if best_blocks is None:
        # Fallback: single deterministic greedy
        rng = np.random.default_rng(seed=0)
        best_blocks = _greedy_cover_fast(v, k, t, all_blocks, block_subsets, n_subsets,
                                          subset_to_blocks, rng, deadline)
    
    return best_blocks


def _verify_coverage(blocks, v, k, t):
    """Check that every t-subset is covered by at least one block."""
    covered = set()
    for b in blocks:
        for sub in combinations(b, t):
            covered.add(sub)
    total = len(list(combinations(range(v), t)))
    return len(covered) == total

# EVOLVE-BLOCK-END
