"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations


def _combo_index(combo, v, t):
    """Map a t-subset (sorted tuple of ints in 0..v-1) to a unique index in [0, C(v,t))."""
    # Combinadic / rank of combination
    # We use the standard combinatorial number system rank.
    # For a combination c0 < c1 < ... < c_{t-1}, the rank is:
    # sum_{i=0}^{t-1} C(c_i, i+1)  ... wait, that's for 0-indexed where c_i >= i.
    # Actually the standard formula: rank = sum_{i=0}^{t-1} C(c_i, i+1) is not quite right for 0-indexed.
    # Let's use: for combination (c0, c1, ..., c_{t-1}) with 0 <= c0 < c1 < ... < c_{t-1} < v,
    # the rank among all C(v,t) combinations (lexicographic order) is:
    # sum_{i=0}^{t-1} C(c_i, i+1)  -- this gives the "colex" rank.
    # Let me verify: for v=5, t=2, combinations in colex order:
    # (0,1)=C(1,1)=1? No.
    # Let me just use a lookup table approach or a simpler mapping.
    # Actually, for v=24, t=3, C(24,3)=2024, which is small. We can precompute.
    # But let's use the combinatorial number system properly.
    # The rank of (c0, c1, ..., c_{t-1}) in colexicographic order is:
    # sum_{i=0}^{t-1} C(c_i, i+1)
    # Let me verify with small case: v=4, t=2
    # Co-lex order: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3)
    # (0,1): C(0,1)+C(1,2) = 0+0 = 0. Good.
    # (0,2): C(0,1)+C(2,2) = 0+1 = 1. Good.
    # (1,2): C(1,1)+C(2,2) = 1+1 = 2. Good.
    # (0,3): C(0,1)+C(3,2) = 0+3 = 3. Good.
    # (1,3): C(1,1)+C(3,2) = 1+3 = 4. Good.
    # (2,3): C(2,1)+C(3,2) = 2+3 = 5. Good.
    # So the formula works.
    rank = 0
    for i in range(t):
        c = combo[i]
        # C(c, i+1)
        if c >= i + 1:
            rank += _binom(c, i + 1)
        else:
            rank += 0
    return rank


def _binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _precompute_binom_table(v, t):
    """Precompute binomial coefficients up to v choose t."""
    # We need C(n, k) for n up to v-1 and k up to t.
    # Build a 2D table.
    max_n = v
    max_k = t
    table = [[0] * (max_k + 1) for _ in range(max_n + 1)]
    for n in range(max_n + 1):
        table[n][0] = 1
        for k in range(1, min(n, max_k) + 1):
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k]
    return table


def _combo_index_fast(combo, t, binom_table):
    rank = 0
    for i in range(t):
        c = combo[i]
        if c >= i + 1:
            rank += binom_table[c][i + 1]
    return rank


def _greedy_cover_with_pool(v, k, t, pool, deadline):
    """Greedy set cover using a precomputed pool of blocks.
    pool: list of blocks (each a frozenset or tuple of k points).
    Returns the covering (list of blocks) if found before deadline, else None.
    """
    total_triples = _binom(v, t)
    binom_table = _precompute_binom_table(v, t)
    
    # Map each triple to its index
    # For each block in pool, compute the set of triple indices it covers
    # We'll represent coverage as a bitset (numpy bool array of length total_triples)
    
    # Precompute for each block in the pool, which triples it covers
    # And also, for greedy, we need to know for each candidate block, how many NEW triples it covers.
    
    # Let's precompute the triple indices for each block in the pool.
    block_triples = []
    for block in pool:
        triples = []
        for tri in combinations(block, t):
            idx = _combo_index_fast(tri, t, binom_table)
            triples.append(idx)
        block_triples.append(set(triples))
    
    # Now greedy: start with all triples uncovered.
    # covered = np.zeros(total_triples, dtype=bool)
    # We need to repeatedly pick the block that covers the most uncovered triples.
    
    # For efficiency, maintain for each block in pool, the count of uncovered triples it covers.
    # covered_counts[i] = number of uncovered triples covered by pool[i]
    
    covered = np.zeros(total_triples, dtype=np.uint8)  # 0 = uncovered, 1 = covered
    covered_counts = np.array([len(bt) for bt in block_triples], dtype=np.int32)
    
    selected = []
    remaining = len(pool)
    pool_indices = list(range(len(pool)))
    
    # We'll use a priority approach: repeatedly pick the block with max covered_counts.
    # To speed up, we can use a max-heap, but updating is expensive.
    # Alternative: since pool size is manageable (a few thousand), we can do a linear scan each time.
    # But that's O(pool_size * num_selected * avg_triples_per_block) which might be slow.
    
    # Let's use numpy to speed up the gain computation.
    # For each step, for each remaining block, compute how many of its triples are uncovered.
    # We can do this by: for block i, gain = sum(1 - covered[j] for j in block_triples[i])
    
    # To vectorize: create a 2D array where row i is the triple indices of block i, padded with -1.
    # Then gain[i] = sum(covered[padded[i]] == 0)
    
    # Let's build this matrix.
    max_triples_per_block = _binom(k, t)
    triple_matrix = np.full((len(pool), max_triples_per_block), -1, dtype=np.int32)
    for i, bt in enumerate(block_triples):
        for j, idx in enumerate(bt):
            triple_matrix[i, j] = idx
    
    # valid_mask[i, j] = (triple_matrix[i, j] >= 0)
    valid_mask = triple_matrix >= 0
    
    while remaining > 0:
        # Compute gains for all remaining blocks
        # For each block i, gain = sum over j where valid_mask[i,j] of (covered[triple_matrix[i,j]] == 0)
        # We can do: gains = np.sum((covered[triple_matrix] == 0) & valid_mask, axis=1)
        # But triple_matrix has -1 entries, so we need to be careful.
        # Let's create a masked version.
        
        # Actually, let's just compute:
        # temp = covered[triple_matrix[valid_mask]]  -- this is a 1D array
        # But we need per-block sums.
        
        # Simpler: use np.bincount or a loop.
        # Let's try: for each block, the gain is the number of its triples that are uncovered.
        # We can precompute a lookup: for each triple index, is it covered?
        # Then for block i, gain = sum(1 - covered[idx] for idx in block_triples[i])
        
        # Let's just do a vectorized approach with the matrix.
        # Replace -1 with 0 in a copy, and use a mask.
        tm = triple_matrix.copy()
        tm[tm < 0] = 0
        # Now gains[i] = sum(valid_mask[i] & (covered[tm[i]] == 0))
        covered_lookup = covered[tm]  # shape (len(pool), max_triples_per_block)
        gains = np.sum(valid_mask & (covered_lookup == 0), axis=1)
        
        # Find the block with max gain
        # Only consider blocks that are still in the pool (not already selected)
        # We need to track which blocks are still available.
        # Let's maintain an available array.
        
        # For simplicity, let's just pick the max gain block, and mark it as used.
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            # No block covers any uncovered triple -- this shouldn't happen if pool is valid
            return None
        
        selected.append(pool[best_idx])
        
        # Update covered
        for idx in block_triples[best_idx]:
            covered[idx] = 1
        
        # Mark this block as used
        gains[best_idx] = -1  # So it won't be picked again
        
        remaining -= 1
        
        # Check if all covered
        if covered.all():
            break
        
        # Deadline check
        if time.time() > deadline:
            return None
    
    if covered.all():
        return selected
    return None


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute binomial table
    binom_table = _precompute_binom_table(v, t)
    total_triples = _binom(v, t)
    
    # Strategy: generate a pool of random blocks, then use greedy deletion.
    # We want a pool that is large enough to allow good covering, but not too large for greedy.
    
    # Generate random blocks.
    # We'll generate multiple batches and pick the best covering.
    
    best_cover = None
    best_size = float('inf')
    
    # Number of blocks to generate in the pool
    pool_size = 500
    
    # Try several random seeds / pool generations
    num_attempts = 5
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Generate pool
        pool = []
        seen = set()
        for _ in range(pool_size):
            block = tuple(sorted(random.sample(range(v), k)))
            if block not in seen:
                seen.add(block)
                pool.append(block)
        
        # Run greedy cover on this pool
        cover = _greedy_cover_with_pool(v, k, t, pool, deadline)
        if cover is not None:
            size = len(cover)
            if size < best_size:
                best_size = size
                best_cover = cover
        
        if time.time() > deadline:
            break
    
    # If no cover found from random pool, fall back to deterministic greedy
    if best_cover is None:
        # Deterministic greedy
        uncovered = set(combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(combinations(block, t))
            if time.time() > deadline:
                break
        best_cover = blocks
    
    # Convert to list of lists
    return [list(b) for b in best_cover]
# EVOLVE-BLOCK-END
