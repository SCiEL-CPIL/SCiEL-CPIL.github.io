"""Covering design C(v, k, t) – NumPy-vectorized greedy with local search refinement."""
import itertools
import time
import random
import numpy as np

_DEADLINE_S = 145.0
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _build_block_indices(v, k, t):
    """Precompute for each t-subset index the bitmask of k-blocks containing it.
    Returns (num_tsubs, block_bitmasks) where block_bitmasks[i] is a list of block indices."""
    all_blocks = list(itertools.combinations(range(v), k))
    all_tsubs = list(itertools.combinations(range(v), t))
    
    # Map t-subset to its index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each block, compute which t-subsets it covers
    # We'll build a sparse structure: for each t-subset, list of blocks covering it
    num_tsubs = len(all_tsubs)
    num_blocks = len(all_blocks)
    
    # Sparse: tsub -> list of block indices
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    
    for bi, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                tsub_to_blocks[idx].append(bi)
    
    return all_blocks, all_tsubs, tsub_to_blocks


def _greedy_construct(v, k, t, rng, deadline_check=None):
    """Vectorized greedy construction using NumPy for block scoring."""
    all_blocks = list(itertools.combinations(range(v), k))
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    num_blocks = len(all_blocks)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix: M[tsub_idx, block_idx] = 1 if block covers tsub
    # For v=19, k=7, t=4: 3876 tsubs x 50388 blocks
    # Use sparse representation for efficiency
    from scipy import sparse
    
    # Build sparse matrix
    row_indices = []
    col_indices = []
    for bi, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            row_indices.append(tsub_to_idx[ts])
            col_indices.append(bi)
    
    M = sparse.csr_matrix((np.ones(len(row_indices), dtype=np.int32), (row_indices, col_indices)),
                           shape=(num_tsubs, num_blocks))
    
    # Transpose for faster column access: M_T[block_idx, tsub_idx]
    M_T = M.T.tocsr()
    
    # Uncovered mask
    uncovered = np.ones(num_tsubs, dtype=np.int32)
    uncovered_count = num_tsubs
    
    blocks = []
    block_set = set()
    
    while uncovered_count > 0:
        if deadline_check and deadline_check() < 2.0:
            break
        
        # Score each block: number of currently uncovered t-subsets it covers
        # score = M @ uncovered (vector of length num_blocks)
        scores = M.dot(uncovered).astype(np.int32)
        
        # Pick the block with highest score
        best_block_idx = int(np.argmax(scores))
        best_score = scores[best_block_idx]
        
        if best_score <= 0:
            break
        
        block = all_blocks[best_block_idx]
        bkey = block  # already a tuple
        if bkey in block_set:
            # Mark this block as "used" to avoid re-picking (set score to 0)
            scores[best_block_idx] = 0
            continue
        
        blocks.append(list(block))
        block_set.add(bkey)
        
        # Update uncovered mask
        # Get t-subsets covered by this block
        covered_tsubs = M_T[best_block_idx].indices
        for ti in covered_tsubs:
            if uncovered[ti]:
                uncovered[ti] = 0
                uncovered_count -= 1
    
    return blocks, uncovered_count


def _greedy_construct_python(v, k, t, rng, deadline_check=None):
    """Fallback Python-based greedy if NumPy/SciPy construction is too slow."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    uncovered = set(all_tsubs)
    blocks = []
    block_set = set()
    
    while uncovered:
        if deadline_check and deadline_check() < 2.0:
            break
        
        # Find the t-subset that is "hardest" to cover (fewest blocks can cover it)
        # For efficiency, pick a random uncovered t-subset and build a block around it
        target = rng.choice(list(uncovered))
        
        # Build a k-block containing target that maximizes coverage
        block = list(target)
        block_set_local = set(block)
        
        remaining_points = [x for x in range(v) if x not in block_set_local]
        rng.shuffle(remaining_points)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k and remaining_points:
            best_pt = None
            best_gain = -1
            for x in remaining_points:
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(combo + (x,)))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_pt = x
            if best_pt is None or best_gain == 0:
                best_pt = remaining_points[0]
            block.append(best_pt)
            block_set_local.add(best_pt)
            remaining_points.remove(best_pt)
        
        block = sorted(block)
        bkey = tuple(block)
        if bkey not in block_set:
            blocks.append(block)
            block_set.add(bkey)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
    
    return blocks, len(uncovered)


def _remove_redundant_blocks(blocks, v, k, t):
    """Remove blocks whose unique coverage is redundant."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # For each t-subset, count how many blocks cover it
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    block_tsubs = []  # For each block, list of t-subset indices it covers
    
    for block in blocks:
        block_ts = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            cover_count[idx] += 1
            block_ts.append(idx)
        block_tsubs.append(block_ts)
    
    # Remove blocks that have no unique t-subset (all their t-subsets are covered by other blocks)
    new_blocks = []
    new_block_tsubs = []
    for i, block in enumerate(blocks):
        has_unique = False
        for ti in block_tsubs[i]:
            if cover_count[ti] == 1:
                has_unique = True
                break
        if has_unique:
            new_blocks.append(block)
            new_block_tsubs.append(block_tsubs[i])
    
    # Update cover counts
    if len(new_blocks) < len(blocks):
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for bts in new_block_tsubs:
            for ti in bts:
                cover_count[ti] += 1
    
    return new_blocks, cover_count


def _local_search(blocks, v, k, t, rng, max_iter=500):
    """Simulated annealing: try replacing a block with a better one."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Build cover counts
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    block_tsubs = []
    for block in blocks:
        bts = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            cover_count[idx] += 1
            bts.append(idx)
        block_tsubs.append(bts)
    
    current_len = len(blocks)
    best_len = current_len
    best_blocks = [b[:] for b in blocks]
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(max_iter, 1))
    T = T_start
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        
        if _time_left() < 0.5:
            break
        
        # Pick a random block to try replacing
        i = rng.randint(0, len(blocks) - 1)
        
        # Try removing it: check if all its unique t-subsets are still covered
        unique_count = 0
        for ti in block_tsubs[i]:
            if cover_count[ti] == 1:
                unique_count += 1
        
        if unique_count == 0:
            # Safe to remove
            blocks.pop(i)
            block_tsubs.pop(i)
            for ti in block_tsubs[i]:
                cover_count[ti] -= 1
            current_len -= 1
            if current_len < best_len:
                best_len = current_len
                best_blocks = [b[:] for b in blocks]
        else:
            # Try to find a replacement block that covers all unique t-subsets
            # and has good additional coverage
            unique_tsubs = [ti for ti in block_tsubs[i] if cover_count[ti] == 1]
            
            # Find candidate blocks that cover all unique t-subsets
            # For efficiency, pick the first unique t-subset and find blocks containing it
            if unique_tsubs:
                first_ts = all_tsubs[unique_tsubs[0]]
                # Generate random k-blocks containing first_ts
                remaining = [x for x in range(v) if x not in first_ts]
                found_replacement = False
                for _ in range(20):
                    if _time_left() < 0.5:
                        break
                    extra = rng.sample(remaining, k - t)
                    candidate = sorted(list(first_ts) + extra)
                    ckey = tuple(candidate)
                    
                    # Check if candidate covers all unique t-subsets
                    covers_all = True
                    for ti in unique_tsubs:
                        if all_tsubs[ti] not in candidate:
                            # Check if the t-subset is a subset of candidate
                            if not set(all_tsubs[ti]).issubset(set(candidate)):
                                covers_all = False
                                break
                    
                    if covers_all:
                        # Compute gain: how many currently-covered t-subsets would be lost
                        # vs how many new ones would be gained
                        # Simplified: just accept if it covers all unique ones
                        # Remove old block
                        for ti in block_tsubs[i]:
                            cover_count[ti] -= 1
                        blocks[i] = candidate
                        new_bts = []
                        for ts in itertools.combinations(candidate, t):
                            idx = tsub_to_idx[ts]
                            cover_count[idx] += 1
                            new_bts.append(idx)
                        block_tsubs[i] = new_bts
                        found_replacement = True
                        break
        
        T *= cooling
    
    # Restore best solution
    if best_len < len(blocks):
        blocks = best_blocks
    
    return blocks


def construct(v, k, t):
    random.seed(12345)
    rng = random.Random(12345)
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Vectorized greedy construction
    try:
        blocks, uncovered_count = _greedy_construct(v, k, t, rng, deadline_check=_time_left)
        if uncovered_count == 0:
            best_blocks = blocks
            best_len = len(blocks)
    except Exception:
        pass
    
    # Phase 2: If vectorized didn't work or didn't complete, use Python fallback
    if best_blocks is None or _time_left() < 10.0:
        if best_blocks is None:
            blocks, uncovered_count = _greedy_construct_python(v, k, t, rng, deadline_check=_time_left)
            if uncovered_count == 0:
                best_blocks = blocks
                best_len = len(blocks)
    
    if best_blocks is None:
        # Last resort: pure greedy without vectorization
        all_tsubs = list(itertools.combinations(range(v), t))
        uncovered = set(all_tsubs)
        blocks = []
        block_set = set()
        while uncovered and _time_left() > 0.5:
            target = min(uncovered)
            block = list(target)
            block_set_local = set(block)
            remaining = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(remaining)
            while len(block) < k and remaining:
                best_pt = None
                best_gain = -1
                for x in remaining:
                    gain = sum(1 for combo in itertools.combinations(block, t-1) 
                              if tuple(sorted(combo + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = x
                if best_pt is None:
                    best_pt = remaining[0]
                block.append(best_pt)
                block_set_local.add(best_pt)
                remaining.remove(best_pt)
            block = sorted(block)
            bkey = tuple(block)
            if bkey not in block_set:
                blocks.append(block)
                block_set.add(bkey)
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
        best_blocks = blocks
        best_len = len(blocks)
    
    # Phase 3: Remove redundant blocks
    if _time_left() > 5.0:
        best_blocks, _ = _remove_redundant_blocks(best_blocks, v, k, t)
        if len(best_blocks) < best_len:
            best_len = len(best_blocks)
    
    # Phase 4: Local search refinement
    if _time_left() > 5.0:
        best_blocks = _local_search(best_blocks, v, k, t, rng, max_iter=200)
        if len(best_blocks) < best_len:
            best_len = len(best_blocks)
    
    # Final redundancy removal
    if _time_left() > 1.0:
        best_blocks, _ = _remove_redundant_blocks(best_blocks, v, k, t)
    
    return best_blocks
