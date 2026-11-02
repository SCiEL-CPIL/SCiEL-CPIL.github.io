"""Covering design C(19,8,5): improved greedy with local search."""
import time
import itertools
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin before 120s kill

    # Enumerate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Enumerate all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)

    # Build incidence matrix: A[i,j] = 1 if t-subset i is contained in k-block j
    # A is n_tsubs x n_kblocks
    rows = []
    cols = []
    for j, block in enumerate(all_kblocks):
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            rows.append(idx)
            cols.append(j)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_tsubs, n_kblocks))

    def time_ok():
        return time.time() < DEADLINE

    # Phase 1: Greedy construction
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    block_indices = []  # indices into all_kblocks

    while uncovered.any() and time_ok():
        # For each candidate block, compute how many currently uncovered t-subsets it covers
        # gain[j] = sum of A[i,j] for i where uncovered[i] is True
        # This is (A * uncovered)[j]
        gains = A @ uncovered
        # Pick the block with maximum gain
        best_j = int(np.argmax(gains))
        if gains[best_j] == 0:
            break
        blocks.append(list(all_kblocks[best_j]))
        block_indices.append(best_j)
        # Update uncovered
        uncovered[best_j]  # just to keep numpy happy
        col = A.getcol(best_j)
        uncovered[col.indices] = False

    # Check if we have a valid covering
    if not uncovered.any():
        # Try to reduce with local search
        return _local_search(blocks, all_kblocks, A, tsub_to_idx, n_tsubs, time_ok, DEADLINE)

    # If greedy didn't finish, try to complete with a faster method
    while uncovered.any() and time.time() < DEADLINE - 5:
        idx = int(np.argmax(uncovered))
        ts = all_tsubs[idx]
        # Find a block containing this t-subset
        # We can use the incidence matrix: find columns j where A[idx, j] = 1
        row = A.getrow(idx)
        candidates = row.indices
        if len(candidates) > 0:
            # Pick the candidate that covers the most uncovered t-subsets
            best_j = candidates[0]
            best_gain = 0
            for j in candidates[:50]:  # limit to first 50 for speed
                col = A.getcol(j)
                gain = uncovered[col.indices].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            blocks.append(list(all_kblocks[best_j]))
            block_indices.append(best_j)
            col = A.getcol(best_j)
            uncovered[col.indices] = False
        else:
            # Should not happen
            break

    # Final validity check
    if uncovered.any():
        # Emergency: add blocks to cover remaining
        while uncovered.any() and time.time() < DEADLINE - 1:
            idx = int(np.argmax(uncovered))
            ts = all_tsubs[idx]
            # Add a block that contains this t-subset
            # Pick a random completion
            ts_set = set(ts)
            remaining = [x for x in range(v) if x not in ts_set]
            random.shuffle(remaining)
            block = list(ts) + remaining[:k - t]
            block.sort()
            blocks.append(block)
            for ts2 in itertools.combinations(block, t):
                uncovered[tsub_to_idx[ts2]] = False

    return blocks


def _local_search(blocks, all_kblocks, A, tsub_to_idx, n_tsubs, time_ok, DEADLINE):
    """Try to reduce the number of blocks via local search."""
    n_blocks = len(blocks)
    if n_blocks < 100:
        return blocks

    # Current coverage count for each t-subset
    coverage = np.zeros(n_tsubs, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            coverage[tsub_to_idx[ts]] += 1

    # Try to remove blocks one by one if coverage remains >= 1
    improved = True
    while improved and time_ok():
        improved = False
        for i in range(len(blocks)):
            if not time_ok():
                break
            block = blocks[i]
            # Check if removing this block would leave any t-subset uncovered
            removable = True
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if coverage[idx] <= 1:
                    removable = False
                    break
            if removable:
                # Remove this block
                for ts in itertools.combinations(block, t):
                    coverage[tsub_to_idx[ts]] -= 1
                blocks.pop(i)
                improved = True
                break  # restart scan from beginning

    # Try to replace pairs of blocks with a single block
    if time_ok() and len(blocks) > 50:
        # For each pair, check if their union of covered t-subsets can be covered by one block
        # This is expensive, so only try a few random pairs
        random.seed(42)
        n_tries = min(1000, len(blocks) * (len(blocks) - 1) // 2)
        tried = set()
        for _ in range(n_tries):
            if not time_ok():
                break
            i, j = random.sample(range(len(blocks)), 2)
            if (min(i,j), max(i,j)) in tried:
                continue
            tried.add((min(i,j), max(i,j)))
            # Check if there's a block that covers all t-subsets covered by blocks i and j
            # This is hard to check efficiently, so skip for now

    return blocks
