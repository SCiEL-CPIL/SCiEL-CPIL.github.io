"""Improved greedy covering for C(v,k,t) using residual-demand matrix +
hardest-subset-driven block completion with frequency tie-breaking."""
import time
import numpy as np

# Global deadline guard (150s wall-clock limit; keep a safety margin).
_START = time.time()
_LIMIT = 150.0
_MARGIN = 3.0


def construct(v, k, t):
    global _START
    _START = time.time()

    # All t-subsets as a flat index space.
    if t == 0:
        return []

    # Precompute: for each point x, the list of t-subset-indices that contain x.
    # We build a compact index for every t-subset.
    from itertools import combinations
    all_ts = list(combinations(range(v), t))
    nT = len(all_ts)
    if nT == 0:
        return []

    # Map (sorted tuple) -> index
    idx_of = {ts: i for i, ts in enumerate(all_ts)}

    # For each point, list of t-subset indices containing it (as int array).
    pt_to_ts = []
    for x in range(v):
        lst = []
        # t-subsets containing x: choose t-1 others from the remaining v-1 points.
        others = [y for y in range(v) if y != x]
        for rest in combinations(others, t - 1):
            ts = tuple(sorted((x,) + rest))
            lst.append(idx_of[ts])
        pt_to_ts.append(np.array(lst, dtype=np.int64))

    # residual[i] = 1 if t-subset i is still uncovered, else 0.
    residual = np.ones(nT, dtype=np.int64)
    # freq[i] = how many blocks already contain t-subset i (for tie-break/diversity).
    freq = np.zeros(nT, dtype=np.int64)

    blocks = []

    def elapsed():
        return time.time() - _START

    def time_ok():
        return elapsed() < (_LIMIT - _MARGIN)

    # Greedy loop: while there are uncovered t-subsets.
    while True:
        if not time_ok():
            break
        # Find any uncovered subset to seed the next block.
        unc_idx = np.flatnonzero(residual)
        if unc_idx.size == 0:
            break

        # Seed: the hardest uncovered subset = highest frequency (most "wanted")
        # to reduce redundancy; tie-break by smallest index for determinism.
        # We want to cover a subset that is currently uncovered; among uncovered,
        # pick the one with the highest freq (closest to being a hotspot) so we
        # diversify. If all freq equal, pick min index.
        sub_freq = freq[unc_idx]
        # Choose max freq; on tie, min original index.
        order = np.lexsort((unc_idx, -sub_freq))
        seed_i = int(unc_idx[order[0]])
        seed = list(all_ts[seed_i])
        block = list(seed)
        block_set = set(block)

        # Build the current block's coverage mask (t-subsets fully inside block).
        # We'll maintain an array of covered t-subset indices for the current block.
        block_cov = []  # list of t-subset indices contained in block

        def recompute_cov():
            # recompute all t-subsets contained in the current block
            cov = []
            for ts in combinations(block, t):
                cov.append(idx_of[tuple(sorted(ts))])
            return cov

        block_cov = recompute_cov()

        # Extend block until it has k points.
        while len(block) < k:
            if not time_ok():
                break
            best_x = None
            best_key = None
            for x in range(v):
                if x in block_set:
                    continue
                # New t-subsets that become covered by adding x:
                # those that contain x and all other t-1 points from block.
                # We can compute via the residual matrix restricted to pt_to_ts[x],
                # but only those t-subsets whose other t-1 points are in block.
                # Efficient approach: for each (t-1)-subset of block, the t-subset
                # is that (t-1)-subset + x. Check if it's in uncovered.
                # Number of (t-1)-subsets of block is C(len(block), t-1).
                gain = 0
                # candidate t-subsets: each (t-1)-subset of block combined with x
                for rest in combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    j = idx_of[ts]
                    if residual[j]:
                        # weight: prefer covering subsets that are "hardest"
                        # (higher freq) to reduce future redundancy.
                        gain += 1 + 0.01 * freq[j]
                # diversity tie-break: prefer x that is less over-used overall.
                # use a small penalty based on how many times x appears in blocks.
                # (cheap proxy: not tracked per-point; skip)
                key = gain
                if best_key is None or key > best_key:
                    best_key = key
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set.add(best_x)
            block_cov = recompute_cov()

        if not block_cov:
            # could not form a useful block; avoid infinite loop
            break

        block.sort()
        blocks.append(block)

        # Update residual and freq for the t-subsets this block covers.
        cov_arr = np.array(block_cov, dtype=np.int64)
        # unique to avoid double counting within the block (should be unique already)
        cov_arr = np.unique(cov_arr)
        residual[cov_arr] = 0
        freq[cov_arr] += 1

    # If we ran out of time, we may have uncovered subsets. We must still return a
    # VALID covering (every t-subset covered). So finish with a fast fallback that
    # guarantees validity, reusing the blocks we have.
    if not time_ok():
        # Fast deterministic completion: cover remaining t-subsets one block each,
        # but that could be huge. Instead, do a quick greedy but cheaper: for each
        # remaining uncovered t-subset, if not covered, add a block containing it
        # padded with arbitrary points. This is guaranteed valid.
        # Recompute uncovered set.
        # Build a set of covered t-subsets from current blocks for fast check.
        covered = set()
        for b in blocks:
            for ts in combinations(b, t):
                covered.add(tuple(sorted(ts)))
        # For each t-subset not covered, add a block.
        for ts in all_ts:
            if ts not in covered:
                b = list(ts)
                # pad with small unused points to reach k
                used = set(b)
                for x in range(v):
                    if len(b) >= k:
                        break
                    if x not in used:
                        b.append(x)
                        used.add(x)
                b.sort()
                if b not in blocks:
                    blocks.append(b)
    return blocks
