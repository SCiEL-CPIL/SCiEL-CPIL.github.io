"""Covering design C(19,7,4) via fast greedy with vectorized NumPy ops and local search."""
import time
import itertools
import numpy as np
from scipy.special import comb

# Precompute all t-subsets (4-subsets) of 19 points
# We'll represent them as indices into a precomputed list.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin before 120s limit

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)  # C(19,4) = 3876
    # Map each t-subset to an index
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # Precompute for each point x, the list of t-subset indices that contain x
    # point_tsubs[x] = array of t-subset indices containing x
    point_tsubs = []
    for x in range(v):
        idxs = []
        for i, sub in enumerate(all_tsubs):
            if x in sub:
                idxs.append(i)
        point_tsubs.append(np.array(idxs, dtype=np.int32))

    # Greedy covering with vectorized gain calculation
    uncovered = np.ones(n_tsubs, dtype=np.int8)  # 1 = uncovered
    blocks = []

    while True:
        if time.time() > deadline:
            break
        if uncovered.sum() == 0:
            break

        # Find the first uncovered t-subset to anchor the block
        unc_idx = np.where(uncovered == 1)[0]
        if len(unc_idx) == 0:
            break
        first_unc = all_tsubs[unc_idx[0]]
        block = list(first_unc)
        block_set = set(block)

        # Add points one by one
        while len(block) < k:
            if time.time() > deadline:
                break

            # For each candidate point x not in block, compute gain:
            # number of currently-uncovered t-subsets that would be covered by adding x
            # A t-subset is covered by block+x if it contains x and (t-1) elements from block
            # More precisely: for each (t-1)-subset S of block, the t-subset S ∪ {x} is covered
            # But we need to count only those that are currently uncovered.

            # Vectorized approach:
            # For candidate x, the t-subsets covered by adding x to block are those
            # that contain x and at least t-1 elements from block.
            # Actually, a t-subset is covered by the final block if all its elements are in the block.
            # So when we add x to block, the new t-subsets covered are those t-subsets
            # that contain x and whose other t-1 elements are all in block.

            # Precompute: for the current block, which t-subsets have all their elements in block?
            # That's what we want to count for each x.

            # Efficient approach:
            # For each (t-1)-subset S of the current block, and for each x not in block,
            # the t-subset S ∪ {x} is covered.
            # So gain(x) = number of (t-1)-subsets S of block such that S ∪ {x} is uncovered.

            # Let's compute this vectorized.
            # For each x not in block, gain(x) = sum over (t-1)-subsets S of block:
            #   uncovered[index of S ∪ {x}] if S ∪ {x} exists as a t-subset

            # Alternative: precompute for each point x, the t-subsets containing x.
            # Then gain(x) = number of t-subsets in point_tsubs[x] that are uncovered
            #   AND whose other t-1 elements are all in block.

            # Let's use the following: for each x, check all t-subsets containing x.
            # A t-subset sub containing x is covered by block ∪ {x} if all elements of sub \ {x} are in block.
            # So gain(x) = count of uncovered t-subsets sub containing x where sub \ {x} ⊆ block.

            candidates = [x for x in range(v) if x not in block_set]
            if not candidates:
                break

            best_x = None
            best_gain = -1

            for x in candidates:
                # t-subsets containing x
                tsub_idxs = point_tsubs[x]
                # For each such t-subset, check if all other t-1 elements are in block
                # We can check: for t-subset sub, is sub \ {x} ⊆ block_set?
                # Vectorized: for each t-subset index in tsub_idxs, check membership.

                # Get the t-subsets
                # We need to check if for t-subset sub, all elements except x are in block_set.
                # Let's precompute a bitmask or use set operations.

                # Since t=4, block size is small, let's just iterate over tsub_idxs
                # But that's up to C(18,3) = 816 per x, and ~19 candidates, so ~15500 checks per step.
                # That's fine.

                gain = 0
                for ti in tsub_idxs:
                    if uncovered[ti]:
                        sub = all_tsubs[ti]
                        # Check if all elements of sub except x are in block_set
                        # sub has t=4 elements, one is x
                        rest = [s for s in sub if s != x]
                        if all(r in block_set for r in rest):
                            gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain == 0:
                # No point adds coverage; pick any remaining
                best_x = candidates[0]

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered: mark all t-subsets contained in this block as covered
        for sub in itertools.combinations(block, t):
            uncovered[tsub_to_idx[sub]] = 0

    # If we didn't finish within deadline, we need to ensure validity.
    # Check if all t-subsets are covered.
    if uncovered.sum() > 0:
        # Fallback: add more blocks greedily to cover remaining
        while uncovered.sum() > 0:
            unc_idx = np.where(uncovered == 1)[0]
            if len(unc_idx) == 0:
                break
            first_unc = all_tsubs[unc_idx[0]]
            block = list(first_unc)
            block_set = set(block)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_x = None
                best_gain = -1
                for x in candidates:
                    gain = 0
                    for ti in point_tsubs[x]:
                        if uncovered[ti]:
                            sub = all_tsubs[ti]
                            rest = [s for s in sub if s != x]
                            if all(r in block_set for r in rest):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    best_x = candidates[0]
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered[tsub_to_idx[sub]] = 0

    # Local search: try to remove redundant blocks
    if time.time() < deadline - 5:
        # Try removing each block and see if coverage is still complete
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Recompute coverage from scratch for simplicity
            covered = np.zeros(n_tsubs, dtype=np.int8)
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    covered[tsub_to_idx[sub]] = 1

            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                # Temporarily remove block i
                # Recompute coverage without block i
                covered2 = np.zeros(n_tsubs, dtype=np.int8)
                for j, block in enumerate(blocks):
                    if j == i:
                        continue
                    for sub in itertools.combinations(block, t):
                        covered2[tsub_to_idx[sub]] = 1
                if covered2.sum() == n_tsubs:
                    # Block i is redundant
                    blocks.pop(i)
                    changed = True
                    break

    return blocks
