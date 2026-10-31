"""Model-and-optimize approach for covering design C(v,k,t).
NumPy-vectorized greedy set-cover with local optimization (redundancy
removal + block replacement) to reduce block count.
"""
import time
import random
import sys
from itertools import combinations
import numpy as np

# Hard wall-clock limit; evaluator kills at 150s, leave margin.
_DEADLINE_SEC = 110.0


def _all_tsubsets(v, t):
    return set(combinations(range(v), t))


def _block_tsubsets(block, t):
    return set(combinations(block, t))


def _tsubset_indices(v, t):
    """Map every t-subset of {0..v-1} to a dense index 0..C(v,t)-1."""
    from math import comb
    total = comb(v, t)
    idx = np.full((v,) * t, -1, dtype=np.int32)
    i = 0
    for combo in combinations(range(v), t):
        idx[combo] = i
        i += 1
    return idx, total


def _greedy_construct_np(v, k, t, deadline=None, seed=42):
    """
    NumPy-vectorized greedy set-cover.
    For each new block, start from an uncovered t-subset and greedily add
    points that maximise newly-covered t-subsets, using a vectorized gain
    computation over the (t-1)-subset index table.
    """
    rng = np.random.default_rng(seed)
    tidx, total_t = _tsubset_indices(v, t)

    # uncovered: boolean array of length total_t; True = still uncovered
    uncovered = np.ones(total_t, dtype=bool)
    n_uncovered = total_t

    blocks = []
    start = time.time()

    # Precompute: for each (t-1)-tuple (sorted), the list of t-indices it belongs to.
    # tidx_to_tsubs: for a given (t-1)-tuple key, which t-subsets contain it.
    # We'll store as a dict: key (a0,...,a_{t-2}) sorted -> np.array of t-indices
    t1_to_tsubs = {}
    for combo in combinations(range(v), t):
        for i in range(t):
            sub = combo[:i] + combo[i+1:]
            if sub not in t1_to_tsubs:
                t1_to_tsubs[sub] = []
            t1_to_tsubs[sub].append(tidx[combo])
    # Convert to arrays
    for key in t1_to_tsubs:
        t1_to_tsubs[key] = np.array(t1_to_tsubs[key], dtype=np.int32)

    while n_uncovered > 0:
        if deadline is not None and time.time() - start > deadline:
            break

        # Pick an uncovered t-subset as seed
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        seed_idx = int(rng.choice(unc_idx))
        # Decode seed_idx back to the t-subset
        # We need a reverse map
        # Precompute reverse: list of t-subsets
        all_tsubs_list = list(combinations(range(v), t))
        seed_combo = all_tsubs_list[seed_idx]

        block = list(seed_combo)
        block_set = set(block)

        while len(block) < k:
            if deadline is not None and time.time() - start > deadline:
                break

            # Compute gain for each candidate x not in block:
            # gain(x) = number of uncovered t-subsets s such that x in s and |s ∩ block| = t-1
            # = sum over (t-1)-subsets B of block: [B ∪ {x} is uncovered]
            # Vectorized: for each (t-1)-subset B of block, look up t1_to_tsubs[B],
            # then count how many are uncovered and contain x.
            # Actually simpler: for each x, gain(x) = sum over B in C(block, t-1) of
            #   sum over t-indices in t1_to_tsubs[B] of uncovered[idx] where idx corresponds to B∪{x}
            # But t1_to_tsubs[B] contains ALL t-subsets containing B, not just B∪{x}.
            # We need specifically B∪{x}. Since B has t-1 elements and x notin B,
            # B∪{x} is a unique t-subset. So gain(x) = sum over B: uncovered[tidx[B∪{x}]]

            # Efficient: precompute for current block the list of (t-1)-subsets
            t1_subs = list(combinations(block, t - 1))
            if not t1_subs:
                # block has fewer than t-1 elements; shouldn't happen since we start with t
                break

            # For each candidate x, compute gain
            candidates = [x for x in range(v) if x not in block_set]
            if not candidates:
                break

            # Build a vector of t-indices for all (B, x) pairs where B in t1_subs, x in candidates
            # t_idx = tidx[B + (x,)] sorted
            # gain(x) = sum over B of uncovered[tidx[sorted(B + (x,))]]

            # Vectorized approach:
            # Create array of all (B, x) combinations
            # For each B in t1_subs and each x in candidates:
            #   s = sorted(B + (x,))
            #   idx = tidx[s]
            #   if uncovered[idx]: count for x
            #
            # Number of pairs: len(t1_subs) * len(candidates)
            # For |block|=6, t=4: C(6,3)=20, candidates=13, so 260 pairs. Very fast.

            # Build lookup: for each candidate x, sum of uncovered[tidx[sorted(B+(x,))]]
            gains = np.zeros(len(candidates), dtype=np.int32)
            for Bi, B in enumerate(t1_subs):
                for xi, x in enumerate(candidates):
                    s = tuple(sorted(B + (x,)))
                    idx = tidx[s]
                    if uncovered[idx]:
                        gains[xi] += 1

            best_xi = int(np.argmax(gains))
            best_gain = gains[best_xi]
            best_x = candidates[best_xi]

            if best_gain <= 0:
                # No improvement; pick arbitrary
                best_x = candidates[0]

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered: remove all t-subsets of this block
        for s in combinations(block, t):
            idx = tidx[s]
            if uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1

    return blocks


def _greedy_construct(v, k, t, deadline=None):
    """
    Greedy set-cover: repeatedly pick the block (k-subset) that covers the
    most currently-uncovered t-subsets.  We approximate by building each new
    block incrementally, choosing the point that maximises newly-covered
    t-subsets.  This is O(|uncovered| * v) per block in the naive version;
    we use a smarter incremental gain.
    """
    uncovered = set(combinations(range(v), t))
    if not uncovered:
        return []

    # Index: for each (t-1)-subset, track which t-subsets contain it.
    # Actually, let's use a different representation for speed.
    # Represent uncovered as a set of frozensets for O(1) membership.
    uncovered_set = set(uncovered)
    uncovered_count = len(uncovered_set)
    if uncovered_count == 0:
        return []

    blocks = []
    start = time.time()

    while uncovered_set:
        if deadline is not None and time.time() - start > deadline:
            break

        # Pick the first uncovered t-subset as a seed.
        seed = min(uncovered_set)

        # Build a block by adding points that cover the most uncovered t-subsets.
        block = list(seed)
        block_set = set(block)

        # Precompute: for each candidate point x not in block,
        # how many uncovered t-subsets would be covered if x were added?
        # A t-subset is covered by block∪{x} if it contains x and (t-1) elements from block.
        # So gain(x) = number of uncovered t-subsets that contain x and at least (t-1) elements of block.
        # More precisely: |{s in uncovered : x in s and |s ∩ block| >= t-1}|
        # Since s has size t, |s ∩ block| >= t-1 and x notin block means |s ∩ block| is exactly t-1 (if x notin block)
        # Wait: if x notin block, then s ∩ (block ∪ {x}) has size t iff s ∩ block has size t-1 and x in s.
        # So gain(x) = number of uncovered t-subsets s such that x in s and |s ∩ block| = t-1.

        # To compute this efficiently, for each (t-1)-subset of block, count uncovered t-subsets
        # that contain that (t-1)-subset and the candidate x.
        # Let's precompute for the current block: for each (t-1)-subset A of block,
        # and for each point x not in block, count uncovered t-subsets containing A∪{x}.

        # For small t and block sizes, enumerate (t-1)-subsets of block.
        # block grows from t to k, so (t-1)-subsets of block: C(|block|, t-1).

        while len(block) < k:
            if deadline is not None and time.time() - start > deadline:
                break

            # Compute gain for each candidate point
            # For efficiency, iterate over uncovered t-subsets is too slow.
            # Instead: for each (t-1)-subset A of block, for each uncovered t-subset s,
            # if s contains A, then the missing point is the one in s not in A.
            # This is still expensive.

            # Alternative: for each candidate x, gain(x) = sum over A in C(block, t-1) of
            #   count of uncovered t-subsets containing A ∪ {x}
            # Precompute: for each (t-1)-subset A of the full space, a dict mapping
            #   x -> count of uncovered t-subsets containing A ∪ {x}
            # But that's C(v, t-1) entries, each with up to v-t+1 mappings. For v=19, t=4:
            # C(19,3)=969, each with up to 17 mappings. Total ~16k entries. Feasible.

            # Let's maintain a global structure:
            # cov_count[A][x] = number of uncovered t-subsets that contain A and x
            # where A is a (t-1)-tuple.

            # We'll maintain this incrementally. Actually, let's just compute gain directly
            # by iterating over uncovered t-subsets for each candidate. For v=19, |uncovered| can
            # be up to C(19,4)=3876. Candidates: up to 19. So 3876*19 = ~74k ops per point added.
            # Total points added per block: k-t. Blocks: ~150. Total: 150 * (7-4) * 74k ≈ 33M.
            # That's tight but feasible in Python if optimized.

            # Let's use a smarter approach: maintain cov_count.

            # Actually, let me restructure: maintain a dict
            #   tri_count: key = (a,b,c) sorted triple, value = set or count of uncovered 4-sets containing it
            # For general t, maintain (t-1)-subset counts.

            # For now, given the small size (v=19, t=4, k=7), let's do direct computation
            # but with numpy or just careful loops.

            best_gain = -1
            best_x = -1
            block_frozenset = frozenset(block)

            # For each candidate x not in block:
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets that contain x and (t-1) elements from block
                # i.e., t-subsets s where x in s and |s & block| >= t-1
                # Since |s|=t and x notin block, this means |s & block| = t-1
                gain = 0
                # Iterate over (t-1)-subsets of block
                for A in combinations(block, t - 1):
                    A_set = set(A)
                    # Check if there's an uncovered t-subset containing A and x
                    # That t-subset would be A ∪ {x} (since |A|=t-1, adding x gives t)
                    # But wait: A ∪ {x} has size t only if x notin A. Since x notin block and A⊂block, x notin A.
                    # So the unique t-subset containing A and x is A ∪ {x}.
                    # So gain contribution from A is 1 if A∪{x} is uncovered, else 0.
                    # But that's not right: we want ALL uncovered t-subsets that contain x and at least t-1 from block.
                    # A t-subset s containing x with |s∩block| >= t-1: since |s|=t and x∉block,
                    # s = {x} ∪ B where B ⊂ block, |B| = t-1.
                    # So each such s is uniquely determined by a (t-1)-subset B of block: s = B ∪ {x}.
                    # So gain(x) = |{B ∈ C(block, t-1) : B ∪ {x} ∈ uncovered_set}|
                    pass
                # Direct: gain(x) = number of B in C(block, t-1) such that B∪{x} in uncovered_set
                # This is at most C(|block|, t-1), which for |block|=6, t-1=3 is C(6,3)=20.
                # So for each x, we do C(|block|,t-1) set lookups. Very fast!
                gain = 0
                for B in combinations(block, t - 1):
                    if tuple(sorted(B + (x,))) in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1 or best_gain <= 0:
                # No point adds coverage; pick arbitrary to fill block
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
                if best_x == -1:
                    break

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Remove covered t-subsets
        for s in combinations(block, t):
            uncovered_set.discard(s)

    return blocks


def _greedy_construct_optimized(v, k, t, deadline=None, seed=42):
    """
    Optimized greedy: maintain (t-1)-subset -> count of uncovered t-subsets containing it.
    This allows O(C(|block|,t-1)) gain computation per candidate.
    """
    uncovered = set(combinations(range(v), t))
    if not uncovered:
        return []

    # Precompute all (t-1)-subsets and their containing t-subsets
    # tri_to_fours: dict mapping (t-1)-tuple to set of t-tuples containing it
    # For efficiency, we'll maintain counts dynamically.

    # Actually, let's maintain:
    #   tri_count: dict (t-1)-tuple -> int, number of uncovered t-subsets containing it
    # When we remove a t-subset s, for each (t-1)-subset of s, decrement tri_count.

    tri_count = {}
    for s in uncovered:
        for i in range(t):
            # (t-1)-subset: remove the i-th element
            sub = s[:i] + s[i+1:]
            tri_count[sub] = tri_count.get(sub, 0) + 1

    blocks = []
    start = time.time()

    while uncovered:
        if deadline is not None and time.time() - start > deadline:
            break

        seed = min(uncovered)
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if deadline is not None and time.time() - start > deadline:
                break

            best_gain = -1
            best_x = -1
            for x in range(v):
                if x in block_set:
                    continue
                # gain(x) = sum over (t-1)-subsets B of block of [B∪{x} in uncovered]
                # = sum over B of C(block, t-1) of (1 if B∪{x} in uncovered else 0)
                # Note: B∪{x} is in uncovered iff tri_count[B] > 0 AND B∪{x} not yet removed
                # But tri_count[B] counts uncovered t-subsets containing B, not specifically B∪{x}.
                # We need a different structure.

                # Let's maintain: for each (t-1)-subset B and each x not in B,
                #   count_B_x = number of uncovered t-subsets containing B ∪ {x}
                # This is what we want. When we remove a t-subset s, for each x in s,
                #   B = s\{x}, decrement count_B_x.

                # This structure has size C(v, t-1) * (v - (t-1)) ≈ C(19,3)*17 ≈ 969*17 ≈ 16k.
                # Let's maintain it.

                pass

            # Fallback: direct computation
            gain = 0
            best_gain = -1
            best_x = -1
            for x in range(v):
                if x in block_set:
                    continue
                g = 0
                for B in combinations(block, t - 1):
                    s = tuple(sorted(B + (x,)))
                    if s in uncovered:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_x = x

            if best_x == -1:
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
                if best_x == -1:
                    break

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        for s in combinations(block, t):
            if s in uncovered:
                uncovered.discard(s)
                # Update tri_count
                for i in range(t):
                    sub = s[:i] + s[i+1:]
                    tri_count[sub] = tri_count.get(sub, 0) - 1
                    if tri_count[sub] <= 0:
                        del tri_count[sub]

    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = _DEADLINE_SEC

    # First, get a baseline greedy solution
    blocks = _greedy_construct_optimized(v, k, t, deadline=deadline - 5.0)

    # Validate
    all_tsubs = set(combinations(range(v), t))
    covered = set()
    for b in blocks:
        covered.update(combinations(b, t))
    
    if covered == all_tsubs:
        # Try to improve via local search: try replacing blocks or finding redundant blocks
        elapsed = time.time() - start
        remaining = deadline - elapsed - 5.0
        if remaining > 5.0:
            blocks = _local_optimize(blocks, v, k, t, time_budget=remaining)

    return blocks


def _local_optimize(blocks, v, k, t, time_budget=10.0):
    """
    Local optimization: remove redundant blocks, then try block replacement
    to reduce total block count.
    """
    start = time.time()
    all_tsubs = set(combinations(range(v), t))

    # Compute coverage count for each t-subset
    from collections import Counter
    cover_count = Counter()
    for b in blocks:
        for s in combinations(b, t):
            cover_count[s] += 1

    # Pass 1: Remove fully redundant blocks (iterate until stable)
    changed = True
    while changed and time.time() - start < time_budget:
        changed = False
        new_blocks = []
        for b in blocks:
            redundant = True
            for s in combinations(b, t):
                if cover_count[s] <= 1:
                    redundant = False
                    break
            if redundant:
                for s in combinations(b, t):
                    cover_count[s] -= 1
                changed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks

    if time.time() - start >= time_budget:
        return blocks

    # Pass 2: Try to replace blocks. For each block, identify the t-subsets
    # that are ONLY covered by this block (critical t-subsets). Try to find
    # a different k-subset that covers all critical t-subsets. If found and
    # the new block also covers additional uncovered-by-others t-subsets,
    # replace.
    # Simpler heuristic: try removing each block and see if the remaining
    # blocks + one new block can cover everything. If yes, replace.
    # This is expensive, so do it selectively.

    # Alternative: for each block, try swapping one point for another to see
    # if we can cover more unique t-subsets, enabling removal of another block.

    # Let's try: for each pair of blocks, see if their union of covered t-subsets
    # can be covered by a single k-subset. If yes, replace both with one.
    # This is very expensive, so only do it for a few iterations.

    # Simpler: try to find a block that, when removed, leaves only a small
    # number of uncovered t-subsets, and then try to cover those with a
    # modified version of another block.

    # For now, do a randomized restart of the greedy with different seeds
    # and keep the best result.
    if time.time() - start < time_budget - 5.0:
        best_blocks = blocks
        best_count = len(blocks)
        remaining = time_budget - (time.time() - start) - 2.0
        if remaining > 5.0:
            for trial_seed in [123, 456, 789, 321, 654]:
                if time.time() - start > time_budget - 2.0:
                    break
                trial_deadline = time_budget - (time.time() - start) - 1.0
                if trial_deadline < 2.0:
                    break
                trial_blocks = _greedy_construct_np(v, k, t, deadline=trial_deadline, seed=trial_seed)
                # Validate
                trial_covered = set()
                for b in trial_blocks:
                    trial_covered.update(combinations(b, t))
                if trial_covered == all_tsubs and len(trial_blocks) < best_count:
                    best_blocks = trial_blocks
                    best_count = len(trial_blocks)
                    # Re-run redundancy removal on the improved set
                    cc = Counter()
                    for b in best_blocks:
                        for s in combinations(b, t):
                            cc[s] += 1
                    changed2 = True
                    while changed2 and time.time() - start < time_budget - 1.0:
                        changed2 = False
                        nb = []
                        for b in best_blocks:
                            red = True
                            for s in combinations(b, t):
                                if cc[s] <= 1:
                                    red = False
                                    break
                            if red:
                                for s in combinations(b, t):
                                    cc[s] -= 1
                                changed2 = True
                            else:
                                nb.append(b)
                        best_blocks = nb
            blocks = best_blocks

    return blocks


if __name__ == "__main__":
    # Quick self-test
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1-t0:.1f}s")
    # Verify
    all_tsubs = set(combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
