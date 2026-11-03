<<<<<<< SEARCH
def _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Greedy covering: build k-blocks one at a time, each started from a random
    uncovered t-subset, then greedily add points maximizing newly covered t-subsets.
    Uses vectorized gain computation via numpy for speed.
    """
    global t_global
    t_global = t

    uncovered = set(range(n_tsubsets))
    blocks = []

    # Precompute: for each point x, bitmask of t-subset indices containing x
    # Represent as numpy bool array of shape (v, n_tsubsets)
    point_mask = np.zeros((v, n_tsubsets), dtype=bool)
    for idx, sub in enumerate(all_tsubsets):
        for x in sub:
            point_mask[x, idx] = True

    while uncovered:
        if time.time() > deadline:
            break

        # Pick a random uncovered t-subset to anchor the block
        start_idx = rng.choice(list(uncovered))
        start_sub = all_tsubsets[start_idx]
        block = list(start_sub)
        block_set = set(block)

        # Current coverage of t-subsets by partial block:
        # A t-subset S is "potentially completable" by block if S contains all of block's
        # elements that are in S, and the remaining t-|block∩S| elements can be added.
        # For gain when adding point x: S is newly covered iff S ⊆ block∪{x} and S ⊄ block.
        # i.e., S contains x, S ⊆ block∪{x}, and S has at least one element not in block.
        # Since |S|=t and |block|>=t-1 at this stage (block starts with t elements),
        # when |block| >= t: S ⊆ block∪{x} and x∈S means S = (S\{x}) ∪ {x} with S\{x} ⊆ block.
        # So S\{x} is a (t-1)-subset of block.
        # When |block| == t-1 (not possible here since we start with t): skip.

        # For |block| >= t: gain(x) = |{ idx : idx in uncovered, x in all_tsubsets[idx], 
        #   all_tsubsets[idx] \ {x} ⊆ block }|
        # = |{ idx in uncovered : x in sub, sub \ {x} ⊆ block }|

        # Precompute for current block: for each (t-1)-subset R of block, the t-subsets
        # that are R ∪ {x} for some x. We need: for each x not in block, count uncovered
        # t-subsets S with S\{x} ⊆ block, i.e., S\{x} is a (t-1)-subset of block.

        # Efficient approach: 
        # For each x not in block, gain(x) = number of uncovered t-subsets S such that
        # x ∈ S and S \ {x} ⊆ block.
        # S \ {x} ⊆ block means all elements of S except x are in block.
        # So S = {x} ∪ R where R ⊆ block, |R| = t-1.
        # gain(x) = sum over (t-1)-subsets R of block of [ {x}∪R is a t-subset and its idx is in uncovered ]
        # = sum over R in C(block, t-1) of [ idx(sorted(R∪{x})) in uncovered ]

        # Precompute: for the current block, for each (t-1)-subset R of block, 
        # which x values make {x}∪R a valid t-subset (i.e., x not in R, x < v).
        # Actually {x}∪R is always a t-subset as long as x ∉ R.
        # So for each R (t-1-subset of block), for each x ∉ R (and x not in block for gain purposes),
        # check if idx(sorted(R∪{x})) is in uncovered.

        # This is expensive if done naively. Use numpy:
        # For each x not in block:
        #   gain(x) = count of uncovered t-subsets S with x∈S and S\{x}⊆block
        #   = count of uncovered S with x∈S and (S ⊆ block∪{x}) and (|S∩(complement of block)| = 1, that element is x)
        #   = count of uncovered S with x∈S and S∩(complement of block) = {x}
        #   = count of uncovered S with x∈S and S⊆block∪{x}

        # Using point_mask: S has x in it iff point_mask[x, idx].
        # S ⊆ block∪{x}: all elements of S are in block∪{x}.
        # For a given x, the t-subsets S with S⊆block∪{x} and x∈S:
        #   These are t-subsets of block∪{x} that contain x.
        #   block∪{x} has size |block|+1. t-subsets containing x: C(|block|, t-1) of them.
        #   But we need to count only those that are in uncovered.

        # Vectorized: for each x not in block, compute mask of t-subsets S where:
        #   point_mask[x, idx] AND all elements of S are in block∪{x}
        # The second condition: S ⊆ block∪{x} iff for all y not in block∪{x}, y ∉ S.
        # Equivalent: (union over y not in block∪{x} of NOT point_mask[y, idx]) AND point_mask[x, idx]
        # But that's expensive. Alternative: S ⊆ block∪{x} iff S is a t-subset of block∪{x}.
        # The set of t-subsets of block∪{x} is small: C(|block|+1, t). We can enumerate them.

        # Since |block| ≤ k = 8 and t = 5, C(9,5) = 126 at most. Very manageable.
        # So for each x not in block, enumerate all t-subsets of block∪{x} that contain x,
        # and count how many are in uncovered.

        while len(block) < k:
            if time.time() > deadline:
                break

            best_gain = -1
            best_points = []
            for x in range(v):
                if x in block_set:
                    continue

                # Enumerate t-subsets of block∪{x} that contain x
                # block∪{x} = block + [x], size = len(block)+1
                # t-subsets containing x: choose t-1 from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = sub_to_idx.get(sub)
                    if idx is not None and idx in uncovered:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)

            if not best_points:
                break

            chosen = rng.choice(best_points)
            block.append(chosen)
            block_set.add(chosen)

        block.sort()
        blocks.append(tuple(block))

        # Remove covered t-subsets from uncovered
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                uncovered.discard(idx)

    return blocks, uncovered
=======
def _greedy_cover(v, k, t, deadline, rng, all_tsubsets, n_tsubsets, sub_to_idx):
    """
    Greedy covering with vectorized gain computation.
    """
    global t_global
    t_global = t

    uncovered = set(range(n_tsubsets))
    blocks = []

    # Precompute point_mask: point_mask[x, idx] = True if x in all_tsubsets[idx]
    point_mask = np.zeros((v, n_tsubsets), dtype=np.int8)
    for idx, sub in enumerate(all_tsubsets):
        for x in sub:
            point_mask[x, idx] = 1

    # Precompute for each (t-1)-subset of points, the list of t-subset indices
    # that extend it (i.e., t-subsets containing that (t-1)-subset)
    # Key: tuple of sorted (t-1) points -> list of t-subset indices
    tminus1_to_tsubs = {}
    for idx, sub in enumerate(all_tsubsets):
        for i in range(t):
            # (t-1)-subset obtained by removing element i
            rest = sub[:i] + sub[i+1:]
            key = tuple(sorted(rest))
            if key not in tminus1_to_tsubs:
                tminus1_to_tsubs[key] = []
            tminus1_to_tsubs[key].append(idx)

    while uncovered:
        if time.time() > deadline:
            break

        start_idx = rng.choice(list(uncovered))
        start_sub = all_tsubsets[start_idx]
        block = list(start_sub)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                break

            # For each candidate x not in block, compute gain:
            # gain(x) = number of uncovered t-subsets S with x in S and S\{x} is a (t-1)-subset of block
            # = sum over (t-1)-subsets R of block of |uncovered ∩ tminus1_to_tsubs[R] ∩ {S: x in S}|
            # But tminus1_to_tsubs[R] gives all t-subsets containing R.
            # Among those, we need ones that also contain x.
            # A t-subset S in tminus1_to_tsubs[R] contains x iff x is in S.
            # Since R has t-1 elements and S has t elements, S = R ∪ {y} for some y not in R.
            # So x in S iff x = y, i.e., x not in R.
            # If x in R, then S = R ∪ {y} with y ≠ x, so x not in S (since x in R and y ≠ x).
            # Wait: S contains all of R plus one more element y. If x in R, then x in S.
            # So actually for x in R: all t-subsets in tminus1_to_tsubs[R] contain x.
            # For x not in R: t-subsets in tminus1_to_tsubs[R] contain x iff the extra element is x.
            
            # Simpler: for each x not in block:
            #   gain(x) = count of uncovered idx such that x in all_tsubsets[idx] and 
            #             all other elements of all_tsubsets[idx] are in block.
            # This means: all_tsubsets[idx] ⊆ block ∪ {x} and x in all_tsubsets[idx].
            # Equivalently: all_tsubsets[idx] \ {x} ⊆ block and x in all_tsubsets[idx].
            # all_tsubsets[idx] \ {x} is a (t-1)-subset of block.

            # Vectorized approach using point_mask:
            # For candidate x, we want uncovered indices idx where:
            #   point_mask[x, idx] == 1 AND 
            #   for all y in all_tsubsets[idx], y in block or y == x
            # The second condition: sum over y not in block∪{x} of point_mask[y, idx] == 0
            # i.e., for all y not in block and y != x: point_mask[y, idx] == 0

            # Let's use: for each x not in block, compute a boolean mask over n_tsubsets:
            #   mask_x[idx] = point_mask[x, idx] AND (sum of point_mask[y, idx] for y not in block∪{x} == 0)
            # But summing over ~11 points per x is expensive.
            
            # Alternative: precompute for current block the set of "compatible" t-subsets.
            # A t-subset S is compatible with block (for adding x) if S\{x} ⊆ block.
            # Let's enumerate (t-1)-subsets of block and look up their extensions.

            gains = np.zeros(v, dtype=np.int32)
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    key = tuple(sorted(rest))
                    if key in tminus1_to_tsubs:
                        for idx in tminus1_to_tsubs[key]:
                            if idx in uncovered:
                                # Check x is in all_tsubsets[idx]
                                if point_mask[x, idx]:
                                    gain += 1
                gains[x] = gain

            # Find best
            valid_mask = np.array([1 if x not in block_set else 0 for x in range(v)], dtype=bool)
            valid_gains = gains[valid_mask]
            if len(valid_gains) == 0 or valid_gains.max() <= 0:
                break

            best_gain = valid_gains.max()
            best_indices = np.where(valid_gains == best_gain)[0]
            # Map back to actual x values
            valid_xs = np.where(valid_mask)[0]
            best_xs = valid_xs[best_indices]
            chosen = int(rng.choice(best_xs))
            block.append(chosen)
            block_set.add(chosen)

        block.sort()
        blocks.append(tuple(block))

        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub)
            if idx is not None:
                uncovered.discard(idx)

    return blocks, uncovered
>>>>>>> REPLACE

