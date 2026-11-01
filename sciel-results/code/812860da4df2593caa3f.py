"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    total_tsubs = len(all_tsubs)

    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubs)}

    # Track uncovered t-subsets
    uncovered = set(range(total_tsubs))  # indices of uncovered t-subsets

    # Point weights: initially uniform, will be updated based on uncovered frequency
    point_weights = [1.0] * v

    # Count how many times each point appears in uncovered t-subsets
    # We'll update this incrementally
    point_uncovered_count = [0] * v
    for tsub in all_tsubs:
        for p in tsub:
            point_uncovered_count[p] += 1

    blocks = []
    seen_blocks = set()

    # Precompute for each point, the list of t-subsets containing it (for fast gain computation)
    # Actually, for efficiency, let's precompute for each (t-1)-subset, the points that complete it to a t-subset
    # And for each t-subset, which points are in it

    # For gain computation: when we have a partial block B (size < k),
    # adding point x covers all t-subsets that are subsets of B ∪ {x} and contain x.
    # i.e., t-subsets of form S ∪ {x} where S ⊂ B, |S| = t-1

    # Let's precompute all (t-1)-subsets and map them to the points that extend them
    tminus1_subs = list(itertools.combinations(range(v), t - 1))
    # For each (t-1)-sub, the set of points x not in it such that (t-1)-sub + x is a valid t-sub
    # This is just all points not in the (t-1)-sub

    # Better approach: for each t-subset, store its index
    # For gain(x, block): count uncovered t-subsets of form (t-1)-sub from block + x

    # Let's precompute: for each point x, a dict mapping (t-1)-subset (as tuple) to t-subset index
    # But that's too large. Instead, compute gain on the fly but efficiently.

    # Alternative: precompute for each t-subset, the set of points in it (already have from all_tsubs)
    # And for each point x, list of t-subsets containing x
    point_tsubs = [[] for _ in range(v)]
    for i, tsub in enumerate(all_tsubs):
        for p in tsub:
            point_tsubs[p].append(i)

    # For a partial block B, gain(x) = number of uncovered t-subsets T such that T ⊆ B∪{x} and x ∈ T
    # T = S ∪ {x} where S ⊆ B, |S| = t-1
    # So: for each (t-1)-subset S of B, check if S∪{x} is a t-subset (it is if x not in S)
    # and if its index is in uncovered

    # Precompute: map from (t-1)-subset tuple to list of (x, tsub_idx) for x not in S
    # Actually simpler: for each (t-1)-subset S, the t-subsets extending it are S∪{x} for x not in S
    # We can precompute a dict: (t-1)-sub_tuple -> dict of x -> tsub_idx
    tminus1_extend = {}
    for tsub in all_tsubs:
        for i in range(t):
            S = tsub[:i] + tsub[i+1:]
            x = tsub[i]
            if S not in tminus1_extend:
                tminus1_extend[S] = {}
            tminus1_extend[S][x] = tsub_to_idx[tsub]

    def compute_gain(block, x, uncovered_set):
        """Compute how many uncovered t-subsets would be covered by adding x to block."""
        gain = 0
        B = set(block)
        if x in B:
            return 0
        # For each (t-1)-subset of B, check if x extends it to a covered t-subset
        if len(B) >= t - 1:
            for S in itertools.combinations(block, t - 1):
                S_tuple = S
                if S_tuple in tminus1_extend:
                    ext_dict = tminus1_extend[S_tuple]
                    if x in ext_dict:
                        idx = ext_dict[x]
                        if idx in uncovered_set:
                            gain += 1
        return gain

    # Adaptive weights: update based on uncovered t-subsets
    def update_weights(uncovered_set, point_uncovered_count, point_weights):
        """Update point weights based on current uncovered distribution."""
        if not uncovered_set:
            return
        # Count uncovered appearances per point
        new_counts = [0] * v
        for idx in uncovered_set:
            tsub = all_tsubs[idx]
            for p in tsub:
                new_counts[p] += 1
        # Update weights: weight proportional to count, with smoothing
        total = sum(new_counts)
        if total == 0:
            return
        for p in range(v):
            # Blend old and new
            point_weights[p] = 0.7 * point_weights[p] + 0.3 * (new_counts[p] / total)
        # Normalize
        wsum = sum(point_weights)
        if wsum > 0:
            for p in range(v):
                point_weights[p] /= wsum

    # Randomized greedy with adaptive weights
    rng = random.Random(42)

    while uncovered and time.time() < deadline:
        # Choose starting point: pick from uncovered t-subsets, bias by weight
        # Pick a random uncovered t-subset, then pick a random point from it
        uncovered_list = list(uncovered)
        if not uncovered_list:
            break
        
        # Weighted selection of t-subset based on point weights
        # Simpler: pick random uncovered t-subset
        chosen_idx = uncovered_list[rng.randrange(len(uncovered_list))]
        chosen_tsub = all_tsubs[chosen_idx]
        
        # Start block with a weighted random point from the chosen t-subset
        pts_in_tsub = list(chosen_tsub)
        weights_in_tsub = [point_weights[p] for p in pts_in_tsub]
        total_w = sum(weights_in_tsub)
        r = rng.random() * total_w
        cum = 0
        start_point = pts_in_tsub[0]
        for p, w in zip(pts_in_tsub, weights_in_tsub):
            cum += w
            if r <= cum:
                start_point = p
                break
        
        block = [start_point]
        
        # Greedily add points with best gain, but with some randomness
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            
            # Compute gains
            gains = []
            for x in candidates:
                g = compute_gain(block, x, uncovered)
                gains.append(g)
            
            # Find max gain
            max_gain = max(gains)
            
            if max_gain == 0:
                # No point improves coverage; pick weighted random
                weights = [point_weights[x] for x in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                for x, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        block.append(x)
                        break
                else:
                    block.append(candidates[-1])
            else:
                # Pick among top candidates with some randomness
                # Use softmax-like selection
                top_candidates = [x for x, g in zip(candidates, gains) if g >= max_gain * 0.8]
                if len(top_candidates) > 1:
                    weights = [point_weights[x] for x in top_candidates]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    for x, w in zip(top_candidates, weights):
                        cum += w
                        if r <= cum:
                            block.append(x)
                            break
                    else:
                        block.append(top_candidates[-1])
                else:
                    block.append(top_candidates[0])
        
        block.sort()
        block_tuple = tuple(block)
        
        if block_tuple not in seen_blocks:
            seen_blocks.add(block_tuple)
            blocks.append(block)
            
            # Update uncovered
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                uncovered.discard(idx)
        
        # Update weights periodically
        if len(blocks) % 10 == 0:
            update_weights(uncovered, point_uncovered_count, point_weights)
        
        # Check deadline
        if time.time() > deadline - 5:
            break

    # Final: if we haven't covered everything and have time, do pure greedy on remaining
    if uncovered and time.time() < deadline - 2:
        while uncovered and time.time() < deadline - 2:
            # Pure greedy: pick uncovered t-subset, build best block
            uncovered_list = list(uncovered)
            if not uncovered_list:
                break
            chosen_idx = uncovered_list[0]
            chosen_tsub = all_tsubs[chosen_idx]
            block = list(chosen_tsub[:1])
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                best_x = max(candidates, key=lambda x: compute_gain(block, x, uncovered))
                block.append(best_x)
            
            block.sort()
            block_tuple = tuple(block)
            if block_tuple not in seen_blocks:
                seen_blocks.add(block_tuple)
                blocks.append(block)
                for tsub in itertools.combinations(block, t):
                    idx = tsub_to_idx[tsub]
                    uncovered.discard(idx)

    return blocks
# EVOLVE-BLOCK-END
