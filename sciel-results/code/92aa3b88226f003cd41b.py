"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin before 150s kill
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # For each point, list of t-subset indices containing that point
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        idx = tsub_to_idx[ts]
        for p in ts:
            point_tsubs[p].append(idx)

    # Precompute: for each pair of points (p, q), the t-subsets that contain both
    # This helps in computing overlap-based gains efficiently
    # For t=3, a t-subset {a,b,c} is "completable" by point p if {a,b} or {a,c} or {b,c} 
    # is already in the block. More generally for t-subset S and candidate point p not in S,
    # S is covered by adding p iff S \ {p} subset of current block.
    
    # For efficient gain computation: for a current block B and candidate point p,
    # gain = number of uncovered t-subsets S such that p in S and S\{p} subset B
    # = number of uncovered (t-1)-subsets of B that, together with p, form a t-subset
    
    # Precompute for each (t-1)-subset, which t-subsets extend it
    # For t=3: each pair -> list of triples containing that pair
    if t >= 2:
        # (t-1)-subset -> list of t-subset indices
        tminus1_to_tsubs = {}
        for ts in all_tsubs:
            ts_set = set(ts)
            # For each (t-1)-subset of this t-subset
            for rm in range(t):
                sub = tuple(sorted(ts_set - {ts[rm]}))
                if sub not in tminus1_to_tsubs:
                    tminus1_to_tsubs[sub] = []
                tminus1_to_tsubs[sub].append(tsub_to_idx[ts])
    else:
        tminus1_to_tsubs = {}

    # Precompute pair-to-triple mapping for t=3 speedup
    if t == 3:
        # For each pair (a,b), list of points c such that {a,b,c} is a valid triple
        pair_to_points = {}
        for ts in all_tsubs:
            a, b, c = ts
            for pair in [(a,b), (a,c), (b,c)]:
                key = tuple(sorted(pair))
                if key not in pair_to_points:
                    pair_to_points[key] = []
                remaining = [x for x in ts if x not in pair]
                pair_to_points[key].append(remaining[0])
        
        # For each point p, precompute which pairs it "completes"
        # point_pair_mask[p] = set of pair indices that p completes
        # And pair_to_tsub_idx[pair_idx] = t-subset index
        all_pairs = list(itertools.combinations(range(v), 2))
        pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
        n_pairs = len(all_pairs)
        
        # For each point p, list of (pair_idx, tsub_idx) pairs
        point_completes = [[] for _ in range(v)]
        for ts in all_tsubs:
            ts_idx = tsub_to_idx[ts]
            a, b, c = ts
            pairs_in_ts = [(a,b), (a,c), (b,c)]
            for i, pt in enumerate([a, b, c]):
                other_pairs = [pairs_in_ts[j] for j in range(3) if j != i]
                for pair in other_pairs:
                    pidx = pair_to_idx[tuple(sorted(pair))]
                    point_completes[pt].append((pidx, ts_idx))
        
        # Convert to numpy arrays for fast computation
        point_completes_np = []
        for p in range(v):
            if point_completes[p]:
                pidxs = np.array([x[0] for x in point_completes[p]], dtype=np.int32)
                tidxs = np.array([x[1] for x in point_completes[p]], dtype=np.int32)
                point_completes_np.append((pidxs, tidxs))
            else:
                point_completes_np.append((np.array([], dtype=np.int32), np.array([], dtype=np.int32)))

    def greedy_cover_with_prob_init():
        """Randomized greedy set cover with probabilistic Johnson-scheme-inspired init."""
        coverage = [0] * n_tsubs
        blocks = []
        block_set_list = []  # sets for fast membership
        
        # Track uncovered t-subsets as a boolean numpy array for fast gain computation
        uncovered_mask = np.ones(n_tsubs, dtype=bool)
        uncovered_count = n_tsubs
        
        while uncovered_count > 0:
            if time.time() > deadline:
                return None

            # Find an uncovered t-subset
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            uncovered_idx = uncovered_indices[0]

            # Build a block that covers this t-subset and maximizes new coverage
            seed_tsub = all_tsubs[uncovered_idx]
            block_set = set(seed_tsub)

            # Greedily add points with probabilistic tie-breaking
            for _ in range(k - t):
                if time.time() > deadline:
                    return None

                best_point = None
                best_gain = -1
                best_gain_list = []  # all points with max gain
                candidates = [p for p in range(v) if p not in block_set]

                for p in candidates:
                    # Calculate gain
                    if t == 3:
                        # Fast path: use precomputed pair completion
                        pidxs, tidxs = point_completes_np[p]
                        if len(pidxs) == 0:
                            gain = 0
                        else:
                            # Get pairs in current block
                            block_pairs = set()
                            bl = sorted(block_set)
                            for i in range(len(bl)):
                                for j in range(i+1, len(bl)):
                                    block_pairs.add(tuple(sorted((bl[i], bl[j]))))
                            
                            # Count how many of p's completing pairs are in block
                            # and the corresponding t-subsets are uncovered
                            gain = 0
                            for pidx, tidx in zip(pidxs, tidxs):
                                pair = all_pairs[pidx]
                                if pair in block_pairs and uncovered_mask[tidx]:
                                    gain += 1
                    else:
                        # General case
                        gain = 0
                        if t >= 2:
                            for sub in itertools.combinations(sorted(block_set), t - 1):
                                ts_candidate = tuple(sorted(sub + (p,)))
                                if ts_candidate in tsub_to_idx:
                                    idx = tsub_to_idx[ts_candidate]
                                    if uncovered_mask[idx]:
                                        gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                        best_gain_list = [p]
                    elif gain == best_gain:
                        best_gain_list.append(p)

                if best_point is not None:
                    # Probabilistic selection among tied best points
                    if len(best_gain_list) > 1:
                        weights = []
                        for p in best_gain_list:
                            w = sum(1 for ti in point_tsubs[p] if uncovered_mask[ti])
                            weights.append(w + 1)
                        total_w = sum(weights)
                        r = rng.random() * total_w
                        cum = 0
                        chosen = best_gain_list[0]
                        for p, w in zip(best_gain_list, weights):
                            cum += w
                            if r <= cum:
                                chosen = p
                                break
                        block_set.add(chosen)
                    else:
                        block_set.add(best_point)

            block = sorted(block_set)
            blocks.append(block)
            block_set_list.append(block_set)

            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if uncovered_mask[idx]:
                    uncovered_mask[idx] = False
                    uncovered_count -= 1
                coverage[idx] += 1

        return blocks

    def local_search_remove(blocks):
        """Try to remove redundant blocks."""
        if not blocks:
            return blocks

        # Build coverage map: t-subset -> count of blocks covering it
        coverage_count = [0] * n_tsubs
        block_tsubs = []  # for each block, list of t-subset indices it covers

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        # Try to remove each block if it doesn't break coverage
        keep = [True] * len(blocks)

        for i in range(len(blocks)):
            if not keep[i]:
                continue
            # Check if all t-subsets in this block are covered by other blocks
            can_remove = True
            for tsub_idx in block_tsubs[i]:
                if coverage_count[tsub_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                keep[i] = False
                # Update coverage
                for tsub_idx in block_tsubs[i]:
                    coverage_count[tsub_idx] -= 1

        return [blocks[i] for i in range(len(blocks)) if keep[i]]

    def overlap_aware_local_search(blocks):
        """Local search that optimizes block overlap to reduce total count.
        
        Strategy: For each block, try to replace it with a different block that
        covers all the same unique t-subsets (those only covered by this block)
        while maximizing overlap with other blocks (to make those other blocks
        potentially removable).
        """
        if not blocks:
            return blocks

        coverage_count = [0] * n_tsubs
        block_tsubs = []

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        improved = True
        max_iterations = 15
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks

                # Identify t-subsets uniquely covered by block i
                unique_tsubs = [ts_idx for ts_idx in block_tsubs[i] if coverage_count[ts_idx] == 1]

                if not unique_tsubs:
                    # Block i is redundant
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    # Rebuild
                    coverage_count = [0] * n_tsubs
                    block_tsubs = []
                    for block in blocks:
                        tsubs_in_block = []
                        for ts in itertools.combinations(block, t):
                            idx = tsub_to_idx[ts]
                            coverage_count[idx] += 1
                            tsubs_in_block.append(idx)
                        block_tsubs.append(tsubs_in_block)
                    break

                # Try to find a replacement block that covers all unique_tsubs
                # and maximizes overlap with other blocks
                
                # Collect all points that must be in the new block
                required_points = set()
                for ts_idx in unique_tsubs:
                    required_points.update(all_tsubs[ts_idx])

                if len(required_points) > k:
                    continue  # Can't fit all required points in one block

                # Try to extend required_points to a full block of size k
                # that maximizes overlap (covers most already-covered t-subsets)
                
                best_new_block = None
                best_overlap_score = -1

                # Try several random extensions
                num_trials = min(30, max(1, int((deadline - time.time()) // 1)))
                for trial in range(num_trials):
                    new_block_set = set(required_points)
                    
                    # Greedily add points to maximize overlap with existing blocks
                    for _ in range(k - len(new_block_set)):
                        best_p = None
                        best_score = -1
                        candidates = [p for p in range(v) if p not in new_block_set]
                        rng.shuffle(candidates)
                        
                        for p in candidates:
                            # Score: number of t-subsets in (new_block_set + {p}) 
                            # that are already covered by other blocks
                            score = 0
                            test_set = new_block_set | {p}
                            for ts in itertools.combinations(sorted(test_set), t):
                                idx = tsub_to_idx.get(ts, -1)
                                if idx >= 0 and coverage_count[idx] > 1:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_p = p
                        if best_p is not None:
                            new_block_set.add(best_p)
                    
                    # Verify the new block covers all unique_tsubs
                    valid = True
                    for ts_idx in unique_tsubs:
                        ts_set = set(all_tsubs[ts_idx])
                        if not ts_set.issubset(new_block_set):
                            valid = False
                            break
                    
                    if not valid:
                        continue

                    # Calculate overlap score: how many t-subsets of new block
                    # are already covered (by blocks other than i)
                    overlap = 0
                    for ts in itertools.combinations(sorted(new_block_set), t):
                        idx = tsub_to_idx.get(ts, -1)
                        if idx >= 0 and coverage_count[idx] > 1:
                            overlap += 1
                    
                    if overlap > best_overlap_score:
                        best_overlap_score = overlap
                        best_new_block = sorted(new_block_set)

                if best_new_block is not None:
                    # Check if replacing block i with best_new_block allows removing other blocks
                    old_block_tsubs = block_tsubs[i]
                    
                    # Remove old block's coverage
                    for ts_idx in old_block_tsubs:
                        coverage_count[ts_idx] -= 1
                    
                    # Add new block's coverage
                    new_block_tsubs = []
                    for ts in itertools.combinations(best_new_block, t):
                        idx = tsub_to_idx[ts]
                        coverage_count[idx] += 1
                        new_block_tsubs.append(idx)
                    
                    # Now check if any other blocks become redundant
                    additional_removals = 0
                    new_keep = [True] * len(blocks)
                    
                    for j in range(len(blocks)):
                        if j == i or not new_keep[j]:
                            continue
                        can_remove = True
                        for ts_idx in block_tsubs[j]:
                            if coverage_count[ts_idx] <= 1:
                                can_remove = False
                                break
                        if can_remove:
                            new_keep[j] = False
                            additional_removals += 1
                            for ts_idx in block_tsubs[j]:
                                coverage_count[ts_idx] -= 1
                    
                    if additional_removals > 0:
                        # Apply the replacement and removals
                        blocks[i] = best_new_block
                        block_tsubs[i] = new_block_tsubs
                        
                        # Remove redundant blocks (in reverse order to preserve indices)
                        for j in range(len(blocks) - 1, -1, -1):
                            if not new_keep[j]:
                                blocks.pop(j)
                                block_tsubs.pop(j)
                        
                        improved = True
                        # Rebuild coverage
                        coverage_count = [0] * n_tsubs
                        block_tsubs = []
                        for block in blocks:
                            tsubs_in_block = []
                            for ts in itertools.combinations(block, t):
                                idx = tsub_to_idx[ts]
                                coverage_count[idx] += 1
                                tsubs_in_block.append(idx)
                            block_tsubs.append(tsubs_in_block)
                        break
                    else:
                        # Revert
                        for ts_idx in old_block_tsubs:
                            coverage_count[ts_idx] += 1
                        for ts_idx in new_block_tsubs:
                            coverage_count[ts_idx] -= 1

        return blocks

    def final_polish(blocks):
        """Final high-precision polish pass: try pairwise block replacements."""
        if not blocks:
            return blocks

        coverage_count = [0] * n_tsubs
        block_tsubs = []

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        improved = True
        max_polish_iterations = 20
        polish_iteration = 0
        while improved and polish_iteration < max_polish_iterations:
            improved = False
            polish_iteration += 1
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks

                # Try to find a block that can replace block i and allow removing another block
                unique_to_i = set(ts_idx for ts_idx in block_tsubs[i] if coverage_count[ts_idx] == 1)
                
                if not unique_to_i:
                    # Block i is redundant
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    coverage_count = [0] * n_tsubs
                    block_tsubs = []
                    for block in blocks:
                        tsubs_in_block = []
                        for ts in itertools.combinations(block, t):
                            idx = tsub_to_idx[ts]
                            coverage_count[idx] += 1
                            tsubs_in_block.append(idx)
                        block_tsubs.append(tsubs_in_block)
                    break

                # For each other block j, check if we can merge their unique requirements
                for j in range(len(blocks)):
                    if i == j:
                        continue
                    if time.time() > deadline:
                        return blocks

                    unique_to_j = set(ts_idx for ts_idx in block_tsubs[j] if coverage_count[ts_idx] == 1)
                    
                    # Points required from block i's unique t-subsets
                    req_i = set()
                    for ts_idx in unique_to_i:
                        req_i.update(all_tsubs[ts_idx])
                    
                    # Points required from block j's unique t-subsets
                    req_j = set()
                    for ts_idx in unique_to_j:
                        req_j.update(all_tsubs[ts_idx])
                    
                    # If one block can cover both req_i and req_j, we save 1.
                    combined_req = req_i | req_j
                    if len(combined_req) <= k:
                        # Try to form a single block from combined_req
                        new_block_set = set(combined_req)
                        # Extend to size k
                        for _ in range(k - len(new_block_set)):
                            best_p = None
                            best_score = -1
                            for p in range(v):
                                if p in new_block_set:
                                    continue
                                score = 0
                                test_set = new_block_set | {p}
                                for ts in itertools.combinations(sorted(test_set), t):
                                    idx = tsub_to_idx.get(ts, -1)
                                    if idx >= 0 and coverage_count[idx] > 1:
                                        score += 1
                                if score > best_score:
                                    best_score = score
                                    best_p = p
                            if best_p is not None:
                                new_block_set.add(best_p)
                        
                        new_block = sorted(new_block_set)
                        
                        # Verify coverage of all unique t-subsets from both i and j
                        valid = True
                        for ts_idx in unique_to_i | unique_to_j:
                            if not set(all_tsubs[ts_idx]).issubset(new_block_set):
                                valid = False
                                break
                        
                        if valid:
                            # Temporarily remove both blocks
                            for ts_idx in block_tsubs[i]:
                                coverage_count[ts_idx] -= 1
                            for ts_idx in block_tsubs[j]:
                                coverage_count[ts_idx] -= 1
                            
                            # Add new block
                            for ts in itertools.combinations(new_block, t):
                                idx = tsub_to_idx[ts]
                                coverage_count[idx] += 1
                            
                            # Check if all t-subsets are still covered
                            all_covered = all(coverage_count[idx] > 0 for idx in range(n_tsubs))
                            
                            if all_covered:
                                # Apply: replace i and j with one block
                                # Remove j first (higher index)
                                blocks.pop(j)
                                block_tsubs.pop(j)
                                blocks[i] = new_block
                                # Rebuild block_tsubs for i
                                block_tsubs[i] = [tsub_to_idx[ts] for ts in itertools.combinations(new_block, t)]
                                improved = True
                                # Rebuild everything
                                coverage_count = [0] * n_tsubs
                                block_tsubs = []
                                for block in blocks:
                                    tsubs_in_block = []
                                    for ts in itertools.combinations(block, t):
                                        idx = tsub_to_idx[ts]
                                        coverage_count[idx] += 1
                                        tsubs_in_block.append(idx)
                                    block_tsubs.append(tsubs_in_block)
                                break
                            else:
                                # Revert
                                for ts_idx in block_tsubs[i]:
                                    coverage_count[ts_idx] += 1
                                for ts_idx in block_tsubs[j]:
                                    coverage_count[ts_idx] += 1
                                for ts in itertools.combinations(new_block, t):
                                    idx = tsub_to_idx[ts]
                                    coverage_count[idx] -= 1
                    
                    if improved:
                        break
                
                if improved:
                    break

        return blocks

    best_blocks = None
    best_count = float('inf')

    # Run greedy with randomization and local search
    num_restarts = 6
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break

        blocks = greedy_cover_with_prob_init()
        if blocks is None:
            continue

        # Local search to remove redundant blocks
        blocks = local_search_remove(blocks)

        # Overlap-aware local search
        if time.time() < deadline - 25:
            blocks = overlap_aware_local_search(blocks)
            blocks = local_search_remove(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()

    # Final polish on the best solution
    if best_blocks is not None and time.time() < deadline - 15:
        improved_blocks = final_polish(best_blocks.copy())
        if improved_blocks is not None and len(improved_blocks) < best_count:
            best_count = len(improved_blocks)
            best_blocks = improved_blocks

    if best_blocks is not None:
        return best_blocks

    # Fallback: simple greedy
    blocks = greedy_cover_with_prob_init()
    if blocks is not None:
        return blocks

    # Last resort
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
# EVOLVE-BLOCK-END
