import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, compute the triplet IDs it covers
    # Store as a flat numpy array + offsets for efficient access
    block_triplet_ids = []
    for block in all_blocks:
        tups = itertools.combinations(block, t)
        t_ids = np.array([triplet_to_id[tup] for tup in tups], dtype=np.int32)
        block_triplet_ids.append(t_ids)

    # Build triplet_to_blocks: for each triplet, list of block indices containing it
    triplet_to_blocks = [[] for _ in range(n_triplets)]
    for b_idx, t_ids in enumerate(block_triplet_ids):
        for tid in t_ids:
            triplet_to_blocks[tid].append(b_idx)

    # === Global Greedy Set Cover ===
    uncovered = np.ones(n_triplets, dtype=bool)
    solution_indices = []

    # Precompute: for each block, a numpy array of its triplet ids (already done above)
    # We need to efficiently compute gain = sum(uncovered[block_triplet_ids[b]])
    # With 346k blocks and ~35 triplets each, we can vectorize partially.

    # Strategy: pick the uncovered triplet with the fewest candidate blocks (most constrained),
    # then among those candidates, pick the one with highest gain.
    # This is a "most-constrained-first" heuristic that often outperforms random pick.

    while uncovered.any():
        if time.time() > deadline - 15:
            break

        # Find all uncovered triplets
        unc_mask = uncovered
        unc_indices = np.where(unc_mask)[0]
        if len(unc_indices) == 0:
            break

        # Pick the uncovered triplet with fewest candidate blocks (most constrained)
        # To avoid O(n_triplets) per iteration, sample a subset
        if len(unc_indices) > 200:
            sample = np.random.choice(unc_indices, size=200, replace=False)
        else:
            sample = unc_indices

        best_tid = -1
        min_candidates = float('inf')
        for tid in sample:
            n_cands = len(triplet_to_blocks[tid])
            if n_cands < min_candidates:
                min_candidates = n_cands
                best_tid = tid

        candidates = triplet_to_blocks[best_tid]
        if not candidates:
            # Should not happen; pick any uncovered
            best_tid = unc_indices[0]
            candidates = triplet_to_blocks[best_tid]
            if not candidates:
                break

        # Among candidates, pick the one with highest gain
        best_block_idx = -1
        best_score = -1
        for b_idx in candidates:
            t_ids = block_triplet_ids[b_idx]
            score = np.sum(uncovered[t_ids])
            if score > best_score:
                best_score = score
                best_block_idx = b_idx
                if best_score == len(t_ids):
                    break  # Can't do better

        if best_block_idx != -1:
            solution_indices.append(best_block_idx)
            uncovered[block_triplet_ids[best_block_idx]] = False

    # === Redundancy Removal ===
    # Count how many times each triplet is covered
    coverage_count = np.zeros(n_triplets, dtype=np.int32)
    for idx in solution_indices:
        coverage_count[block_triplet_ids[idx]] += 1

    changed = True
    while changed:
        if time.time() > deadline - 10:
            break
        changed = False
        new_indices = []
        for idx in solution_indices:
            t_ids = block_triplet_ids[idx]
            # If any triplet in this block has coverage_count == 1, it's essential
            if np.any(coverage_count[t_ids] == 1):
                new_indices.append(idx)
            else:
                # Remove it
                coverage_count[t_ids] -= 1
                changed = True
        solution_indices = new_indices

    # === Block Merging: try to replace 2 blocks with 1 block ===
    # If two blocks B1, B2 in solution have B1 ∪ B2 ⊆ some block B3 (size k),
    # and all triplets covered by B1 or B2 are also covered by B3 ∪ (other blocks),
    # then we can replace B1, B2 with B3.
    # This is expensive; do it only if time allows.
    if time.time() < deadline - 20:
        sol_set = set(solution_indices)
        # For each pair of blocks in solution, check if their union fits in a k-block
        # and if replacing them maintains coverage
        # This is O(n_sol^2 * k) which could be large; limit it
        n_sol = len(solution_indices)
        if n_sol < 200:  # Only do this if solution is small enough
            # Build a set of all points in each solution block
            sol_points = [set(all_blocks[idx]) for idx in solution_indices]

            # Precompute: for each pair (i,j), check if there exists a block containing sol_points[i] ∪ sol_points[j]
            # That means |union| <= k and the union is a subset of some k-block
            merged = False
            for i in range(n_sol):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, n_sol):
                    union_pts = sol_points[i] | sol_points[j]
                    if len(union_pts) > k:
                        continue
                    # Check if there's a block containing all points in union_pts
                    # A k-block contains union_pts iff union_pts ⊆ block
                    # Find all blocks containing union_pts:
                    # Use the fact that if |union_pts| >= t, we can check via triplets
                    # Simpler: iterate through all_blocks and check
                    # But that's 346k checks per pair... too slow.
                    # Alternative: precompute for each subset of size <= k, which blocks contain it.
                    # Too expensive. Skip for now.
                    pass

    # === Final verification and repair ===
    final_blocks = [all_blocks[i] for i in solution_indices]

    # Check coverage
    covered = set()
    for block in final_blocks:
        for triplet in itertools.combinations(block, t):
            covered.add(triplet)

    if len(covered) < n_triplets:
        # Repair: add blocks for missing triplets
        missing = set(all_triplets) - covered
        for triplet in missing:
            tid = triplet_to_id[triplet]
            if triplet_to_blocks[tid]:
                b_idx = triplet_to_blocks[tid][0]
                final_blocks.append(all_blocks[b_idx])
            else:
                # Should not happen; construct a block containing this triplet
                pts = list(triplet)
                remaining = [p for p in range(v) if p not in pts]
                random.shuffle(remaining)
                pts.extend(remaining[:k - len(pts)])
                final_blocks.append(tuple(sorted(pts)))

    return final_blocks


if __name__ == "__main__":
    pass
