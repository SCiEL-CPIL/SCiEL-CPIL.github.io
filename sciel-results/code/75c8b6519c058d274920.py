import numpy as np
import itertools
import time
import random

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_MILP = True
except Exception:
    _HAS_MILP = False

def construct(v, k, t):
    start_time = time.time()
    time_limit = 118.0  # Safety margin
    
    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute t-subset indices for each k-block
    # shape: (n_ks, C(k,t))
    t_count = 1
    for i in range(t):
        t_count *= (k - i)
        t_count //= (i + 1)
        
    sub_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            sub_indices[i, j] = idx_map[sub]
            
    best_solution = None
    best_score = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        selected_blocks = []
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        
        while uncovered.sum() > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None
            
            # Vectorized gain calculation
            # uncovered[sub_indices] is a boolean array of shape (n_ks, t_count)
            # Summing along axis 1 gives the number of newly covered subsets for each block
            gains = uncovered[sub_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if uncovered > 0 and we have valid blocks
            
            candidates = np.where(gains == max_gain)[0]
            best = random.choice(candidates)
            
            selected_blocks.append(best)
            subs = sub_indices[best]
            coverage_count[subs] += 1
            uncovered[subs] = False
            
        return selected_blocks

    def reduce_cover(blocks, seed=0):
        random.seed(seed)
        
        if not blocks:
            return []
            
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1
            
        current_blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            
            # Calculate min coverage for each block's subsets
            # This helps prioritize blocks that are "redundant"
            # min_cov[b] = min(coverage_count[sub_indices[b]])
            # If min_cov[b] > 1, it is safe to remove.
            
            # We can just iterate and check. To be faster, we can sort.
            # But sorting every iteration is O(N log N). N is ~70. It's fine.
            
            # Calculate safety score: how "redundant" is the block?
            # A block is redundant if all its subsets are covered by others.
            # Let's just try to remove blocks in a random order or sorted by "uniqueness".
            
            # Heuristic: Remove blocks that cover the most "common" subsets first?
            # Or just random shuffle to avoid bias.
            
            order = list(range(len(current_blocks)))
            random.shuffle(order)
            
            new_blocks = []
            for idx in order:
                b = current_blocks[idx]
                subs = sub_indices[b]
                # Check if all subsets in this block are covered by other blocks
                # i.e., coverage_count[subs] > 1
                if np.all(coverage_count[subs] > 1):
                    # Safe to remove
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_blocks.append(b)
                    
            current_blocks = new_blocks
            
        return current_blocks

    def polish(blocks, seed=0):
        """
        Try to improve the solution by swapping blocks and removing redundant ones.
        Hill climbing with swap moves.
        """
        random.seed(seed)
        if not blocks:
            return blocks

        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1

        current_blocks = list(blocks)
        n_blocks = len(current_blocks)

        # First do a reduction pass
        current_blocks = reduce_cover(current_blocks, seed + 500)
        if not current_blocks:
            return []

        # Rebuild coverage count after reduction
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in current_blocks:
            coverage_count[sub_indices[b]] += 1

        n_blocks = len(current_blocks)
        block_set = set(current_blocks)

        improved = True
        max_iters = 200
        iter_count = 0

        while improved and iter_count < max_iters:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            iter_count += 1

            n_blocks = len(current_blocks)
            if n_blocks == 0:
                break

            # Try swap moves: remove one block, add another
            # Randomly pick a block to try removing
            order = list(range(n_blocks))
            random.shuffle(order)

            for idx in order:
                if time.time() - start_time > time_limit - 5.0:
                    break

                b = current_blocks[idx]

                # Temporarily remove block b
                coverage_count[sub_indices[b]] -= 1

                # Find uncovered triples
                uncovered_mask = (coverage_count == 0)
                n_uncovered = int(np.sum(uncovered_mask))

                if n_uncovered == 0:
                    # Block was redundant - remove it
                    current_blocks.pop(idx)
                    block_set.discard(b)
                    improved = True
                    # Rebuild block_set and coverage
                    block_set = set(current_blocks)
                    # coverage_count already updated
                    break

                # Try to find a single block that covers all uncovered triples
                # and is not already in the set
                gains = uncovered_mask[sub_indices].sum(axis=1)

                # We need a block that covers ALL uncovered triples
                # So we need gains[i] == n_uncovered for some i
                full_cover_candidates = np.where(gains == n_uncovered)[0]

                # Filter out blocks already in the set
                valid_candidates = [c for c in full_cover_candidates if c not in block_set]

                if valid_candidates:
                    # Pick a candidate that maximizes "overlap" with existing blocks
                    # to help future reductions
                    # For simplicity, pick randomly
                    new_block = random.choice(valid_candidates)

                    # Perform the swap
                    current_blocks[idx] = new_block
                    block_set.discard(b)
                    block_set.add(new_block)
                    coverage_count[sub_indices[new_block]] += 1
                    improved = True
                    break
                else:
                    # No single block can cover all uncovered - revert
                    coverage_count[sub_indices[b]] += 1
                    continue

        return current_blocks

    def ilp_complete(covered_blocks, seed=0):
        """
        Given a set of already-selected blocks (indices), find a minimal set of
        ADDITIONAL blocks to cover the remaining uncovered t-subsets.
        Uses scipy.optimize.milp on the residual instance.
        Returns the list of additional block indices, or None on failure.
        """
        if not _HAS_MILP:
            return None

        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in covered_blocks:
            coverage_count[sub_indices[b]] += 1

        uncovered_mask = (coverage_count == 0)
        n_uncovered = int(np.sum(uncovered_mask))
        if n_uncovered == 0:
            return []

        # Build the residual incidence matrix: rows = uncovered triples, cols = all blocks
        # A[i, j] = 1 if block j covers uncovered-triple i
        # We only need columns (blocks) that cover at least one uncovered triple.
        # To keep the ILP small, restrict to blocks that cover >= 2 uncovered triples
        # (blocks covering only 1 are still needed sometimes, so keep >=1).
        A_full = sub_indices.T  # shape (n_ts, n_ks)
        A_res = A_full[uncovered_mask, :]  # shape (n_uncovered, n_ks)

        # Count how many uncovered triples each block covers
        col_counts = A_res.sum(axis=0)
        # Keep blocks that cover at least 1 uncovered triple
        active_cols = np.where(col_counts >= 1)[0]
        if len(active_cols) == 0:
            return []

        A = A_res[:, active_cols]  # shape (n_uncovered, m)
        m = A.shape[1]

        # Objective: minimize sum x_j
        c = np.ones(m, dtype=float)

        # Constraints: A x >= 1 (each uncovered triple covered at least once)
        # milp uses LinearConstraint with (lb, ub); we need A x >= 1 -> A x >= 1, no upper bound
        # LinearConstraint(A, lb=1, ub=inf)
        constraint = LinearConstraint(A, lb=np.ones(n_uncovered), ub=np.inf)
        bounds = Bounds(lb=0, ub=1)

        # Time limit for the solver (seconds)
        # We'll give it a modest budget; if it fails, we fall back to greedy
        result = milp(c=c, constraints=[constraint], integrality=np.ones(m, dtype=int),
                      bounds=bounds, options={'time_limit': 5.0, 'presolve': True})

        if result.status == 0:
            x = result.x
            selected = [int(active_cols[i]) for i in range(m) if x[i] > 0.5]
            return selected
        else:
            return None

    def greedy_complete(uncovered_mask, seed=0):
        """
        Greedy completion: given a boolean mask of uncovered triples,
        greedily add blocks to cover them. Returns list of block indices added.
        """
        random.seed(seed)
        remaining = uncovered_mask.copy()
        added = []
        while remaining.sum() > 0:
            if time.time() - start_time > time_limit - 3.0:
                break
            gains = remaining[sub_indices].sum(axis=1)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best = random.choice(candidates)
            added.append(int(best))
            remaining[sub_indices[best]] = False
        return added

    # Main search loop
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks = greedy_cover(seed)

        if blocks is None:
            break

        reduced = reduce_cover(blocks, seed + 100)
        polished = polish(reduced, seed + 200)

        if polished is not None and len(polished) < best_score:
            best_score = len(polished)
            best_solution = list(polished)

        # Greedy-Then-Exact Hybrid: try ILP completion on the residual
        if polished is not None and time.time() - start_time < time_limit - 15.0:
            # Build coverage for the polished solution
            cov = np.zeros(n_ts, dtype=np.int32)
            for b in polished:
                cov[sub_indices[b]] += 1
            unc = (cov == 0)
            n_unc = int(unc.sum())

            if n_unc > 0:
                # Try ILP completion
                extra = ilp_complete(polished, seed + 300)
                if extra is not None:
                    candidate = list(polished) + extra
                    # Verify coverage
                    cov2 = np.zeros(n_ts, dtype=np.int32)
                    for b in candidate:
                        cov2[sub_indices[b]] += 1
                    if np.all(cov2 >= 1):
                        # Reduce and polish the combined solution
                        cand_reduced = reduce_cover(candidate, seed + 400)
                        cand_polished = polish(cand_reduced, seed + 500)
                        if cand_polished is not None and len(cand_polished) < best_score:
                            best_score = len(cand_polished)
                            best_solution = list(cand_polished)
                else:
                    # ILP failed or timed out; try greedy completion as fallback
                    extra_g = greedy_complete(unc, seed + 600)
                    if extra_g:
                        candidate = list(polished) + extra_g
                        cov2 = np.zeros(n_ts, dtype=np.int32)
                        for b in candidate:
                            cov2[sub_indices[b]] += 1
                        if np.all(cov2 >= 1):
                            cand_reduced = reduce_cover(candidate, seed + 700)
                            cand_polished = polish(cand_reduced, seed + 800)
                            if cand_polished is not None and len(cand_polished) < best_score:
                                best_score = len(cand_polished)
                                best_solution = list(cand_polished)

        restart_count += 1

    if best_solution is None:
        # Fallback: run one greedy without time limit check inside (should be fast)
        blocks = greedy_cover(0)
        if blocks:
            best_solution = reduce_cover(blocks, 1)
            best_solution = polish(best_solution, 2)
            
    if best_solution is None:
        return []

    # Convert indices to actual blocks
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
