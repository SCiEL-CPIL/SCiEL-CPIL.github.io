import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    # Bitsets for fast set operations
    tsub_mask = [0] * n_blocks
    tsub_blocks_mask = [0] * n_tsubsets

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask

    for i in range(n_tsubsets):
        tsub_blocks[i] = []
        m = tsub_blocks_mask[i]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_blocks[i].append(idx)
            m ^= lsb

    full_tsub_mask = (1 << n_tsubsets) - 1

    # Precompute conflict masks for Packing LB (independent sets)
    # Two t-subsets conflict if they are covered by the same block
    tsub_conflict_mask = [0] * n_tsubsets
    for b_idx in range(n_blocks):
        mask = tsub_mask[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_conflict_mask[idx] |= mask
            m ^= lsb

    def packing_lb(uncovered_mask):
        """
        Lower bound via maximal independent set of uncovered t-subsets.
        Two t-subsets are independent if no single block covers both.
        """
        count = 0
        remaining = uncovered_mask
        while remaining:
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            count += 1
            remaining &= ~tsub_conflict_mask[idx]
        return count

    def lagrangian_dual_lb(uncovered_mask, n_iter=20, step=0.05):
        """
        Lagrangian dual lower bound via subgradient method.
        We relax coverage constraints with multipliers y >= 0.
        Dual: max_{y>=0} sum_{ts in U} y_ts - sum_{b} max(0, sum_{ts in U∩B_b} y_ts)
        Subgradient for y_ts: 1 - sum_{b: ts in B_b} [1 if sum_{ts' in U∩B_b} y_{ts'} > 0 else 0]
        """
        U = []
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            U.append(idx)
            m ^= lsb

        if not U:
            return 0

        U_set = set(U)
        y = [0.0] * len(U)
        y_idx = {ts: i for i, ts in enumerate(U)}
        best_lb = 0.0

        # Precompute for each block the list of U-subsets it covers
        block_U = [[] for _ in range(n_blocks)]
        for b in range(n_blocks):
            mask = tsub_mask[b] & uncovered_mask
            mm = mask
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                block_U[b].append(idx)
                mm ^= lsb

        for _ in range(n_iter):
            # Compute dual value
            dual_val = sum(y)
            for b in range(n_blocks):
                s = 0.0
                for ts in block_U[b]:
                    s += y[y_idx[ts]]
                if s > 0:
                    dual_val -= s
            if dual_val > best_lb:
                best_lb = dual_val

            # Compute subgradient
            # For each ts in U: grad[ts] = 1 - (# of blocks b where ts in B_b and sum_{ts' in U∩B_b} y_{ts'} > 0)
            # We need for each block b, whether its sum over U is > 0
            block_active = [False] * n_blocks
            for b in range(n_blocks):
                if block_U[b]:
                    s = 0.0
                    for ts in block_U[b]:
                        s += y[y_idx[ts]]
                    if s > 0:
                        block_active[b] = True

            grad = [0.0] * len(U)
            for ts in U:
                cnt = 0
                for b in tsub_blocks[ts]:
                    if block_active[b]:
                        cnt += 1
                grad[y_idx[ts]] = 1.0 - cnt

            # Update y
            for i in range(len(U)):
                y[i] = max(0.0, y[i] + step * grad[i])

        return int(np.floor(best_lb + 1e-9))

    def greedy_cover(seed):
        """
        Standard greedy using bitsets for gain calculation.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []

        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break

            best_block = -1
            best_gain = 0

            for b_idx in range(n_blocks):
                if not (tsub_mask[b_idx] & uncovered_mask):
                    continue
                gain = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    if best_gain == n_tsubsets:
                        break

            if best_block == -1:
                break

            selected.append(best_block)
            uncovered_mask &= ~tsub_mask[best_block]

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 2.0:
                    break
                if b not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                cand_mask = (1 << n_blocks) - 1
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b)

                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 20:
                        break

                if not candidates:
                    continue

                rng.shuffle(candidates)

                swapped = False
                for r in candidates:
                    if r in block_set:
                        continue

                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    def two_for_one_search(blocks, seed):
        """
        Try to replace two blocks with one.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks

        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checks = 100
            for _ in range(n_checks):
                if time.time() > deadline - 2.0:
                    break
                if len(block_set) < 2:
                    break

                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]

                if b1 not in block_set or b2 not in block_set:
                    continue

                required_mask = 0
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)

                if required_mask == 0:
                    continue

                cand_mask = (1 << n_blocks) - 1
                m = required_mask
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                    m ^= lsb

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b1)
                cand_mask &= ~(1 << b2)

                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 5:
                        break

                if not candidates:
                    continue

                rng.shuffle(candidates)
                r = candidates[0]

                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1

                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1

                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1

                improved = True
                break

        return list(block_set)

    def branch_and_bound(uncovered_mask, current_blocks, current_size, best_size, seed):
        """
        Branch-and-bound with Lagrangian dual lower bound.
        """
        if time.time() > deadline - 3.0:
            return False, None

        if uncovered_mask == 0:
            return True, list(current_blocks)

        n_uncovered = uncovered_mask.bit_count()
        if n_uncovered == 0:
            return True, list(current_blocks)

        # Lower bound via Lagrangian dual
        lb = lagrangian_dual_lb(uncovered_mask, n_iter=15, step=0.05)
        
        # Also use packing LB as a complementary bound
        lb_pack = packing_lb(uncovered_mask)
        lb = max(lb, lb_pack)

        if current_size + lb >= best_size:
            return False, None

        # Select the most constrained uncovered t-subset (fewest candidate blocks)
        best_ts = -1
        best_candidates = None
        best_cand_count = float('inf')
        
        m = uncovered_mask
        while m:
            lsb = m & (-m)
            ts_idx = lsb.bit_length() - 1
            candidates = tsub_blocks_mask[ts_idx]
            cand_count = candidates.bit_count()
            if cand_count < best_cand_count:
                best_cand_count = cand_count
                best_ts = ts_idx
                best_candidates = candidates
            if cand_count == 0:
                return False, None
            m ^= lsb

        # Branch on which block to use for best_ts
        candidate_list = []
        m = best_candidates
        while m:
            lsb = m & (-m)
            b_idx = lsb.bit_length() - 1
            if b_idx not in current_blocks:
                cover = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                candidate_list.append((cover, b_idx))
            m ^= lsb
        
        candidate_list.sort(reverse=True)
        
        max_branch = min(8, len(candidate_list))
        
        for cover, b_idx in candidate_list[:max_branch]:
            new_blocks = current_blocks + [b_idx]
            new_uncovered = uncovered_mask & ~tsub_mask[b_idx]
            
            found, sol = branch_and_bound(
                new_uncovered, new_blocks, current_size + 1, best_size, seed
            )
            
            if found and sol is not None:
                best_size = len(sol)
                if best_size <= current_size + 1:
                    return True, sol
        
        return False, None

    def bnb_from_greedy(greedy_sol, seed):
        """
        Use branch-and-bound starting from a greedy solution.
        """
        if time.time() > deadline - 10.0:
            return greedy_sol

        upper_bound = len(greedy_sol)
        
        found, sol = branch_and_bound(
            full_tsub_mask, [], 0, upper_bound, seed
        )
        
        if found and sol is not None:
            return sol
        else:
            return greedy_sol

    best_solution = None
    best_size = float('inf')

    # Run multiple restarts
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        seed = 1000 + i
        greedy_sol = greedy_cover(seed)

        if not greedy_sol:
            continue

        # Local Search
        ls_sol = local_search(greedy_sol, seed)

        # 2-for-1 Search
        t2o_sol = two_for_one_search(ls_sol, seed)

        # Try B&B to potentially improve further
        if time.time() < deadline - 15.0 and len(t2o_sol) > 140:
            bnb_sol = bnb_from_greedy(t2o_sol, seed)
            if bnb_sol is not None:
                t2o_sol = bnb_sol

        if len(t2o_sol) < best_size:
            best_size = len(t2o_sol)
            best_solution = t2o_sol.copy()

    # Final pass: try more aggressive local search on the best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        final_ls = local_search(best_solution, 9999)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_solution = final_ls

    # If no solution found (should not happen), fallback to basic greedy
    if best_solution is None:
        best_solution = greedy_cover(0)

    return [list(all_blocks[idx]) for idx in best_solution]
