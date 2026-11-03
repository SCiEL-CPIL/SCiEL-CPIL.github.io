"""Combined: Lazy Greedy (submodular) + Lagrangian-guided WalkSAT + pair-replacement refinement.
Uses submodular lazy greedy for initial construction, then Lagrangian-guided stochastic local
search with pair-replacement moves for structural improvements."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets and their covered t-subset indices
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    block_covers = []
    for ks in k_subs:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_sub_index[ts])
        block_covers.append(covered)

    # Inverse: t-subset -> list of blocks covering it
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, cov in enumerate(block_covers):
        for ti in cov:
            t_to_blocks[ti].append(bi)

    # Convert to numpy for speed where possible
    block_covers_np = [np.array(c, dtype=np.int32) for c in block_covers]
    t_to_blocks_np = [np.array(b, dtype=np.int32) for b in t_to_blocks]

    # ---------------- Lazy Greedy with submodular marginal gains ----------------
    def lazy_greedy_cover(seed=0, max_restarts=8):
        """
        Lazy greedy for maximum coverage: maintains a priority queue of blocks
        with their last computed marginal gain. Only recompute gain when popped.
        This exploits submodularity: if a block's gain hasn't changed since last
        computation, it's still valid.
        """
        rng = random.Random(seed)
        best_blocks = None
        best_size = n_k + 1

        for restart in range(max_restarts):
            if time.time() > deadline - 40.0:
                break

            uncovered = np.ones(n_t, dtype=bool)
            selected = []
            # Lazy queue: (gain, block_idx, last_computed_gain)
            # We'll use a simple approach: maintain gains array and a set of "dirty" blocks
            gains = np.zeros(n_k, dtype=np.int32)
            dirty = np.ones(n_k, dtype=bool)  # All blocks need gain computation initially

            # Initial gains
            for bi in range(n_k):
                gains[bi] = np.sum(uncovered[block_covers_np[bi]])

            while np.any(uncovered):
                if time.time() > deadline - 35.0:
                    break

                # Find max gain among non-dirty blocks, or recompute if all dirty
                # For efficiency, we track the max and only recompute when needed
                max_gain = -1
                best_bi = -1

                # Check if we need to recompute any gains
                # In lazy greedy, we only recompute when a block is about to be selected
                # For simplicity, we do a full scan but skip blocks whose gain can't improve
                for bi in range(n_k):
                    if gains[bi] > max_gain:
                        max_gain = gains[bi]
                        best_bi = bi

                if max_gain <= 0:
                    break

                # Select the block with max gain
                selected.append(best_bi)
                gains[best_bi] = -1  # Mark as selected

                # Update gains for blocks sharing t-subsets with selected block
                for ts_idx in block_covers_np[best_bi]:
                    if uncovered[ts_idx]:
                        uncovered[ts_idx] = False
                        for bi in t_to_blocks_np[ts_idx]:
                            if gains[bi] >= 0:
                                gains[bi] -= 1

            if len(selected) < best_size:
                best_size = len(selected)
                best_blocks = selected.copy()

        return best_blocks

    # Run lazy greedy with multiple restarts
    best_greedy = lazy_greedy_cover(seed=0, max_restarts=10)
    if best_greedy is None:
        # Fallback to simple greedy
        uncovered = np.ones(n_t, dtype=bool)
        gains = np.array([len(block_covers_np[i]) for i in range(n_k)], dtype=np.int32)
        best_greedy = []
        while np.any(uncovered) and time.time() < deadline - 30.0:
            max_gain = gains.max()
            if max_gain <= 0:
                break
            chosen = int(np.argmax(gains))
            best_greedy.append(chosen)
            gains[chosen] = -1
            for ts_idx in block_covers_np[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bi in t_to_blocks_np[ts_idx]:
                        gains[bi] -= 1

    selected = set(best_greedy)

    # Build cover_count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    # ---------------- Deletion reduction ----------------
    def reduce_blocks(selected_set, cc):
        changed = True
        while changed:
            changed = False
            for bi in list(selected_set):
                if time.time() > deadline - 20.0:
                    return selected_set, cc
                cc[block_covers_np[bi]] -= 1
                if np.any(cc[block_covers_np[bi]] == 0):
                    cc[block_covers_np[bi]] += 1
                else:
                    selected_set.discard(bi)
                    changed = True
        return selected_set, cc

    selected, cover_count = reduce_blocks(selected, cover_count)

    # ---------------- Pair replacement: try to replace 2 blocks with 1 ----------------
    def pair_replace(selected_set, cc, max_trials=800):
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 2:
            return selected_set, cc
        sel_list = list(selected_set)
        for trial in range(max_trials):
            if time.time() > deadline - 30.0:
                break
            i, j = rng.sample(range(n_sel), 2)
            bi, bj = sel_list[i], sel_list[j]
            # Union of t-subsets covered by bi and bj
            union_t = set(block_covers_np[bi].tolist()) | set(block_covers_np[bj].tolist())
            if len(union_t) > 35:  # Can't be covered by one block
                continue
            # Check if any single block covers all t-subsets in union_t
            common = None
            for ti in union_t:
                blocks_covering = set(t_to_blocks[ti])
                if common is None:
                    common = blocks_covering
                else:
                    common &= blocks_covering
                if not common:
                    break
            if not common:
                continue
            # Try each candidate block
            for cand in common:
                if cand == bi or cand == bj:
                    continue
                # Check if replacing bi, bj with cand maintains coverage
                cc[block_covers_np[bi]] -= 1
                cc[block_covers_np[bj]] -= 1
                cc[block_covers_np[cand]] += 1
                if np.all(cc >= 1):
                    # Valid replacement
                    selected_set.discard(bi)
                    selected_set.discard(bj)
                    selected_set.add(cand)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    # Reduce again
                    selected_set, cc = reduce_blocks(selected_set, cc)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    break
                else:
                    # Restore
                    cc[block_covers_np[bi]] += 1
                    cc[block_covers_np[bj]] += 1
                    cc[block_covers_np[cand]] -= 1
        return selected_set, cc

    selected, cover_count = pair_replace(selected, cover_count)

    # ---------------- Lagrangian-guided WalkSAT ----------------
    x = np.array([bi in selected for bi in range(n_k)], dtype=bool)
    n_true = x.sum()

    # Clause satisfaction counts
    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat[block_covers_np[bi]] += 1

    # Penalty weights for Lagrangian relaxation
    penalty = np.ones(n_t, dtype=np.float64)

    def smooth_cost(x_arr, clause_sat_arr, penalty_arr):
        n_true_val = x_arr.sum()
        safe_sat = np.maximum(clause_sat_arr, 1).astype(np.float64)
        soft_deficit = np.where(clause_sat_arr > 0, 1.0 / (safe_sat + 1.0), 1.0)
        pen_val = np.sum(penalty_arr * soft_deficit)
        return n_true_val + 50.0 * pen_val

    best_x = x.copy()
    best_cost = smooth_cost(x, clause_sat, penalty)
    best_n_true = n_true

    # WalkSAT with Lagrangian guidance
    remaining = deadline - time.time() - 20.0
    if remaining > 10.0:
        max_steps = int(min(remaining * 15000, 750000))
        max_steps = max(max_steps, 5000)

        unsat_count = int((clause_sat == 0).sum())

        true_blocks = np.where(x)[0].tolist()
        for step in range(max_steps):
            if time.time() > deadline - 20.0:
                break

            if unsat_count > 0:
                # Pick an unsatisfied clause, weighted by penalty
                unsat_mask = clause_sat == 0
                unsat_indices = np.where(unsat_mask)[0]
                weights = penalty[unsat_indices]
                total_w = weights.sum()
                if total_w == 0:
                    ti = unsat_indices[_RNG.randint(0, len(unsat_indices) - 1)]
                else:
                    probs = weights / total_w
                    ti = unsat_indices[_RNG.choice(len(unsat_indices), p=probs)]

                # Pick a block covering this clause
                cands = t_to_blocks[ti]
                bi = cands[_RNG.randint(0, len(cands) - 1)]

                # Flip
                old = x[bi]
                x[bi] = not old
                delta = 1 if x[bi] else -1
                affected = block_covers_np[bi]
                clause_sat[affected] += delta

                # Update unsat_count
                if delta == 1:
                    newly_sat = (clause_sat[affected] == 1)
                    unsat_count -= int(newly_sat.sum())
                else:
                    newly_unsat = (clause_sat[affected] == 0)
                    unsat_count += int(newly_unsat.sum())

                # Update penalty: increase for still-unsat, decrease for sat
                unsat_mask = clause_sat == 0
                penalty[unsat_mask] = np.minimum(penalty[unsat_mask] * 1.15, 50.0)
                sat_mask = ~unsat_mask
                penalty[sat_mask] = np.maximum(penalty[sat_mask] * 0.97, 0.1)

                c = smooth_cost(x, clause_sat, penalty)
                if c < best_cost:
                    best_cost = c
                    best_x = x.copy()
                    best_n_true = x.sum()
            else:
                # Satisfied: try to remove a block
                if not true_blocks:
                    break
                bi = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                # Try removal
                affected = block_covers_np[bi]
                clause_sat[affected] -= 1
                failed = np.any(clause_sat[affected] == 0)
                if failed:
                    x[bi] = True
                    clause_sat[affected] += 1
                else:
                    x[bi] = False
                    true_blocks.remove(bi)
                    c = smooth_cost(x, clause_sat, penalty)
                    if c < best_cost:
                        best_cost = c
                        best_x = x.copy()
                        best_n_true = x.sum()

                # Also try 2-block swap: remove one true block, add one false block
                if _RNG.random() < 0.3 and true_blocks:
                    bi_remove = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                    false_blocks = np.where(~x)[0]
                    if len(false_blocks) > 0:
                        bi_add = false_blocks[_RNG.randint(0, len(false_blocks) - 1)]
                        aff_rm = block_covers_np[bi_remove]
                        aff_add = block_covers_np[bi_add]
                        clause_sat[aff_rm] -= 1
                        clause_sat[aff_add] += 1
                        if np.all(clause_sat >= 1):
                            x[bi_remove] = False
                            x[bi_add] = True
                            true_blocks.remove(bi_remove)
                            true_blocks.append(bi_add)
                            c = smooth_cost(x, clause_sat, penalty)
                            if c < best_cost:
                                best_cost = c
                                best_x = x.copy()
                                best_n_true = x.sum()
                        else:
                            clause_sat[aff_rm] += 1
                            clause_sat[aff_add] -= 1

        # Use best found
        x = best_x
        clause_sat = np.zeros(n_t, dtype=np.int32)
        for bi in range(n_k):
            if x[bi]:
                clause_sat[block_covers_np[bi]] += 1
        selected = set(int(bi) for bi in range(n_k) if x[bi])
        cover_count = clause_sat

    # ---------------- Final reduction ----------------
    selected, cover_count = reduce_blocks(selected, cover_count)

    # ---------------- MILP polish (if time permits) ----------------
    if time.time() < deadline - 35.0:
        try:
            A_rows, A_cols, A_data = [], [], []
            for ti in range(n_t):
                for bi in t_to_blocks[ti]:
                    A_rows.append(ti)
                    A_cols.append(bi)
                    A_data.append(1)
            A = csr_matrix((A_data, (A_rows, A_cols)), shape=(n_t, n_k))
            c_obj = np.ones(n_k)
            lb = np.ones(n_t)
            ub = np.full(n_t, np.inf)
            integrality = np.ones(n_k)
            bounds = Bounds(lb=np.zeros(n_k), ub=np.ones(n_k))
            constraints = LinearConstraint(A, lb, ub)
            remaining_for_milp = max(15, deadline - 35.0 - time.time())
            res = milp(c_obj, constraints=constraints, integrality=integrality,
                       bounds=bounds, options={"time_limit": remaining_for_milp, "mip_gap": 0.05})
            if res.status == 0 and res.x is not None:
                milp_sel = set(int(bi) for bi in range(n_k) if res.x[bi] > 0.5)
                if len(milp_sel) < len(selected):
                    cc = np.zeros(n_t, dtype=np.int32)
                    for bi in milp_sel:
                        cc[block_covers_np[bi]] += 1
                    if np.all(cc >= 1):
                        selected = milp_sel
                        selected, cover_count = reduce_blocks(selected, cc)
        except Exception:
            pass

    # Final reduction
    if selected:
        cc = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc[block_covers_np[bi]] += 1
        selected, cover_count = reduce_blocks(selected, cc)

    # Second pair replacement pass after final reduction
    if time.time() < deadline - 10.0:
        selected, cover_count = pair_replace(selected, cover_count, max_trials=200)
        if selected:
            cc = np.zeros(n_t, dtype=np.int32)
            for bi in selected:
                cc[block_covers_np[bi]] += 1
            selected, cover_count = reduce_blocks(selected, cc)

    # Build output
    blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    # Final validation
    if blocks:
        cc_final = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc_final[block_covers_np[bi]] += 1
        if np.any(cc_final == 0):
            # Fallback: rebuild from best_greedy
            selected = set(best_greedy)
            blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    return blocks
