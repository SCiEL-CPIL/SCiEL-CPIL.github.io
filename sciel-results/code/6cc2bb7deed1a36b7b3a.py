"""Combined: Lagrangian-guided WalkSAT + vectorized greedy + pair-replacement refinement.
Uses dynamic penalty weights on uncovered t-subsets to guide stochastic local search,
with pair-replacement moves for structural improvements beyond single-block flips."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import defaultdict
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

    # Precompute intersection sizes between all pairs of blocks (for pair_replace speedup)
    # This is expensive for large n_k, so only do it if n_k is manageable
    if n_k <= 50000:
        pair_intersection_cache = {}
    else:
        pair_intersection_cache = None

    # ---------------- Vectorized Greedy with random tie-breaking ----------------
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gain = np.zeros(n_k, dtype=np.int32)
        for bi in range(n_k):
            gain[bi] = np.sum(uncovered[block_covers_np[bi]])
        gain = gain.copy()
        blocks_used = []
        while np.any(uncovered):
            if time.time() > deadline - 30.0:
                break
            max_gain = gain.max()
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            chosen = candidates[rng.randint(0, len(candidates) - 1)]
            blocks_used.append(chosen)
            for ts_idx in block_covers_np[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bi in t_to_blocks_np[ts_idx]:
                        gain[bi] -= 1
            gain[chosen] = -1
        return blocks_used

    # Run multiple greedy restarts, keep best
    best_greedy = None
    best_greedy_size = n_k + 1
    for seed in range(12):
        if time.time() > deadline - 50.0:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_greedy_size:
            best_greedy_size = len(blocks)
            best_greedy = blocks
        if len(blocks) <= 135:  # Good enough
            break

    if best_greedy is None:
        best_greedy = greedy_cover(0)

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
            # For each t-subset in union_t, find common blocks
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
                # Temporarily remove bi, bj
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
    # State: boolean array
    x = np.array([bi in selected for bi in range(n_k)], dtype=bool)
    n_true = x.sum()

    # Clause satisfaction counts
    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat[block_covers_np[bi]] += 1

    # Penalty weights for Lagrangian relaxation
    penalty = np.ones(n_t, dtype=np.float64)

    # Soft-min surrogate: smooth penalty for coverage deficit
    # Using log-sum-exp style: soft_deficit = logsumexp(-log(clause_sat)) for clause_sat>0, else large
    def smooth_cost(x_arr, clause_sat_arr, penalty_arr):
        n_true_val = x_arr.sum()
        # Smooth deficit: for each clause, if sat=0 deficit=1, else deficit = 1/sat (soft)
        # Use: deficit_i = 1.0 / max(clause_sat_i, 1) if sat_i >= 1 else 1.0
        # Actually simpler: deficit = max(0, 1 - sat) but smooth via:
        # soft_deficit = np.where(clause_sat_arr > 0, 1.0/(clause_sat_arr+1), 1.0)
        # This gives 0.5 for sat=1, 0.33 for sat=2, 1.0 for sat=0
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

        # Maintain unsat count for quick check
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
                    # Block added: clauses going from 0 to 1
                    newly_sat = (clause_sat[affected] == 1)
                    unsat_count -= int(newly_sat.sum())
                else:
                    # Block removed: clauses going from 1 to 0
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
                    # Find a candidate block to add
                    # Pick a random false block
                    false_blocks = np.where(~x)[0]
                    if len(false_blocks) > 0:
                        bi_add = false_blocks[_RNG.randint(0, len(false_blocks) - 1)]
                        # Try the swap
                        aff_rm = block_covers_np[bi_remove]
                        aff_add = block_covers_np[bi_add]
                        clause_sat[aff_rm] -= 1
                        clause_sat[aff_add] += 1
                        if np.all(clause_sat >= 1):
                            # Valid swap
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
                            # Restore
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
                    # Verify
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
