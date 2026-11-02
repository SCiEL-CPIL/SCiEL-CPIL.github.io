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
    n_true = int(x.sum())

    # Clause satisfaction counts
    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        clause_sat[block_covers_np[bi]] += 1

    # Penalty weights for Lagrangian relaxation
    penalty = np.ones(n_t, dtype=np.float64)

    # Incremental soft-cost tracking
    # soft_deficit_i = 1.0/(sat_i + 1) if sat_i > 0 else 1.0
    def _sd(s):
        """soft deficit for a single satisfaction count"""
        return 1.0 / (s + 1.0) if s > 0 else 1.0

    # Initial soft cost
    soft_cost_val = float(np.sum(penalty * np.where(clause_sat > 0, 1.0 / (clause_sat.astype(np.float64) + 1.0), 1.0)))
    total_cost = n_true + 50.0 * soft_cost_val

    best_x = x.copy()
    best_cost = total_cost
    best_n_true = n_true

    # WalkSAT with Lagrangian guidance
    remaining = deadline - time.time() - 20.0
    if remaining > 10.0:
        max_steps = int(min(remaining * 20000, 1000000))
        max_steps = max(max_steps, 5000)

        # Maintain unsat count for quick check
        unsat_count = int((clause_sat == 0).sum())

        true_blocks = np.where(x)[0].tolist()
        # Precompute false_blocks index (updated lazily)
        false_blocks_arr = np.where(~x)[0]

        for step in range(max_steps):
            if time.time() > deadline - 20.0:
                break

            if unsat_count > 0:
                # Pick an unsatisfied clause, weighted by penalty
                unsat_indices = np.where(clause_sat == 0)[0]
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
                old_sat = clause_sat[affected].copy()
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

                # Incremental cost update
                n_true += delta
                delta_soft = 0.0
                for idx in range(len(affected)):
                    ti_idx = affected[idx]
                    os = old_sat[idx]
                    ns = clause_sat[ti_idx]
                    old_sd = 1.0 / (os + 1.0) if os > 0 else 1.0
                    new_sd = 1.0 / (ns + 1.0) if ns > 0 else 1.0
                    delta_soft += penalty[ti_idx] * (new_sd - old_sd)
                soft_cost_val += delta_soft
                total_cost = n_true + 50.0 * soft_cost_val

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_x = x.copy()
                    best_n_true = n_true

                # Update true_blocks / false_blocks tracking
                if delta == 1:
                    true_blocks.append(bi)
                    # Remove bi from false_blocks (lazy: rebuild periodically)
                else:
                    try:
                        true_blocks.remove(bi)
                    except ValueError:
                        pass
                    false_blocks_arr = np.where(~x)[0]

            else:
                # Satisfied: try to remove a block
                if not true_blocks:
                    break
                bi = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                # Try removal
                affected = block_covers_np[bi]
                old_sat = clause_sat[affected].copy()
                clause_sat[affected] -= 1
                failed = np.any(clause_sat[affected] == 0)
                if failed:
                    x[bi] = True
                    clause_sat[affected] += 1
                else:
                    x[bi] = False
                    true_blocks.remove(bi)
                    n_true -= 1
                    delta_soft = 0.0
                    for idx in range(len(affected)):
                        ti_idx = affected[idx]
                        os = old_sat[idx]
                        ns = clause_sat[ti_idx]
                        old_sd = 1.0 / (os + 1.0) if os > 0 else 1.0
                        new_sd = 1.0 / (ns + 1.0) if ns > 0 else 1.0
                        delta_soft += penalty[ti_idx] * (new_sd - old_sd)
                    soft_cost_val += delta_soft
                    total_cost = n_true + 50.0 * soft_cost_val
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_x = x.copy()
                        best_n_true = n_true
                    false_blocks_arr = np.where(~x)[0]

                # Also try 2-block swap: remove one true block, add one false block
                if _RNG.random() < 0.3 and true_blocks and len(false_blocks_arr) > 0:
                    bi_remove = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                    bi_add = false_blocks_arr[_RNG.randint(0, len(false_blocks_arr) - 1)]
                    # Try the swap
                    aff_rm = block_covers_np[bi_remove]
                    aff_add = block_covers_np[bi_add]
                    old_sat_rm = clause_sat[aff_rm].copy()
                    old_sat_add = clause_sat[aff_add].copy()
                    clause_sat[aff_rm] -= 1
                    clause_sat[aff_add] += 1
                    if np.all(clause_sat >= 1):
                        # Valid swap (n_true unchanged)
                        x[bi_remove] = False
                        x[bi_add] = True
                        true_blocks.remove(bi_remove)
                        true_blocks.append(bi_add)
                        delta_soft = 0.0
                        for idx in range(len(aff_rm)):
                            ti_idx = aff_rm[idx]
                            os = old_sat_rm[idx]
                            ns = clause_sat[ti_idx]
                            old_sd = 1.0 / (os + 1.0) if os > 0 else 1.0
                            new_sd = 1.0 / (ns + 1.0) if ns > 0 else 1.0
                            delta_soft += penalty[ti_idx] * (new_sd - old_sd)
                        for idx in range(len(aff_add)):
                            ti_idx = aff_add[idx]
                            os = old_sat_add[idx]
                            ns = clause_sat[ti_idx]
                            old_sd = 1.0 / (os + 1.0) if os > 0 else 1.0
                            new_sd = 1.0 / (ns + 1.0) if ns > 0 else 1.0
                            delta_soft += penalty[ti_idx] * (new_sd - old_sd)
                        soft_cost_val += delta_soft
                        total_cost = n_true + 50.0 * soft_cost_val
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_x = x.copy()
                            best_n_true = n_true
                        false_blocks_arr = np.where(~x)[0]
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

    # ---------------- Additional local search polish (replaces MILP) ----------------
    if time.time() < deadline - 15.0:
        # Run another round of WalkSAT from the current state
        x2 = np.array([bi in selected for bi in range(n_k)], dtype=bool)
        n_true2 = int(x2.sum())
        clause_sat2 = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            clause_sat2[block_covers_np[bi]] += 1
        penalty2 = np.ones(n_t, dtype=np.float64)
        soft_cost_val2 = float(np.sum(penalty2 * np.where(clause_sat2 > 0, 1.0 / (clause_sat2.astype(np.float64) + 1.0), 1.0)))
        total_cost2 = n_true2 + 50.0 * soft_cost_val2
        best_x2 = x2.copy()
        best_cost2 = total_cost2
        best_n_true2 = n_true2
        unsat_count2 = int((clause_sat2 == 0).sum())
        true_blocks2 = np.where(x2)[0].tolist()
        false_blocks_arr2 = np.where(~x2)[0]

        remaining2 = deadline - time.time() - 10.0
        if remaining2 > 5.0:
            max_steps2 = int(min(remaining2 * 20000, 500000))
            max_steps2 = max(max_steps2, 2000)
            for step2 in range(max_steps2):
                if time.time() > deadline - 10.0:
                    break
                if unsat_count2 > 0:
                    unsat_indices2 = np.where(clause_sat2 == 0)[0]
                    weights2 = penalty2[unsat_indices2]
                    total_w2 = weights2.sum()
                    if total_w2 == 0:
                        ti2 = unsat_indices2[_RNG.randint(0, len(unsat_indices2) - 1)]
                    else:
                        probs2 = weights2 / total_w2
                        ti2 = unsat_indices2[_RNG.choice(len(unsat_indices2), p=probs2)]
                    cands2 = t_to_blocks[ti2]
                    bi2 = cands2[_RNG.randint(0, len(cands2) - 1)]
                    old2 = x2[bi2]
                    x2[bi2] = not old2
                    delta2 = 1 if x2[bi2] else -1
                    affected2 = block_covers_np[bi2]
                    old_sat2 = clause_sat2[affected2].copy()
                    clause_sat2[affected2] += delta2
                    if delta2 == 1:
                        newly_sat2 = (clause_sat2[affected2] == 1)
                        unsat_count2 -= int(newly_sat2.sum())
                    else:
                        newly_unsat2 = (clause_sat2[affected2] == 0)
                        unsat_count2 += int(newly_unsat2.sum())
                    unsat_mask2 = clause_sat2 == 0
                    penalty2[unsat_mask2] = np.minimum(penalty2[unsat_mask2] * 1.15, 50.0)
                    sat_mask2 = ~unsat_mask2
                    penalty2[sat_mask2] = np.maximum(penalty2[sat_mask2] * 0.97, 0.1)
                    n_true2 += delta2
                    delta_soft2 = 0.0
                    for idx2 in range(len(affected2)):
                        ti_idx2 = affected2[idx2]
                        os2 = old_sat2[idx2]
                        ns2 = clause_sat2[ti_idx2]
                        old_sd2 = 1.0 / (os2 + 1.0) if os2 > 0 else 1.0
                        new_sd2 = 1.0 / (ns2 + 1.0) if ns2 > 0 else 1.0
                        delta_soft2 += penalty2[ti_idx2] * (new_sd2 - old_sd2)
                    soft_cost_val2 += delta_soft2
                    total_cost2 = n_true2 + 50.0 * soft_cost_val2
                    if total_cost2 < best_cost2:
                        best_cost2 = total_cost2
                        best_x2 = x2.copy()
                        best_n_true2 = n_true2
                    if delta2 == 1:
                        true_blocks2.append(bi2)
                    else:
                        try:
                            true_blocks2.remove(bi2)
                        except ValueError:
                            pass
                    false_blocks_arr2 = np.where(~x2)[0]
                else:
                    if not true_blocks2:
                        break
                    bi2 = true_blocks2[_RNG.randint(0, len(true_blocks2) - 1)]
                    affected2 = block_covers_np[bi2]
                    old_sat2 = clause_sat2[affected2].copy()
                    clause_sat2[affected2] -= 1
                    failed2 = np.any(clause_sat2[affected2] == 0)
                    if failed2:
                        x2[bi2] = True
                        clause_sat2[affected2] += 1
                    else:
                        x2[bi2] = False
                        true_blocks2.remove(bi2)
                        n_true2 -= 1
                        delta_soft2 = 0.0
                        for idx2 in range(len(affected2)):
                            ti_idx2 = affected2[idx2]
                            os2 = old_sat2[idx2]
                            ns2 = clause_sat2[ti_idx2]
                            old_sd2 = 1.0 / (os2 + 1.0) if os2 > 0 else 1.0
                            new_sd2 = 1.0 / (ns2 + 1.0) if ns2 > 0 else 1.0
                            delta_soft2 += penalty2[ti_idx2] * (new_sd2 - old_sd2)
                        soft_cost_val2 += delta_soft2
                        total_cost2 = n_true2 + 50.0 * soft_cost_val2
                        if total_cost2 < best_cost2:
                            best_cost2 = total_cost2
                            best_x2 = x2.copy()
                            best_n_true2 = n_true2
                        false_blocks_arr2 = np.where(~x2)[0]
                    if _RNG.random() < 0.3 and true_blocks2 and len(false_blocks_arr2) > 0:
                        bi_rm2 = true_blocks2[_RNG.randint(0, len(true_blocks2) - 1)]
                        bi_ad2 = false_blocks_arr2[_RNG.randint(0, len(false_blocks_arr2) - 1)]
                        aff_rm2 = block_covers_np[bi_rm2]
                        aff_ad2 = block_covers_np[bi_ad2]
                        old_sat_rm2 = clause_sat2[aff_rm2].copy()
                        old_sat_ad2 = clause_sat2[aff_ad2].copy()
                        clause_sat2[aff_rm2] -= 1
                        clause_sat2[aff_ad2] += 1
                        if np.all(clause_sat2 >= 1):
                            x2[bi_rm2] = False
                            x2[bi_ad2] = True
                            true_blocks2.remove(bi_rm2)
                            true_blocks2.append(bi_ad2)
                            delta_soft2 = 0.0
                            for idx2 in range(len(aff_rm2)):
                                ti_idx2 = aff_rm2[idx2]
                                os2 = old_sat_rm2[idx2]
                                ns2 = clause_sat2[ti_idx2]
                                old_sd2 = 1.0 / (os2 + 1.0) if os2 > 0 else 1.0
                                new_sd2 = 1.0 / (ns2 + 1.0) if ns2 > 0 else 1.0
                                delta_soft2 += penalty2[ti_idx2] * (new_sd2 - old_sd2)
                            for idx2 in range(len(aff_ad2)):
                                ti_idx2 = aff_ad2[idx2]
                                os2 = old_sat_ad2[idx2]
                                ns2 = clause_sat2[ti_idx2]
                                old_sd2 = 1.0 / (os2 + 1.0) if os2 > 0 else 1.0
                                new_sd2 = 1.0 / (ns2 + 1.0) if ns2 > 0 else 1.0
                                delta_soft2 += penalty2[ti_idx2] * (new_sd2 - old_sd2)
                            soft_cost_val2 += delta_soft2
                            total_cost2 = n_true2 + 50.0 * soft_cost_val2
                            if total_cost2 < best_cost2:
                                best_cost2 = total_cost2
                                best_x2 = x2.copy()
                                best_n_true2 = n_true2
                            false_blocks_arr2 = np.where(~x2)[0]
                        else:
                            clause_sat2[aff_rm2] += 1
                            clause_sat2[aff_ad2] -= 1

        # Use best from second round
        x2 = best_x2
        sel2 = set(int(bi) for bi in range(n_k) if x2[bi])
        if len(sel2) < len(selected):
            cc2 = np.zeros(n_t, dtype=np.int32)
            for bi in sel2:
                cc2[block_covers_np[bi]] += 1
            if np.all(cc2 >= 1):
                selected = sel2
                cover_count = cc2
                selected, cover_count = reduce_blocks(selected, cover_count)

    # Final reduction
    if selected:
        cc = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc[block_covers_np[bi]] += 1
        selected, cover_count = reduce_blocks(selected, cc)

    # Fast numpy-based pair replacement: for each pair, check if union of covered
    # t-subsets can be covered by a single block (using vectorized intersection)
    if time.time() < deadline - 10.0 and len(selected) >= 2:
        sel_list = list(selected)
        n_sel = len(sel_list)
        # Precompute sorted t-subset arrays for each selected block
        sorted_covers = [np.sort(block_covers_np[bi]) for bi in sel_list]
        # For each pair, compute union size and check feasibility
        improved = True
        max_rounds = 5
        for rnd in range(max_rounds):
            if not improved or time.time() > deadline - 5.0:
                break
            improved = False
            n_sel = len(sel_list)
            if n_sel < 2:
                break
            for i in range(n_sel):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, n_sel):
                    if time.time() > deadline - 5.0:
                        break
                    # Union of t-subsets
                    union_t = np.union1d(sorted_covers[i], sorted_covers[j])
                    if len(union_t) > 35:  # max t-subsets per block = C(7,4)=35
                        continue
                    # Find blocks that cover all t-subsets in union_t
                    # Use the first t-subset to get candidates, then filter
                    first_ti = union_t[0]
                    candidates = t_to_blocks[first_ti]
                    if not candidates:
                        continue
                    # Filter: candidate must cover all t-subsets in union_t
                    found = False
                    for cand in candidates:
                        if cand == sel_list[i] or cand == sel_list[j]:
                            continue
                        cand_cov = block_covers_np[cand]
                        # Check if union_t is subset of cand_cov
                        if len(union_t) <= len(cand_cov):
                            # Use set intersection: union_t should be contained in cand_cov
                            if len(np.intersect1d(union_t, cand_cov, assume_unique=False)) == len(union_t):
                                # Verify coverage
                                cc_test = cover_count.copy()
                                cc_test[sorted_covers[i]] -= 1
                                cc_test[sorted_covers[j]] -= 1
                                cc_test[cand_cov] += 1
                                if np.all(cc_test >= 1):
                                    # Apply replacement
                                    cover_count = cc_test
                                    selected.discard(sel_list[i])
                                    selected.discard(sel_list[j])
                                    selected.add(cand)
                                    # Rebuild lists
                                    sel_list = list(selected)
                                    sorted_covers = [np.sort(block_covers_np[bi]) for bi in sel_list]
                                    n_sel = len(sel_list)
                                    improved = True
                                    # Run reduction
                                    selected, cover_count = reduce_blocks(selected, cover_count)
                                    sel_list = list(selected)
                                    sorted_covers = [np.sort(block_covers_np[bi]) for bi in sel_list]
                                    n_sel = len(sel_list)
                                    found = True
                                    break
                    if found:
                        continue  # restart inner loop with new lists
                    # If no improvement from this pair, continue
        # Final reduction after pair replacement
        if selected:
            cc = np.zeros(n_t, dtype=np.int32)
            for bi in selected:
                cc[block_covers_np[bi]] += 1
            selected, cover_count = reduce_blocks(selected, cc)
    else:
        # Second pair replacement pass after final reduction (original)
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
