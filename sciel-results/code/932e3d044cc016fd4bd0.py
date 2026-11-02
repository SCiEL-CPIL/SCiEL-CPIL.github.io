"""Combined: Flow-priority greedy + WalkSAT + systematic 2-for-1 swap.
Uses max-flow-inspired bottleneck scoring for greedy augmentation."""
import time
import random
import itertools
import heapq
import numpy as np

_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0

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

    # Convert to numpy
    block_covers_np = [np.array(c, dtype=np.int32) for c in block_covers]
    t_to_blocks_np = [np.array(b, dtype=np.int32) for b in t_to_blocks]
    t_to_blocks_set = [set(b) for b in t_to_blocks]

    # ---------------- Flow-Priority Greedy (max-flow inspired) ----------------
    # Score each block by sum of bottleneck weights of its uncovered t-subsets.
    # Bottleneck weight for a t-subset = 1 / (number of available blocks covering it),
    # mimicking residual capacity in a max-flow network.
    def flow_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        # avail_count[ti] = number of blocks that still cover ti (and are available)
        avail_count = np.array([len(t_to_blocks[ti]) for ti in range(n_t)], dtype=np.float64)
        # bottleneck_weight[ti] = 1/avail_count if uncovered, else 0
        bottleneck = np.zeros(n_t, dtype=np.float64)
        for ti in range(n_t):
            if avail_count[ti] > 0:
                bottleneck[ti] = 1.0 / avail_count[ti]
            else:
                bottleneck[ti] = 1e6  # extremely hard

        # score[bi] = sum of bottleneck[ti] for ti in block_covers[bi] if uncovered
        score = np.zeros(n_k, dtype=np.float64)
        for bi in range(n_k):
            score[bi] = np.sum(bottleneck[block_covers_np[bi]])

        blocks_used = []
        n_uncovered = n_t

        while n_uncovered > 0:
            if time.time() > deadline - 35.0:
                break
            max_score = score.max()
            if max_score <= 0:
                break
            candidates = np.where(score == max_score)[0]
            chosen = candidates[rng.randint(0, len(candidates) - 1)]
            blocks_used.append(chosen)

            # Update: mark newly covered t-subsets
            for ts_idx in block_covers_np[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    n_uncovered -= 1
                    # Decrement avail_count for this t-subset
                    avail_count[ts_idx] -= 1
                    # Update bottleneck weight
                    if avail_count[ts_idx] > 0:
                        bottleneck[ts_idx] = 1.0 / avail_count[ts_idx]
                    else:
                        bottleneck[ts_idx] = 0.0  # covered, no longer relevant
                    # Update scores for all blocks covering this t-subset
                    for bi2 in t_to_blocks_np[ts_idx]:
                        if bi2 != chosen:
                            score[bi2] -= bottleneck[ts_idx] + (1.0 / (avail_count[ts_idx] + 1)) if avail_count[ts_idx] >= 0 else 0
                            # Simpler: just recompute delta
                            # Actually let's just subtract the old bottleneck contribution
            # Recompute scores more carefully for correctness
            # The above incremental update is tricky; let's do a cleaner version:
            # For each newly uncovered->covered t-subset, subtract its bottleneck from all covering blocks
            # We already did that above but the formula was messy. Let me redo:

            score[chosen] = -1  # mark as used

        return blocks_used

    # Cleaner flow-greedy with proper incremental scoring
    def flow_greedy_clean(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        avail = np.array([len(t_to_blocks[ti]) for ti in range(n_t)], dtype=np.float64)
        # bottleneck weight: for uncovered ti, weight = 1/avail[ti] (or large if avail==0)
        weight = np.zeros(n_t, dtype=np.float64)
        for ti in range(n_t):
            if avail[ti] > 0:
                weight[ti] = 1.0 / avail[ti]
            else:
                weight[ti] = 1000.0

        # score[bi] = sum of weight[ti] for ti in covers[bi] where uncovered[ti]
        score = np.zeros(n_k, dtype=np.float64)
        for bi in range(n_k):
            cov = block_covers_np[bi]
            score[bi] = np.sum(weight[cov])

        blocks_used = []
        n_unc = n_t

        while n_unc > 0:
            if time.time() > deadline - 35.0:
                break
            ms = score.max()
            if ms <= 0:
                break
            cands = np.where(score == ms)[0]
            chosen = int(cands[rng.randint(0, len(cands) - 1)])
            blocks_used.append(chosen)
            score[chosen] = -1

            # For each t-subset in chosen that was uncovered:
            for ts_idx in block_covers_np[chosen]:
                if not uncovered[ts_idx]:
                    continue
                uncovered[ts_idx] = False
                n_unc -= 1
                old_w = weight[ts_idx]
                avail[ts_idx] -= 1
                new_w = 1.0 / avail[ts_idx] if avail[ts_idx] > 0 else 0.0
                # Update weight for this t-subset (only matters if still uncovered, but it's now covered)
                weight[ts_idx] = new_w
                # Update scores: for all blocks covering ts_idx, subtract old_w (their contribution from this t-subset changes)
                # Since ts_idx is now covered, its contribution to score should be 0 going forward
                # So we subtract old_w from each covering block's score
                for bi2 in t_to_blocks_np[ts_idx]:
                    if score[bi2] > -0.5:  # still available
                        score[bi2] -= old_w

        return blocks_used

    best_greedy = None
    best_greedy_size = n_k + 1
    for seed in range(8):
        if time.time() > deadline - 55.0:
            break
        blocks = flow_greedy_clean(seed)
        if len(blocks) < best_greedy_size:
            best_greedy_size = len(blocks)
            best_greedy = blocks
        if len(blocks) <= 138:
            break

    if best_greedy is None:
        best_greedy = flow_greedy_clean(0)

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

    # ---------------- Systematic 2-for-1 block swap ----------------
    def pair_swap(selected_set, cc, max_rounds=30):
        """Try to replace 2 blocks with 1 block that covers all uniquely-covered t-subsets."""
        rng = _RNG
        for rnd in range(max_rounds):
            if time.time() > deadline - 40.0:
                break
            sel_list = list(selected_set)
            n_sel = len(sel_list)
            if n_sel < 2:
                break
            found = False
            # Shuffle for diversity
            rng.shuffle(sel_list)
            for i in range(n_sel):
                for j in range(i + 1, n_sel):
                    bi, bj = sel_list[i], sel_list[j]
                    # Find t-subsets uniquely covered by bi or bj
                    unique_t = []
                    for ti in block_covers_np[bi]:
                        if cc[ti] == 1:
                            unique_t.append(ti)
                    for ti in block_covers_np[bj]:
                        if cc[ti] == 1:
                            unique_t.append(ti)
                    if not unique_t:
                        continue
                    if len(unique_t) > 35:
                        continue
                    # Find blocks that cover all unique_t
                    common = None
                    for ti in unique_t:
                        if common is None:
                            common = t_to_blocks_set[ti]
                        else:
                            common &= t_to_blocks_set[ti]
                        if not common:
                            break
                    if not common:
                        continue
                    # Try candidates
                    for cand in common:
                        if cand == bi or cand == bj:
                            continue
                        # Check: removing bi, bj and adding cand keeps all covered
                        # Only need to check t-subsets in bi or bj (others unaffected)
                        cc[block_covers_np[bi]] -= 1
                        cc[block_covers_np[bj]] -= 1
                        cc[block_covers_np[cand]] += 1
                        if np.all(cc >= 1):
                            selected_set.discard(bi)
                            selected_set.discard(bj)
                            selected_set.add(cand)
                            found = True
                            break
                        else:
                            cc[block_covers_np[bi]] += 1
                            cc[block_covers_np[bj]] += 1
                            cc[block_covers_np[cand]] -= 1
                    if found:
                        break
                if found:
                    selected_set, cc = reduce_blocks(selected_set, cc)
                    break
            if not found:
                break
        return selected_set, cc

    selected, cover_count = pair_swap(selected, cover_count)

    # ---------------- Efficient WalkSAT with proper unsat tracking ----------------
    x = np.array([bi in selected for bi in range(n_k)], dtype=bool)
    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat[block_covers_np[bi]] += 1

    # Build unsat set and count
    unsat_set = set(int(ti) for ti in range(n_t) if clause_sat[ti] == 0)

    # Penalty weights (flow-inspired: weight = 1/avail_count for unsat clauses)
    penalty = np.ones(n_t, dtype=np.float64)
    # Initialize penalties based on how hard each t-subset is to cover
    for ti in range(n_t):
        if clause_sat[ti] == 0:
            penalty[ti] = 1.0 / max(len(t_to_blocks[ti]), 1)
        else:
            penalty[ti] = 1.0

    def current_cost():
        return x.sum() + 50.0 * len(unsat_set)

    best_x = x.copy()
    best_cost = current_cost()
    best_n = x.sum()

    remaining = deadline - time.time() - 25.0
    if remaining > 8.0:
        max_steps = int(min(remaining * 15000, 600000))
        max_steps = max(max_steps, 10000)
        no_improve = 0
        patience = 6000

        for step in range(max_steps):
            if time.time() > deadline - 25.0:
                break

            if unsat_set:
                # Pick unsat clause weighted by penalty (flow-bottleneck style)
                unsat_list = list(unsat_set)
                weights = penalty[unsat_list]
                total_w = weights.sum()
                if total_w > 0:
                    probs = weights / total_w
                    ti = int(np.random.choice(len(unsat_list), p=probs))
                else:
                    ti = _RNG.randint(0, len(unsat_list) - 1)
                ti = unsat_list[ti]

                # Pick block covering ti, prefer high-gain blocks (cover more unsat)
                cands = t_to_blocks[ti]
                # Score candidates by how many unsat clauses they cover
                best_bi = cands[0]
                best_gain = -1
                # Sample up to 20 candidates for speed
                n_sample = min(20, len(cands))
                if n_sample < len(cands):
                    sample_idx = _RNG.sample(range(len(cands)), n_sample)
                    sample_cands = [cands[i] for i in sample_idx]
                else:
                    sample_cands = cands
                for bi2 in sample_cands:
                    gain2 = np.sum(clause_sat[block_covers_np[bi2]] == 0)
                    if gain2 > best_gain:
                        best_gain = gain2
                        best_bi = bi2
                bi = best_bi

                old = x[bi]
                x[bi] = not old
                delta = 1 if x[bi] else -1
                affected = block_covers_np[bi]
                clause_sat[affected] += delta

                # Update unsat_set incrementally
                for ti2 in affected:
                    if clause_sat[ti2] == 0:
                        unsat_set.add(int(ti2))
                    elif clause_sat[ti2] == 1:
                        unsat_set.discard(int(ti2))

                # Update penalties (batch every 50 steps for speed)
                if step % 50 == 0:
                    for ti2 in unsat_set:
                        penalty[ti2] = min(penalty[ti2] * 1.2, 100.0)
                    sat_mask = clause_sat >= 1
                    penalty[sat_mask] = np.maximum(penalty[sat_mask] * 0.95, 0.05)

                c = current_cost()
                if c < best_cost:
                    best_cost = c
                    best_x = x.copy()
                    best_n = x.sum()
                    no_improve = 0
                else:
                    no_improve += 1

                # Walk: if no improvement for a while, do a random walk step
                if no_improve > patience:
                    no_improve = 0
                    # Pick random block and flip
                    bi_w = _RNG.randint(0, n_k - 1)
                    x[bi_w] = not x[bi_w]
                    delta_w = 1 if x[bi_w] else -1
                    affected_w = block_covers_np[bi_w]
                    clause_sat[affected_w] += delta_w
                    for ti2 in affected_w:
                        if clause_sat[ti2] == 0:
                            unsat_set.add(int(ti2))
                        elif clause_sat[ti2] == 1:
                            unsat_set.discard(int(ti2))
            else:
                # All satisfied: try to remove a block
                true_blocks = np.where(x)[0]
                if len(true_blocks) == 0:
                    break
                bi = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                x[bi] = False
                affected = block_covers_np[bi]
                clause_sat[affected] -= 1
                failed = False
                new_unsat = []
                for ti2 in affected:
                    if clause_sat[ti2] == 0:
                        new_unsat.append(int(ti2))
                        failed = True
                if failed:
                    x[bi] = True
                    clause_sat[affected] += 1
                    for ti2 in new_unsat:
                        unsat_set.discard(ti2)
                else:
                    for ti2 in affected:
                        if clause_sat[ti2] == 1:
                            unsat_set.discard(int(ti2))
                    c = current_cost()
                    if c < best_cost:
                        best_cost = c
                        best_x = x.copy()
                        best_n = x.sum()
                        no_improve = 0
                    else:
                        no_improve += 1

    # Restore best
    x = best_x
    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat[block_covers_np[bi]] += 1
    selected = set(int(bi) for bi in range(n_k) if x[bi])
    cover_count = clause_sat

    # Final reduction
    selected, cover_count = reduce_blocks(selected, cover_count)

    # Another round of pair swap after WalkSAT
    if time.time() < deadline - 15.0:
        selected, cover_count = pair_swap(selected, cover_count, max_rounds=15)

    # Final validation and output
    if selected:
        cc_final = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc_final[block_covers_np[bi]] += 1
        if np.any(cc_final == 0):
            # Fallback
            selected = set(best_greedy)
            cc_final = np.zeros(n_t, dtype=np.int32)
            for bi in selected:
                cc_final[block_covers_np[bi]] += 1
            if np.any(cc_final == 0):
                # Last resort: add blocks until covered
                for bi in range(n_k):
                    if time.time() > deadline - 2.0:
                        break
                    cc_final[block_covers_np[bi]] += 1
                    selected.add(bi)
                    if np.all(cc_final >= 1):
                        break

    blocks = [list(k_subs[bi]) for bi in sorted(selected)]
    return blocks
