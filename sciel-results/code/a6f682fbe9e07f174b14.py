"""Combined: Lagrangian-guided WalkSAT + vectorized greedy + pair-replacement refinement.
Uses dynamic penalty weights on uncovered t-subsets to guide stochastic local search,
with pair-replacement moves for structural improvements beyond single-block flips.
Adds a bitset-based branch-and-bound polish for final refinement."""
import time
import random
import itertools
import numpy as np
from collections import defaultdict

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

    # ---------------- Bitset-based Branch-and-Bound polish ----------------
    # For each block, precompute the set of t-subsets it covers as a bitset
    # Use Python integers as bitsets for fast AND/OR operations
    n_t_bits = n_t  # number of bits needed
    block_bitsets = []
    for bi in range(n_k):
        bs = 0
        for ti in block_covers_np[bi]:
            bs |= (1 << ti)
        block_bitsets.append(bs)

    # For each t-subset, precompute the set of blocks that cover it
    # t_to_blocks already exists, but we need bitsets of blocks
    t_to_blocks_bitset = []
    for ti in range(n_t):
        bs = 0
        for bi in t_to_blocks[ti]:
            bs |= (1 << bi)
        t_to_blocks_bitset.append(bs)

    def branch_and_bound_polish(selected_set, max_blocks, time_limit):
        """Try to reduce the number of blocks below max_blocks using B&B with bitset lower bounds."""
        start_bb = time.time()
        sel_list = list(selected_set)
        n_sel = len(sel_list)
        
        # Current coverage as a bitset
        current_cover = 0
        for bi in sel_list:
            current_cover |= block_bitsets[bi]
        
        # Full coverage bitset
        full_cover = (1 << n_t) - 1
        
        if current_cover == full_cover and n_sel < max_blocks:
            return selected_set
        
        # Greedy ordering: sort blocks by number of unique t-subsets they cover
        # (that aren't covered by other selected blocks)
        # For B&B, we work with the complement: which t-subsets are NOT covered
        uncovered = full_cover & ~current_cover
        if uncovered == 0:
            # Already fully covered, just try to remove blocks
            for bi in sel_list[:]:
                if time.time() - start_bb > time_limit:
                    break
                new_cover = 0
                for bj in sel_list:
                    if bj != bi:
                        new_cover |= block_bitsets[bj]
                if new_cover == full_cover:
                    sel_list.remove(bi)
                    current_cover = new_cover
                    n_sel -= 1
        
        # B&B: try to find a smaller set
        # Lower bound: for uncovered t-subsets, we need at least ceil(|uncovered| / max_cover_per_block) blocks
        # Better lower bound: use maximal independent set of uncovered t-subsets
        # Two t-subsets are "independent" if no single block covers both
        
        # Precompute: for each pair of t-subsets, is there a block covering both?
        # This is expensive, so use a heuristic lower bound instead
        # LB = number of uncovered t-subsets / max number of uncovered t-subsets any single block can cover
        
        # For each block, how many uncovered t-subsets does it cover?
        # We'll compute this on the fly
        
        best_solution = list(sel_list)
        best_size = n_sel
        
        # Recursive B&B with time limit
        def bb_recursive(remaining_uncovered, chosen, depth, time_limit_local):
            nonlocal best_solution, best_size
            
            if time.time() - start_bb > time_limit_local:
                return
            
            if remaining_uncovered == 0:
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best_solution = list(chosen)
                return
            
            # Lower bound: each block can cover at most max_cover uncovered subsets
            # Compute max cover
            max_cover = 0
            for bi in range(n_k):
                cover_count = bin(remaining_uncovered & block_bitsets[bi]).count('1')
                if cover_count > max_cover:
                    max_cover = cover_count
            
            if max_cover == 0:
                return  # Can't cover any more
            
            lb = (bin(remaining_uncovered).count('1') + max_cover - 1) // max_cover
            if len(chosen) + lb >= best_size:
                return  # Prune
            
            # Upper bound: greedy completion
            greedy_uncovered = remaining_uncovered
            greedy_chosen = list(chosen)
            while greedy_uncovered != 0 and time.time() - start_bb < time_limit_local:
                best_block = -1
                best_gain = 0
                for bi in range(n_k):
                    gain = bin(greedy_uncovered & block_bitsets[bi]).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_block = bi
                if best_block == -1 or best_gain == 0:
                    break
                greedy_chosen.append(best_block)
                greedy_uncovered &= ~block_bitsets[best_block]
            
            if len(greedy_chosen) < best_size:
                best_size = len(greedy_chosen)
                best_solution = list(greedy_chosen)
            
            # Branch: pick a t-subset from remaining_uncovered
            # Find the first set bit
            ts_idx = 0
            temp = remaining_uncovered
            while temp:
                if temp & 1:
                    break
                ts_idx += 1
                temp >>= 1
            
            # Get all blocks that cover this t-subset
            candidates = []
            for bi in t_to_blocks[ts_idx]:
                if bi not in chosen:
                    gain = bin(remaining_uncovered & block_bitsets[bi]).count('1')
                    if gain > 0:
                        candidates.append((gain, bi))
            
            # Sort by gain descending (best first)
            candidates.sort(key=lambda x: -x[0])
            
            # Limit branching factor for performance
            max_branch = 15
            for gain, bi in candidates[:max_branch]:
                new_uncovered = remaining_uncovered & ~block_bitsets[bi]
                chosen.append(bi)
                bb_recursive(new_uncovered, chosen, depth + 1, time_limit_local)
                chosen.pop()
                if time.time() - start_bb > time_limit_local:
                    break
        
        # Run B&B
        bb_recursive(uncovered, [], 0, time_limit)
        
        if best_size < n_sel:
            new_selected = set(best_solution)
            # Verify
            cc = np.zeros(n_t, dtype=np.int32)
            for bi in new_selected:
                cc[block_covers_np[bi]] += 1
            if np.all(cc >= 1):
                return new_selected
        
        return selected_set

    if time.time() < deadline - 40.0:
        remaining_for_bb = min(30.0, deadline - 40.0 - time.time())
        if remaining_for_bb > 5.0:
            try:
                new_selected = branch_and_bound_polish(selected, len(selected) + 1, remaining_for_bb)
                if len(new_selected) < len(selected):
                    selected = new_selected
                    cover_count = np.zeros(n_t, dtype=np.int32)
                    for bi in selected:
                        cover_count[block_covers_np[bi]] += 1
                    selected, cover_count = reduce_blocks(selected, cover_count)
            except Exception:
                pass

    # Final reduction
    if selected:
        cc = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc[block_covers_np[bi]] += 1
        selected, cover_count = reduce_blocks(selected, cc)

    # Second pair replacement pass after final reduction
    if time.time() < deadline - 15.0:
        selected, cover_count = pair_replace(selected, cover_count, max_trials=300)
        if selected:
            cc = np.zeros(n_t, dtype=np.int32)
            for bi in selected:
                cc[block_covers_np[bi]] += 1
            selected, cover_count = reduce_blocks(selected, cc)

    # Final aggressive single-block removal pass
    if time.time() < deadline - 5.0:
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for bi in list(selected):
                cc[block_covers_np[bi]] -= 1
                if np.any(cc[block_covers_np[bi]] == 0):
                    cc[block_covers_np[bi]] += 1
                else:
                    selected.discard(bi)
                    changed = True
        cover_count = cc

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
            # Re-validate
            cc_re = np.zeros(n_t, dtype=np.int32)
            for bi in selected:
                cc_re[block_covers_np[bi]] += 1
            if np.any(cc_re == 0):
                # Last resort: use all k-subsets that cover any uncovered t-subset
                uncovered_final = np.where(cc_re == 0)[0]
                for ti in uncovered_final:
                    for bi in t_to_blocks[ti]:
                        selected.add(bi)
                blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    return blocks
