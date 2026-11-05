import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0

    def time_left():
        return deadline - time.time()

    # Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = [tset_index[ts] for ts in itertools.combinations(b, t)]
        block_tsets.append(ts_indices)

    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)

    # Convert to numpy for speed
    block_tsets_np = [np.array(ts, dtype=np.int32) for ts in block_tsets]
    # Also store as lists for set operations where needed
    block_tsets_set = [set(ts) for ts in block_tsets]

    # 1. Criticality-prioritized greedy construction
    def critical_greedy(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        uncovered = np.ones(n_tsub, dtype=bool)
        # count[ti] = number of selected blocks covering ti
        count = np.zeros(n_tsub, dtype=np.int32)
        selected = []
        remaining = n_tsub

        # For each tset, track how many candidate blocks cover it
        # We'll use a priority based on remaining candidates
        # Initially all blocks are candidates
        # We want to pick the most constrained tset (fewest candidate blocks)

        # To make this efficient, we maintain for each tset the number of
        # blocks that cover it and are still "available" (not yet selected).
        # But actually, since we're selecting blocks, a block is always available
        # unless we decide not to use it. Let's think differently.
        
        # Standard approach: at each step, among all uncovered tsets, find the one
        # with the fewest blocks that cover it AND that haven't been selected yet.
        # But we haven't eliminated any blocks, so every block that covers a tset
        # is a candidate. The number of candidates for tset ti is len(tset_to_blocks[ti]).
        # This is static! So the most constrained tsets are always the same.
        
        # Wait, that's not quite right. As we select blocks, we cover tsets.
        # The "constraint" should be about which tsets are still uncovered and
        # how many blocks could cover them. But since we haven't removed any blocks
        # from consideration, the number of blocks covering each tset is fixed.
        
        # So the criticality is: among uncovered tsets, pick the one with minimum
        # len(tset_to_blocks[ti]), breaking ties randomly.
        
        # Precompute the candidate count for each tset
        cand_count = np.array([len(tset_to_blocks[ti]) for ti in range(n_tsub)], dtype=np.int32)

        while remaining > 0:
            if time_left() < 10.0:
                break

            # Find uncovered tsets
            uncov_indices = np.where(uncovered)[0]
            if len(uncov_indices) == 0:
                break

            # Among uncovered, find the one with minimum cand_count
            # Ties broken randomly
            min_cand = np.min(cand_count[uncov_indices])
            candidates = uncov_indices[cand_count[uncov_indices] == min_cand]
            if len(candidates) > 1:
                ti = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                ti = int(candidates[0])

            # Now pick the best block covering this tset
            # Best = maximum marginal gain (number of newly covered tsets)
            best_bi = -1
            best_gain = -1
            for bj in tset_to_blocks[ti]:
                gain = 0
                for t_idx in block_tsets[bj]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bj
                elif gain == best_gain and gain > 0:
                    # Tie-break randomly among top gainers
                    if rng.random() < 0.5:
                        best_bi = bj

            if best_bi == -1 or best_gain == 0:
                break

            selected.append(best_bi)
            for t_idx in block_tsets[best_bi]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    remaining -= 1

        return selected

    # 2. Standard greedy (for comparison)
    def run_greedy(randomize_ties=False, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.array([len(ts) for ts in block_tsets], dtype=np.int32)

        selected = []
        remaining = n_tsub

        while remaining > 0:
            if time_left() < 10.0:
                break
            max_gain = np.max(gain)
            if max_gain == 0:
                break

            if randomize_ties:
                candidates = np.where(gain == max_gain)[0]
                bi = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                bi = int(np.argmax(gain))

            selected.append(bi)
            gain[bi] = 0

            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        gain[bj] -= 1

        return selected

    # 3. Fast reduction using numpy
    def reduce_solution(sel):
        if not sel:
            return []

        if time_left() < 5.0:
            return sel

        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            cov[block_tsets_np[bi]] += 1

        new_sel = []
        for bi in sel:
            bts = block_tsets_np[bi]
            if np.any(cov[bts] == 1):
                new_sel.append(bi)
            else:
                cov[bts] -= 1

        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel

    # 4. Local Search with 1-out-1 swaps (randomized)
    def local_search_1swap(selected, max_rounds=30, seed=None):
        if not selected or time_left() < 10.0:
            return selected

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        best_selected = list(selected)
        best_size = len(selected)

        for rnd in range(max_rounds):
            if time_left() < 10.0:
                break

            # Phase 1: Reduction
            selected = reduce_solution(selected)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)

            if not selected:
                break

            # Rebuild coverage
            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in selected:
                cov[block_tsets_np[bi]] += 1

            improved = False
            selected_set = set(selected)

            # Try 1-out-1 swaps
            n_swaps_to_try = min(len(selected) * 2, 100)
            for _ in range(n_swaps_to_try):
                if time_left() < 10.0:
                    break
                if not selected:
                    break

                idx_remove = rng.randint(0, len(selected) - 1)
                bi_remove = selected[idx_remove]

                bts_remove = block_tsets_np[bi_remove]
                newly_uncovered = bts_remove[cov[bts_remove] == 1]

                if len(newly_uncovered) == 0:
                    # Block is redundant
                    cov[bts_remove] -= 1
                    selected.pop(idx_remove)
                    selected_set.discard(bi_remove)
                    improved = True
                    continue

                # Find candidate blocks that cover ALL newly_uncovered tsets
                if len(newly_uncovered) > 6:
                    continue

                cand_set = set(tset_to_blocks[int(newly_uncovered[0])])
                for ti in newly_uncovered[1:]:
                    cand_set &= set(tset_to_blocks[int(ti)])

                cand_set.discard(bi_remove)
                cand_set -= selected_set

                if not cand_set:
                    continue

                cand_list = list(cand_set)
                if len(cand_list) > 5:
                    scores = [(len(block_tsets[bj]), bj) for bj in cand_list]
                    scores.sort(key=lambda x: -x[0])
                    cand_list = [s[1] for s in scores[:5]]

                bi_add = rng.choice(cand_list)
                bts_add = block_tsets_np[bi_add]

                cov[bts_remove] -= 1
                cov[bts_add] += 1

                selected[idx_remove] = bi_add
                selected_set.discard(bi_remove)
                selected_set.add(bi_add)
                improved = True

                # Quick reduction after swap
                selected = reduce_solution(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                    selected_set = set(selected)

                # Rebuild coverage for next iteration
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_np[bi]] += 1

            if not improved:
                break

        return best_selected

    # 5. Simulated Annealing
    def simulated_annealing(initial_blocks, seed=None):
        if not initial_blocks:
            return initial_blocks

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        current_blocks = list(initial_blocks)

        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_tsets_np[b_idx]] += 1

        current_energy = float(len(current_blocks)) + 1000.0 * float(np.sum(cover_count == 0))
        best_blocks = list(current_blocks)
        best_energy = current_energy

        T = 30.0
        cooling_rate = 0.995
        min_T = 0.5
        penalty = 1000.0

        while time_left() > 10.0 and T > min_T:
            # Move 1: Try to Remove
            if len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]

                valid_removal = True
                new_uncovered = 0
                for t_idx in block_tsets[b_idx]:
                    if cover_count[t_idx] == 1:
                        valid_removal = False
                        new_uncovered += 1

                if valid_removal:
                    cover_count[block_tsets_np[b_idx]] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    delta_E = 1.0 + penalty * new_uncovered
                    if rng.random() < np.exp(-delta_E / T):
                        cover_count[block_tsets_np[b_idx]] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E

            # Move 2: Try to Swap
            if len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]

                newly_uncovered = []
                for t_idx in block_tsets[b_idx]:
                    if cover_count[t_idx] == 1:
                        newly_uncovered.append(t_idx)

                if len(newly_uncovered) == 0:
                    cover_count[block_tsets_np[b_idx]] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                elif len(newly_uncovered) <= 5:
                    cand_set = set(tset_to_blocks[newly_uncovered[0]])
                    for ti in newly_uncovered[1:]:
                        cand_set &= set(tset_to_blocks[ti])

                    cand_set.discard(b_idx)
                    cand_set -= set(current_blocks)

                    if cand_set:
                        cand_list = list(cand_set)
                        if len(cand_list) > 5:
                            scores = [(len(block_tsets[bj]), bj) for bj in cand_list]
                            scores.sort(key=lambda x: -x[0])
                            cand_list = [s[1] for s in scores[:5]]

                        b_add = rng.choice(cand_list)

                        cover_count[block_tsets_np[b_idx]] -= 1
                        cover_count[block_tsets_np[b_add]] += 1
                        current_blocks[i] = b_add
                        current_energy = float(len(current_blocks)) + penalty * float(np.sum(cover_count == 0))

            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)

            T *= cooling_rate

        return best_blocks

    # 6. Main
    best_overall = None
    best_size_overall = float('inf')

    # Critical greedy (deterministic)
    if time_left() > 25.0:
        sel = critical_greedy(seed=None)
        sel = reduce_solution(sel)
        sel = local_search_1swap(sel, max_rounds=20, seed=42)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel

    # Standard greedy (deterministic)
    if time_left() > 20.0:
        sel = run_greedy(randomize_ties=False)
        sel = reduce_solution(sel)
        sel = local_search_1swap(sel, max_rounds=20, seed=43)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel

    # Randomized critical greedy restarts
    n_restarts = 0
    while time_left() > 15.0 and n_restarts < 4:
        n_restarts += 1
        seed = n_restarts * 104729
        sel = critical_greedy(seed=seed)
        if not sel:
            continue
        sel = reduce_solution(sel)
        sel = local_search_1swap(sel, max_rounds=15, seed=seed + 1)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel

    # Randomized standard greedy restarts
    while time_left() > 15.0 and n_restarts < 8:
        n_restarts += 1
        seed = n_restarts * 7919
        sel = run_greedy(randomize_ties=True, seed=seed)
        if not sel:
            continue
        sel = reduce_solution(sel)
        sel = local_search_1swap(sel, max_rounds=15, seed=seed + 1)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel

    if best_overall is None:
        best_overall = run_greedy(randomize_ties=False)
        best_overall = reduce_solution(best_overall)

    # Apply simulated annealing to the best solution
    if time_left() > 25.0:
        sa_result = simulated_annealing(best_overall, seed=12345)
        if sa_result and len(sa_result) < best_size_overall:
            best_size_overall = len(sa_result)
            best_overall = sa_result

    # Final reduction
    best_overall = reduce_solution(best_overall)

    return [list(all_blocks[bi]) for bi in best_overall]
