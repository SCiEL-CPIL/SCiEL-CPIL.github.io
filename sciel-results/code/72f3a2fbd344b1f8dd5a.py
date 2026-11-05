import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            ts_indices.append(tset_index[ts])
        block_tsets.append(ts_indices)

    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)

    # 2. Reduction Pass - iterative
    def reduce_solution(sel):
        if not sel:
            return []
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1

        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)

        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel

    # 3. Randomized Greedy Set Cover
    def greedy_solution(noise_scale=0.0):
        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            gain[bi] = len(block_tsets[bi])

        selected = []
        remaining = n_tsub

        while remaining > 0:
            if time.time() > deadline - 30.0:
                break
            if noise_scale > 0:
                noise = np.random.uniform(0, noise_scale, n_blocks)
                scores = gain.astype(np.float64) + noise
                bi = int(np.argmax(scores))
            else:
                bi = int(np.argmax(gain))
            if gain[bi] == 0:
                break
            selected.append(bi)
            gain[bi] = 0

            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        gain[bj] -= 1

        return selected

    # 4. Local search: systematic deletion + swap + simulated annealing
    def local_search(selected, time_budget):
        if not selected:
            return selected

        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in selected:
            for ti in block_tsets[bi]:
                cov[ti] += 1

        best_selected = list(selected)
        best_size = len(selected)
        end_time = time.time() + time_budget

        # Systematic deletion pass
        def deletion_pass(sel, cov):
            changed = True
            while changed:
                changed = False
                random.shuffle(sel)
                for i in range(len(sel)):
                    if time.time() > end_time - 5.0:
                        return sel, cov
                    bi = sel[i]
                    redundant = True
                    for ti in block_tsets[bi]:
                        if cov[ti] == 1:
                            redundant = False
                            break
                    if redundant:
                        for ti in block_tsets[bi]:
                            cov[ti] -= 1
                        sel.pop(i)
                        changed = True
                        break
            return sel, cov

        # Swap pass: try to replace a block with a better one
        def swap_pass(sel, cov):
            if time.time() > end_time - 5.0:
                return sel, cov
            improved = True
            while improved:
                improved = False
                indices = list(range(len(sel)))
                random.shuffle(indices)
                for i in indices:
                    if time.time() > end_time - 5.0:
                        return sel, cov
                    bi_old = sel[i]
                    # Find unique subsets covered by bi_old
                    unique_subs = [ti for ti in block_tsets[bi_old] if cov[ti] == 1]
                    if not unique_subs:
                        continue
                    # Find candidates that cover all unique_subs
                    candidates = None
                    for ti in unique_subs:
                        if candidates is None:
                            candidates = set(tset_to_blocks[ti])
                        else:
                            candidates.intersection_update(tset_to_blocks[ti])
                            if not candidates:
                                break
                    if not candidates:
                        continue
                    candidates.discard(bi_old)
                    if not candidates:
                        continue
                    # Try each candidate
                    for bi_new in candidates:
                        if bi_new in sel:
                            continue
                        # Check validity
                        valid = True
                        for ti in block_tsets[bi_old]:
                            if cov[ti] == 1 and ti not in block_tsets[bi_new]:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Perform swap
                        sel[i] = bi_new
                        for ti in block_tsets[bi_old]:
                            cov[ti] -= 1
                        for ti in block_tsets[bi_new]:
                            cov[ti] += 1
                        improved = True
                        break
            return sel, cov

        # Simulated annealing
        T_start = 5.0
        T_end = 0.1
        cooling_rate = 0.995
        T = T_start

        while time.time() < end_time - 5.0 and T > T_end:
            if not selected:
                break

            # Periodic systematic passes
            if random.random() < 0.1:
                selected, cov = deletion_pass(selected, cov)
                selected, cov = swap_pass(selected, cov)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                T *= cooling_rate
                continue

            move_type = random.random()

            if move_type < 0.5:
                # Kick: remove one block, greedily refill uncovered
                idx = random.randint(0, len(selected) - 1)
                bi_kick = selected[idx]

                saved_selected = list(selected)
                saved_cov = cov.copy()

                selected.pop(idx)
                for ti in block_tsets[bi_kick]:
                    cov[ti] -= 1

                uncovered_tsets = [ti for ti in block_tsets[bi_kick] if cov[ti] == 0]

                if not uncovered_tsets:
                    # Redundant block removed
                    selected, cov = deletion_pass(selected, cov)
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = list(selected)
                    T *= cooling_rate
                    continue

                # Greedily cover uncovered
                new_blocks = []
                uncovered_set = set(uncovered_tsets)

                while uncovered_set and time.time() < end_time - 5.0:
                    best_bj = -1
                    max_cover = 0
                    candidates = set()
                    for ti in uncovered_set:
                        candidates.update(tset_to_blocks[ti])

                    for bj in candidates:
                        cnt = 0
                        for ti in block_tsets[bj]:
                            if ti in uncovered_set:
                                cnt += 1
                        if cnt > max_cover:
                            max_cover = cnt
                            best_bj = bj

                    if best_bj == -1:
                        break

                    new_blocks.append(best_bj)
                    for ti in block_tsets[best_bj]:
                        if ti in uncovered_set:
                            uncovered_set.remove(ti)

                for bj in new_blocks:
                    selected.append(bj)
                    for ti in block_tsets[bj]:
                        cov[ti] += 1

                selected, cov = deletion_pass(selected, cov)

                delta = len(selected) - len(saved_selected)
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = list(selected)
                else:
                    selected = saved_selected
                    cov = saved_cov

            else:
                # Swap: remove one block, add one different block
                if len(selected) < 2:
                    T *= cooling_rate
                    continue

                idx_remove = random.randint(0, len(selected) - 1)
                bi_remove = selected[idx_remove]

                single_covered = [ti for ti in block_tsets[bi_remove] if cov[ti] == 1]

                if single_covered:
                    candidates = set()
                    for ti in single_covered:
                        candidates.update(tset_to_blocks[ti])
                    candidates.discard(bi_remove)
                    if not candidates:
                        T *= cooling_rate
                        continue
                    bi_add = random.choice(list(candidates))
                else:
                    bi_add = random.randint(0, n_blocks - 1)
                    if bi_add == bi_remove:
                        T *= cooling_rate
                        continue

                if bi_add in selected:
                    T *= cooling_rate
                    continue

                saved_selected = list(selected)
                saved_cov = cov.copy()

                selected[idx_remove] = bi_add
                for ti in block_tsets[bi_remove]:
                    cov[ti] -= 1
                for ti in block_tsets[bi_add]:
                    cov[ti] += 1

                valid = True
                for ti in block_tsets[bi_remove]:
                    if cov[ti] == 0:
                        valid = False
                        break

                if not valid:
                    selected = saved_selected
                    cov = saved_cov
                    T *= cooling_rate
                    continue

                selected, cov = deletion_pass(selected, cov)

                delta = len(selected) - len(saved_selected)
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = list(selected)
                else:
                    selected = saved_selected
                    cov = saved_cov

            T *= cooling_rate

        return best_selected

    # 5. Main: multiple randomized greedy restarts + local search
    best_overall = None
    best_size_overall = float('inf')

    # Phase 1: Multiple greedy restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        noise = 0.0 if r == 0 else 0.5
        sel = greedy_solution(noise)
        if not sel:
            continue
        sel = reduce_solution(sel)
        if len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = list(sel)

    if best_overall is None:
        # Fallback: pure greedy
        best_overall = greedy_solution(0.0)
        best_overall = reduce_solution(best_overall)
        return [list(all_blocks[bi]) for bi in best_overall]

    # Phase 2: Local search on best solution
    remaining_time = deadline - 10.0 - time.time()
    if remaining_time > 10.0:
        best_overall = local_search(best_overall, remaining_time)

    # Phase 3: Final polish - aggressive deletion
    best_overall = reduce_solution(best_overall)

    return [list(all_blocks[bi]) for bi in best_overall]
