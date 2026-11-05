import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s kill

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

    # Precompute tset_block_count for ordering
    tset_block_count = [len(tset_to_blocks[ti]) for ti in range(n_tsub)]

    def greedy_cover(seed=0):
        """Efficient greedy set cover with optional randomized tie-breaking."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsub, dtype=bool)
        gains = np.array([len(block_tsets[i]) for i in range(n_blocks)], dtype=np.int32)

        selected = []
        remaining = n_tsub

        while remaining > 0 and time_left() > 5.0:
            # Find best block (with small noise for tie-breaking if seed > 0)
            if seed > 0:
                scores = gains.astype(np.float64)
                scores += rng.random() * 0.05
                bi = int(np.argmax(scores))
            else:
                bi = int(np.argmax(gains))

            if gains[bi] == 0:
                break

            selected.append(bi)
            gains[bi] = 0

            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        if gains[bj] > 0:
                            gains[bj] -= 1

        return selected

    def reduce_solution(sel):
        """Remove redundant blocks via iterative passes."""
        if not sel:
            return []
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1

        changed = True
        while changed:
            changed = False
            new_sel = []
            for bi in sel:
                essential = False
                for ti in block_tsets[bi]:
                    if cov[ti] == 1:
                        essential = True
                        break
                if essential:
                    new_sel.append(bi)
                else:
                    changed = True
                    for ti in block_tsets[bi]:
                        cov[ti] -= 1
            sel = new_sel

        return sel

    def local_search(sel):
        """Iterative removal/swap local search."""
        if not sel:
            return sel
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1

        sel_set = set(sel)
        rng = random.Random(123)
        improved = True
        while improved and time_left() > 5.0:
            improved = False
            for i in range(len(sel)):
                if time_left() < 5.0:
                    break
                bi_remove = sel[i]

                newly_uncovered = []
                for ti in block_tsets[bi_remove]:
                    if cov[ti] == 1:
                        newly_uncovered.append(ti)

                if not newly_uncovered:
                    # Safe removal
                    sel.pop(i)
                    for ti in block_tsets[bi_remove]:
                        cov[ti] -= 1
                    sel_set.discard(bi_remove)
                    improved = True
                    break

                # Try to swap with a block that covers all newly uncovered
                candidates = None
                for ti in newly_uncovered:
                    if candidates is None:
                        candidates = set(tset_to_blocks[ti])
                    else:
                        candidates.intersection_update(tset_to_blocks[ti])

                if not candidates:
                    continue
                candidates.difference_update(sel_set)
                if not candidates:
                    continue

                best_bj = rng.choice(list(candidates))
                sel[i] = best_bj
                sel_set.discard(bi_remove)
                sel_set.add(best_bj)

                for ti in block_tsets[bi_remove]:
                    cov[ti] -= 1
                for ti in block_tsets[best_bj]:
                    cov[ti] += 1

                improved = True
                break

        return sel

    def simulated_annealing(initial_blocks, max_time):
        """SA on block selection with remove/swap moves."""
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in block_tsets[b_idx]:
                cover_count[t_idx] += 1

        penalty = 1000.0
        current_energy = float(len(current_blocks)) + penalty * float(np.sum(cover_count == 0))
        best_blocks = list(current_blocks)
        best_energy = current_energy

        T = 50.0
        cooling_rate = 0.99
        min_T = 0.1
        rng = random.Random(42)

        while time_left() > 5.0 and T > min_T:
            # Try removal
            if len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]

                new_uncovered = 0
                for t_idx in block_tsets[b_idx]:
                    if cover_count[t_idx] == 1:
                        new_uncovered += 1

                if new_uncovered == 0:
                    # Valid removal
                    for t_idx in block_tsets[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    delta_E = 1.0 + penalty * new_uncovered
                    if rng.random() < np.exp(-delta_E / T):
                        for t_idx in block_tsets[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E

            # Try swap
            if len(current_blocks) > 0 and time_left() > 5.0:
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]

                newly_uncovered = []
                for t_idx in block_tsets[b_idx]:
                    if cover_count[t_idx] == 1:
                        newly_uncovered.append(t_idx)

                if newly_uncovered:
                    candidates = None
                    for ti in newly_uncovered:
                        if candidates is None:
                            candidates = set(tset_to_blocks[ti])
                        else:
                            candidates.intersection_update(tset_to_blocks[ti])
                    if candidates:
                        candidates.difference_update(set(current_blocks))
                        if candidates:
                            b_new = rng.choice(list(candidates))
                            # Compute energy change
                            gain = 0
                            for t_idx in block_tsets[b_new]:
                                if cover_count[t_idx] == 0:
                                    gain += 1
                            delta_E = -float(gain)  # size unchanged, gain reduces uncovered
                            if rng.random() < np.exp(-delta_E / T):
                                for t_idx in block_tsets[b_idx]:
                                    cover_count[t_idx] -= 1
                                for t_idx in block_tsets[b_new]:
                                    cover_count[t_idx] += 1
                                current_blocks[i] = b_new
                                current_energy += delta_E

            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)

            T *= cooling_rate

        return best_blocks

    def perturbation_repair(current, n_remove):
        """Remove n_remove random blocks, then greedy-repair."""
        if not current:
            return []
        current = list(current)
        if len(current) <= n_remove:
            current = []
        else:
            remove_idx = random.sample(range(len(current)), n_remove)
            for idx in sorted(remove_idx, reverse=True):
                current.pop(idx)

        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in current:
            for ti in block_tsets[bi]:
                cov[ti] += 1

        uncovered_mask = (cov == 0)
        remaining = int(np.sum(uncovered_mask))

        if remaining == 0:
            return current

        current_set = set(current)
        gains = np.zeros(n_blocks, dtype=np.int32)
        for bi in range(n_blocks):
            if bi in current_set:
                continue
            cnt = 0
            for ti in block_tsets[bi]:
                if uncovered_mask[ti]:
                    cnt += 1
            gains[bi] = cnt

        while remaining > 0 and time_left() > 5.0:
            bi = int(np.argmax(gains))
            if gains[bi] == 0:
                break
            current.append(bi)
            current_set.add(bi)
            gains[bi] = 0

            for ti in block_tsets[bi]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        if gains[bj] > 0:
                            gains[bj] -= 1

        return current

    # Main search loop
    best_sol = greedy_cover(0)
    best_sol = reduce_solution(best_sol)
    best_sol = local_search(best_sol)
    best_size = len(best_sol)

    # Try multiple greedy seeds for diversity
    for seed in range(1, 5):
        if time_left() < 20.0:
            break
        sol = greedy_cover(seed)
        sol = reduce_solution(sol)
        sol = local_search(sol)
        if len(sol) < best_size:
            best_sol = sol
            best_size = len(sol)

    # Simulated annealing polish
    if time_left() > 30.0:
        sa_sol = simulated_annealing(best_sol, deadline - 30.0)
        sa_sol = reduce_solution(sa_sol)
        sa_sol = local_search(sa_sol)
        if len(sa_sol) < best_size:
            best_sol = sa_sol
            best_size = len(sa_sol)

    # Perturbation + repair loop
    rng = random.Random(7)
    while time_left() > 15.0:
        n_remove = rng.randint(2, 5)
        perturbed = perturbation_repair(best_sol, n_remove)
        perturbed = reduce_solution(perturbed)
        perturbed = local_search(perturbed)
        if len(perturbed) < best_size:
            best_sol = perturbed
            best_size = len(perturbed)

        # Occasional SA on the perturbed solution
        if time_left() > 40.0 and rng.random() < 0.3:
            sa_sol = simulated_annealing(perturbed, deadline - 40.0)
            sa_sol = reduce_solution(sa_sol)
            sa_sol = local_search(sa_sol)
            if len(sa_sol) < best_size:
                best_sol = sa_sol
                best_size = len(sa_sol)

    # Final polish
    best_sol = local_search(best_sol)
    best_sol = reduce_solution(best_sol)

    return [list(all_blocks[bi]) for bi in best_sol]
