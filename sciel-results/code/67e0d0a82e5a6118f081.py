import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix
    M = candidates @ candidates.T

    # Conflict matrix: conflict[i,j] = True if M[i,j] > 2
    conflict = (M > 2)
    np.fill_diagonal(conflict, False)

    # Integer adjacency for fast matrix-vector products
    conflict_adj = conflict.astype(np.int32)
    degrees = np.sum(conflict_adj, axis=1)

    rng = np.random.default_rng(42)

    best_set = None
    best_count = 0

    def count_conflicts(x):
        """Count number of conflicting pairs in binary vector x."""
        return (x @ (conflict_adj @ x)) // 2

    # Simulated annealing: maximize |x| - C * (#conflicting pairs)
    # with C large enough that valid solutions are always preferred
    C_penalty = 3.0

    num_restarts = 12

    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break

        # Greedy initialization with varied ordering
        if restart == 0:
            order = np.argsort(degrees, kind='stable')
        elif restart == 1:
            order = np.argsort(-degrees, kind='stable')
        elif restart == 2:
            order = np.arange(N)
        elif restart == 3:
            order = np.arange(N - 1, -1, -1)
        elif restart % 4 == 0:
            order = rng.permutation(N)
        elif restart % 4 == 1:
            noise = rng.random(N) * 0.5
            order = np.argsort(degrees + noise, kind='stable')
        elif restart % 4 == 2:
            weights = 1.0 / (degrees + 1)
            order = rng.choice(N, size=N, replace=False, p=weights / weights.sum())
        else:
            order = rng.permutation(N)

        # Build initial independent set greedily
        x = np.zeros(N, dtype=np.int32)
        available = np.ones(N, dtype=bool)
        for idx in order:
            if not available[idx]:
                continue
            x[idx] = 1
            available[idx] = False
            available[conflict[idx]] = False
            if not np.any(available):
                break

        cur_cnt = np.sum(x)
        cur_conf = count_conflicts(x)

        if cur_conf == 0 and cur_cnt > best_count:
            best_count = cur_cnt
            best_set = np.where(x == 1)[0].copy()

        # Simulated annealing
        T = 1.2
        cooling_rate = 0.996
        min_T = 0.003
        max_iters = 4000

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Pick random vertex to flip
            idx = rng.integers(N)

            # Compute objective change
            if x[idx] == 0:
                new_conf = np.sum(conflict_adj[idx] * x)
                delta = 1.0 - C_penalty * new_conf
            else:
                new_conf = np.sum(conflict_adj[idx] * x)
                delta = -1.0 + C_penalty * new_conf

            # Metropolis acceptance
            if delta >= 0:
                x[idx] = 1 - x[idx]
            else:
                if T > min_T and rng.random() < np.exp(delta / T):
                    x[idx] = 1 - x[idx]

            # Periodic check for valid improvement
            if it % 40 == 0:
                cc = np.sum(x)
                cf = count_conflicts(x)
                if cf == 0 and cc > best_count:
                    best_count = cc
                    best_set = np.where(x == 1)[0].copy()

            # Cool
            T *= cooling_rate
            if T < min_T:
                T = min_T

        # Post-SA local repair: eliminate conflicts if any
        if best_set is not None:
            x_check = np.zeros(N, dtype=np.int32)
            for i in best_set:
                x_check[i] = 1
            if count_conflicts(x_check) > 0:
                for _ in range(200):
                    sel = np.where(x_check == 1)[0]
                    if len(sel) == 0:
                        break
                    max_conf = -1
                    worst = -1
                    for i in sel:
                        c = np.sum(conflict_adj[i] * x_check)
                        if c > max_conf:
                            max_conf = c
                            worst = i
                    if max_conf <= 0:
                        break
                    x_check[worst] = 0
                cnt = np.sum(x_check)
                if cnt > best_count:
                    best_count = cnt
                    best_set = np.where(x_check == 1)[0].copy()

    # Final extension: greedily add compatible vertices to best_set
    if best_set is not None and len(best_set) > 0:
        x_final = np.zeros(N, dtype=np.int32)
        for i in best_set:
            x_final[i] = 1

        available = np.ones(N, dtype=bool)
        for i in best_set:
            available[i] = False
        for i in best_set:
            available[conflict[i]] = False

        avail_list = np.where(available)[0]
        if len(avail_list) > 0:
            avail_conf = np.array([np.sum(conflict_adj[i] & available) for i in avail_list])
            sorted_avail = avail_list[np.argsort(avail_conf)]
            for idx in sorted_avail:
                if not available[idx]:
                    continue
                x_final[idx] = 1
                available[idx] = False
                available[conflict[idx]] = False
                if not np.any(available):
                    break

        final_cnt = np.sum(x_final)
        if final_cnt > best_count:
            best_count = final_cnt
            best_set = np.where(x_final == 1)[0].copy()

    if best_set is not None and len(best_set) > 0:
        return candidates[best_set].astype(np.int64)

    # Fallback: Type 1 vectors (all orthogonal, dot product 0 <= 2)
    return candidates[:22].astype(np.int64)
