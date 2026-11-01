import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # ---------- Precomputation ----------
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    block_tuple_to_idx = {b: i for i, b in enumerate(all_k_blocks)}

    # block_to_ts[j]: indices of t-subsets contained in k-block j
    # ts_to_blocks[i]: indices of k-blocks containing t-subset i
    block_to_ts = [[] for _ in range(num_k_blocks)]
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, block in enumerate(all_k_blocks):
        lst = block_to_ts[j]
        for ts in itertools.combinations(block, t):
            idx = ts_index[ts]
            lst.append(idx)
            ts_to_blocks[idx].append(j)

    base_gain = len(block_to_ts[0])  # C(k, t)

    # ---------- Fast greedy (randomized) ----------
    def greedy_random(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, base_gain, dtype=np.int32)
        uncovered = np.ones(num_t_subsets, dtype=bool)
        used = np.zeros(num_k_blocks, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 2.0:
                return None
            # candidates: not used and max gain
            g = gain.copy()
            g[used] = -1
            max_gain = g.max()
            if max_gain <= 0:
                break
            cands = np.where(g == max_gain)[0]
            b = int(cands[random.randint(0, len(cands) - 1)])
            sol.append(b)
            used[b] = True
            for ts_idx in block_to_ts[b]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for nb in ts_to_blocks[ts_idx]:
                        if not used[nb]:
                            gain[nb] -= 1
        if uncovered.any():
            return None
        return sol

    # ---------- Local search: redundancy prune + 1-exchange ----------
    def local_search(sol):
        if not sol:
            return sol
        # cover_count
        cc = np.zeros(num_t_subsets, dtype=np.int32)
        for b in sol:
            for ts_idx in block_to_ts[b]:
                cc[ts_idx] += 1
        sol = list(sol)

        # Phase 1: remove redundant blocks
        changed = True
        while changed:
            if time.time() > deadline - 1.0:
                break
            changed = False
            for i in range(len(sol)):
                redundant = True
                for ts_idx in block_to_ts[sol[i]]:
                    if cc[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    b = sol.pop(i)
                    for ts_idx in block_to_ts[b]:
                        cc[ts_idx] -= 1
                    changed = True
                    break

        # Phase 2: 1-exchange (remove block i, add block covering all its unique t-subsets)
        improved = True
        while improved:
            if time.time() > deadline - 1.0:
                break
            improved = False
            for i in range(len(sol)):
                if time.time() > deadline - 1.0:
                    break
                b = sol[i]
                unique = [ts_idx for ts_idx in block_to_ts[b] if cc[ts_idx] == 1]
                if not unique:
                    continue
                # candidate blocks that cover every unique t-subset
                cset = set(ts_to_blocks[unique[0]])
                for ts_idx in unique[1:]:
                    cset = cset.intersection(ts_to_blocks[ts_idx])
                    if not cset:
                        break
                if not cset:
                    continue
                # remove b
                sol.pop(i)
                for ts_idx in block_to_ts[b]:
                    cc[ts_idx] -= 1
                # pick a candidate not already in sol, preferring one that adds fewest new blocks? just pick min index
                in_sol = set(sol)
                add = None
                for cand in cset:
                    if cand not in in_sol:
                        add = cand
                        break
                if add is None:
                    # all candidates already present: revert and skip
                    sol.insert(i, b)
                    for ts_idx in block_to_ts[b]:
                        cc[ts_idx] += 1
                    continue
                # add it
                sol.insert(i, add)
                for ts_idx in block_to_ts[add]:
                    cc[ts_idx] += 1
                improved = True
                break

        return sol

    # ---------- Main search ----------
    best_sol = None
    best_len = float('inf')

    # Deterministic first (seed 0)
    s0 = greedy_random(0)
    if s0 is not None:
        best_sol = s0
        best_len = len(s0)

    # Randomized restarts for a better starting point
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        s = greedy_random(r + 1)
        if s is not None and len(s) < best_len:
            best_len = len(s)
            best_sol = s

    if best_sol is None:
        # Fallback: deterministic pure greedy (shouldn't happen, but safe)
        s = greedy_random(0)
        best_sol = s if s is not None else []
        best_len = len(best_sol)

    # Polish the best solution with local search
    if time.time() < deadline - 2.0:
        polished = local_search(best_sol)
        if len(polished) < best_len:
            best_len = len(polished)
            best_sol = polished

    # Convert to lists of ints
    result = [list(all_k_blocks[b]) for b in best_sol]
    return result
