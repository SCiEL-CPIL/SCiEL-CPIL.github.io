import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before the 120s kill

    # ---------- Precomputation ----------
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {s: i for i, s in enumerate(all_ts)}

    all_ks = list(itertools.combinations(range(v), k))
    n_ks = len(all_ks)

    # number of t-subsets inside a k-set
    t_count = int(np.prod([k - i for i in range(t)]) / np.prod([i + 1 for i in range(t)]))

    # block_t_indices[i, j] = index of the j-th t-subset contained in k-subset i
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = ts_index[sub]

    # For each point, the list of k-subset indices that contain it.
    point_to_blocks = [[] for _ in range(v)]
    for i, block in enumerate(all_ks):
        for p in block:
            point_to_blocks[p].append(i)

    # For each t-subset index, the list of k-subset indices that contain it.
    ts_to_blocks = [[] for _ in range(n_ts)]
    for i, block in enumerate(all_ks):
        for sub in itertools.combinations(block, t):
            ts_to_blocks[ts_index[sub]].append(i)

    def _covered_set(block_indices):
        cov = np.zeros(n_ts, dtype=bool)
        for b in block_indices:
            cov[block_t_indices[b]] = True
        return cov

    def _peeling(v, k, t, deadline):
        """Hypergraph peeling: repeatedly take the residual point of minimum
        uncovered-degree and add the precomputed block that maximally covers the
        uncovered triples incident to that point."""
        cov = np.zeros(n_ts, dtype=bool)
        selected = []
        while True:
            if time.time() > deadline - 5.0:
                return None
            if cov.all():
                break
            # uncovered triples incident to each point
            deg = np.zeros(v, dtype=np.int64)
            for p in range(v):
                deg[p] = sum(1 for b in point_to_blocks[p] if (~cov[block_t_indices[b]]).any())
            p = int(np.argmin(deg))
            if deg[p] == 0:
                return None  # stuck: some uncovered triple not incident to any point
            # choose the block through p that covers the most uncovered triples
            best_b = None
            best_gain = -1
            for b in point_to_blocks[p]:
                g = int((~cov[block_t_indices[b]]).sum())
                if g > best_gain:
                    best_gain = g
                    best_b = b
            if best_b is None or best_gain <= 0:
                return None
            selected.append(best_b)
            cov[block_t_indices[best_b]] = True
        return selected

    def _greedy_randomized(seed, deadline):
        """Whole-block randomized greedy set cover."""
        rng = np.random.RandomState(seed)
        cov = np.zeros(n_ts, dtype=bool)
        selected = []
        while True:
            if time.time() > deadline - 5.0:
                return None
            if cov.all():
                break
            gains = (~cov[block_t_indices]).sum(axis=1)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                return None
            cands = np.where(gains == max_gain)[0]
            b = int(cands[rng.randint(len(cands))])
            selected.append(b)
            cov[block_t_indices[b]] = True
        return selected

    def _reduce(block_indices, deadline):
        """Remove redundant blocks (every t-subset covered by >= 2 blocks)."""
        current = list(block_indices)
        while True:
            if time.time() > deadline - 2.0:
                break
            coverage = np.zeros(n_ts, dtype=np.int32)
            for b in current:
                coverage[block_t_indices[b]] += 1
            kept = []
            removed_any = False
            for b in current:
                subs = block_t_indices[b]
                if np.all(coverage[subs] > 1):
                    coverage[subs] -= 1
                    removed_any = True
                else:
                    kept.append(b)
            current = kept
            if not removed_any:
                break
        return current

    def _local_search(block_indices, deadline):
        """Swap a point inside a block to increase net new-triple coverage."""
        rng = random.Random(int(time.time() * 1000) % 100000)
        best = list(block_indices)
        while time.time() < deadline - 3.0:
            coverage = np.zeros(n_ts, dtype=np.int32)
            for b in best:
                coverage[block_t_indices[b]] += 1
            improved = False
            for bi in range(len(best)):
                if time.time() > deadline - 3.0:
                    break
                b = best[bi]
                block_pts = list(all_ks[b])
                for rp in block_pts:
                    new_set = set(block_pts) - {rp}
                    # find best add point
                    best_add = None
                    best_delta = 0
                    for ap in range(v):
                        if ap in new_set:
                            continue
                        cand = tuple(sorted(new_set | {ap}))
                        # find a k-subset index for cand
                        # (cand is a valid k-subset)
                        # compute delta in coverage
                        old_subs = set(block_t_indices[b])
                        new_subs = set(ts_index[sub] for sub in itertools.combinations(cand, t))
                        gain = sum(1 for s in new_subs if coverage[s] == 0)
                        loss = sum(1 for s in old_subs - new_subs if coverage[s] == 1)
                        delta = gain - loss
                        if delta > best_delta:
                            best_delta = delta
                            best_add = cand
                    if best_add is not None and best_delta > 0:
                        # apply
                        old_subs = block_t_indices[b]
                        coverage[old_subs] -= 1
                        new_idx = all_ks.index(best_add)
                        coverage[block_t_indices[new_idx]] += 1
                        best[bi] = new_idx
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    best_blocks = None
    best_size = float('inf')

    def _accept(indices):
        nonlocal best_blocks, best_size
        if indices is None:
            return
        reduced = _reduce(indices, deadline)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = list(reduced)

    # Peeling attempts (fast, diverse)
    for s in range(60):
        if time.time() > deadline - 25.0:
            break
        _accept(_peeling(v, k, t, deadline))

    # Greedy set-cover attempts
    while time.time() < deadline - 15.0:
        s = int((time.time() - start_time) * 1000) % 1000000
        _accept(_greedy_randomized(s, deadline))

    # Local search polish on the best found
    if best_blocks is not None and time.time() < deadline - 8.0:
        polished = _local_search(best_blocks, deadline)
        polished = _reduce(polished, deadline)
        if len(polished) < best_size:
            best_size = len(polished)
            best_blocks = polished

    if best_blocks is None:
        # last-resort deterministic greedy
        idx = _greedy_randomized(0, deadline)
        if idx is not None:
            best_blocks = _reduce(idx, deadline)
        else:
            return []

    return [list(all_ks[i]) for i in best_blocks]
