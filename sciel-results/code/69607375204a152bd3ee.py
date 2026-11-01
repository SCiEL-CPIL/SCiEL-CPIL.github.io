import time
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0
    rng = np.random.default_rng(12345)
    pts = np.arange(v)

    # ---- index all t-subsets and map each to a row id ----
    tsubs = list(itertools.combinations(range(v), t))
    n_t = len(tsubs)
    tset = set(tsubs)
    t_index = {c: i for i, c in enumerate(tsubs)}

    # ---- all k-subsets (blocks) ----
    ksubs = list(itertools.combinations(range(v), k))
    n_k = len(ksubs)

    # ---- for each t-subset, list of block indices that cover it ----
    covers = [None] * n_t
    for i, c in enumerate(tsubs):
        cs = set(c)
        covers[i] = [j for j, b in enumerate(ksubs) if cs.issubset(set(b))]

    # ---- for each block, list of t-subset indices it covers ----
    block_cov = [None] * n_k
    for j, b in enumerate(ksubs):
        bs = set(b)
        block_cov[j] = [i for i, c in enumerate(tsubs) if set(c) <= bs]

    cov_arr = np.array([np.array(x, dtype=np.int64) for x in block_cov])  # (n_k, t)

    # ---- scarcity weight for each t-subset (inverse count of supersets) ----
    w = np.array([1.0 / max(1, len(covers[i])) for i in range(n_t)], dtype=np.float64)

    # ---- weighted greedy initial cover ----
    covered = np.zeros(n_t, dtype=bool)
    chosen = []
    Wtot = w.sum()
    while not covered.all():
        if time.time() > deadline:
            break
        best_j, best_score = -1, -1.0
        for j in range(n_k):
            idx = cov_arr[j]
            gain = w[~covered[idx]].sum()
            if gain > best_score:
                best_score = gain
                best_j = j
        chosen.append(best_j)
        covered[cov_arr[best_j]] = True
        Wtot -= best_score
        if Wtot <= 1e-9:
            break

    # ---- local search: minimize number of blocks while keeping full coverage ----
    def local_search(chosen):
        chosen = list(chosen)
        n = len(chosen)
        # coverage counts
        counts = np.zeros(n_t, dtype=np.int32)
        for j in chosen:
            counts[cov_arr[j]] += 1
        uncovered = int((counts == 0).sum())
        improved = True
        while improved and uncovered == 0 and time.time() < deadline:
            improved = False
            # ADD-MINIMIZE
            add_gain = np.array([int((counts[cov_arr[j]] == 0).sum()) for j in range(n_k)], dtype=np.int64)
            best_j = int(np.argmax(add_gain))
            if add_gain[best_j] > 0:
                counts[cov_arr[best_j]] += 1
                chosen.append(best_j)
                # remove any redundant blocks (in reverse order)
                m = len(chosen)
                for a in range(m - 1, -1, -1):
                    counts[cov_arr[chosen[a]]] -= 1
                    if (counts == 0).any():
                        counts[cov_arr[chosen[a]]] += 1
                    else:
                        chosen.pop(a)
                        improved = True
                        break
            # SWAP
            if time.time() < deadline:
                keep = np.array(chosen, dtype=np.int64)
                for pos in range(len(keep)):
                    base = counts - cov_arr[keep[pos]]  # coverage if we drop pos
                    if (base < 0).any():
                        continue
                    # deficit of subsets that become uncovered when dropping pos
                    deficit = base == 0
                    if not deficit.any():
                        # pos is redundant: remove it
                        counts = base
                        chosen.pop(pos)
                        improved = True
                        break
                    # find a candidate block covering many deficit subsets
                    cand_gain = np.array(
                        [int((deficit[cov_arr[j]]).sum()) for j in range(n_k)], dtype=np.int64)
                    cand_gain[keep] = -1
                    bj = int(np.argmax(cand_gain))
                    if cand_gain[bj] > 0:
                        counts = base + cov_arr[bj]
                        chosen[pos] = bj
                        improved = True
                        break
        return chosen

    if uncovered == 0:
        chosen = local_search(chosen)
    else:
        # greedy didn't finish; fill remaining uncovered with a quick greedy, then LS
        while not covered.all() and time.time() < deadline:
            best_j, best_score = -1, -1.0
            for j in range(n_k):
                gain = w[~covered[cov_arr[j]]].sum()
                if gain > best_score:
                    best_score, best_j = gain, j
            chosen.append(best_j)
            covered[cov_arr[best_j]] = True
        if covered.all():
            chosen = local_search(chosen)

    # ---- build output blocks (distinct, sorted) ----
    seen = set()
    blocks = []
    for j in chosen:
        b = tuple(ksubs[j])
        if b not in seen:
            seen.add(b)
            blocks.append(list(b))

    # ---- safety: ensure full coverage; if not, append greedy blocks ----
    if not blocks:
        # fallback: pure greedy from scratch with deadline
        blocks = []
        cov = set()
        all_t = tsubs
        while True:
            rem = [c for c in all_t if c not in cov]
            if not rem:
                break
            if time.time() > deadline:
                break
            best_b, best_g = None, -1
            for j in range(n_k):
                g = len(set(ksubs[j]) & set(rem))
                if g > best_g:
                    best_g, best_b = g, ksubs[j]
            blocks.append(list(best_b))
            cov |= set(itertools.combinations(best_b, t))
        return blocks

    # verify coverage; patch if needed
    cov = set()
    for b in blocks:
        cov |= set(itertools.combinations(b, t))
    missing = [c for c in tsubs if c not in cov]
    if missing:
        # patch greedily
        mset = set(missing)
        while mset and time.time() < deadline:
            best_b, best_g = None, -1
            for j in range(n_k):
                g = len(mset & set(itertools.combinations(ksubs[j], t)))
                if g > best_g:
                    best_g, best_b = g, ksubs[j]
            if best_b is None:
                break
            blocks.append(list(best_b))
            mset -= set(itertools.combinations(best_b, t))
    return blocks
