import time
import itertools
import numpy as np

# ---- global deadline state (per construct call) ----
_DEADLINE = 0.0
_TIME_BUDGET = 150.0
def _ok():
    return time.time() < _DEADLINE

def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_BUDGET
    pts = list(range(v))
    all_t = np.array(list(itertools.combinations(pts, t)), dtype=np.int32)
    nT = all_t.shape[0]
    C = np.zeros(nT, dtype=np.int32)   # coverage count per t-subset

    # ---- helper: coverage of a block as a set of t-subset indices ----
    def block_cover(block):
        idx = []
        for comb in itertools.combinations(block, t):
            pos = np.searchsorted(all_t, comb, side="left")
            if pos < nT and all_t[pos].tolist() == list(comb):
                idx.append(pos)
        return idx

    def add_block(block):
        for p in block_cover(block):
            C[p] += 1

    def remove_block(block):
        for p in block_cover(block):
            C[p] -= 1

    # ---- fast gain: how many uncovered t-subsets a candidate covers given partial block ----
    def gain(cand, partial):
        if len(partial) < t - 1:
            return 0
        subs = []
        for comb in itertools.combinations(partial, t - 1):
            s = list(comb) + [cand]
            s.sort()
            subs.append(s)
        idx = np.array(subs, dtype=np.int32)
        pos = np.searchsorted(all_t, idx, side="left")
        ok = pos < nT
        if not ok.any():
            return 0
        matched = (all_t[pos[ok]] == idx[ok]).all(axis=1)
        good = pos[ok][matched]
        if good.size == 0:
            return 0
        return int((C[good] == 0).sum())

    # ---- weighted greedy initialization ----
    blocks = []
    while _ok():
        unc = np.nonzero(C == 0)[0]
        if unc.size == 0:
            break
        # weight = number of uncovered t-subsets each point participates in
        w = np.zeros(v, dtype=np.int64)
        w[all_t[unc]] += 1
        # seed block with the highest-weight point
        block = [int(np.argmax(w))]
        while len(block) < k:
            cands = [x for x in pts if x not in block]
            best = max(cands, key=lambda x: (gain(x, block), w[x]))
            block.append(best)
        block.sort()
        add_block(block)
        blocks.append(block)

    # ---- local search: block replacement + removal, criticality-guided ----
    def uncovered_count():
        return int((C == 0).sum())

    def try_remove(blocks):
        for i, b in enumerate(blocks):
            cov = block_cover(b)
            if len(cov) == 0:
                continue
            if (C[cov] > 1).all():
                remove_block(b)
                blocks.pop(i)
                return True
        return False

    def try_replace(blocks):
        n = len(blocks)
        if n == 0:
            return False
        base_unc = int((C == 0).sum())
        # criticality: blocks whose removal would create the most new uncovered subsets
        crit = np.zeros(n, dtype=np.int64)
        for i, b in enumerate(blocks):
            cov = block_cover(b)
            if len(cov) == 0:
                continue
            crit[i] = int((C[cov] == 1).sum())
        order = np.argsort(-crit)
        for i in order:
            if not _ok():
                return False
            b = blocks[i]
            cov = block_cover(b)
            if len(cov) == 0:
                continue
            # points that appear in the t-subsets this block uniquely covers
            unique_pts = set()
            for p in cov:
                if C[p] == 1:
                    unique_pts.update(all_t[p].tolist())
            if not unique_pts:
                continue
            # candidate points: unique points plus a few others
            cands = set(unique_pts)
            cands.update(np.random.choice(v, size=min(v, 10), replace=False).tolist())
            cands = [x for x in cands if x not in b]
            if not cands:
                continue
            # try removing this block and adding a new block seeded by a candidate
            remove_block(b)
            best_new = None
            best_gain = 0
            for cand in cands:
                nb = [cand]
                # build new block greedily to cover newly-uncovered subsets
                while len(nb) < k:
                    c2 = [x for x in pts if x not in nb]
                    if not c2:
                        break
                    best2 = max(c2, key=lambda x: gain(x, nb))
                    if gain(best2, nb) == 0:
                        # no gain; stop extending
                        break
                    nb.append(best2)
                if len(nb) < t:
                    continue
                nb.sort()
                g = gain(nb, nb)  # full block coverage of currently-uncovered
                if g > best_gain:
                    best_gain = g
                    best_new = nb
            if best_new is not None and best_gain > 0:
                add_block(best_new)
                blocks[i] = best_new
                return True
            else:
                add_block(b)  # restore
                # try a plain fresh block (seeded by a random/unique point)
                seed = list(unique_pts)
                if not seed:
                    seed = [int(np.random.randint(v))]
                nb = [seed[0]]
                while len(nb) < k:
                    c2 = [x for x in pts if x not in nb]
                    if not c2:
                        break
                    best2 = max(c2, key=lambda x: gain(x, nb))
                    if gain(best2, nb) == 0:
                        break
                    nb.append(best2)
                if len(nb) >= t:
                    nb.sort()
                    if gain(nb, nb) > 0:
                        add_block(nb)
                        blocks.append(nb)
                        return True
        return False

    # ---- local search loop ----
    while _ok():
        improved = False
        if try_remove(blocks):
            improved = True
        if not improved and try_replace(blocks):
            improved = True
        if not improved:
            # perturb to escape local optimum: drop a low-criticality block, re-cover
            if len(blocks) > 10:
                n = len(blocks)
                crit = np.zeros(n, dtype=np.int64)
                for i, b in enumerate(blocks):
                    cov = block_cover(b)
                    if len(cov):
                        crit[i] = int((C[cov] == 1).sum())
                low = int(np.argmin(crit))
                b = blocks.pop(low)
                # re-run greedy to cover any gaps
                while _ok():
                    unc = np.nonzero(C == 0)[0]
                    if unc.size == 0:
                        break
                    w = np.zeros(v, dtype=np.int64)
                    w[all_t[unc]] += 1
                    blk = [int(np.argmax(w))]
                    while len(blk) < k:
                        c2 = [x for x in pts if x not in blk]
                        best2 = max(c2, key=lambda x: (gain(x, blk), w[x]))
                        blk.append(best2)
                    blk.sort()
                    add_block(blk)
                    blocks.append(blk)
                improved = True
        if not improved and not _ok():
            break

    # ---- ensure validity: cover any remaining gaps (safety net) ----
    while True:
        unc = np.nonzero(C == 0)[0]
        if unc.size == 0:
            break
        w = np.zeros(v, dtype=np.int64)
        w[all_t[unc]] += 1
        blk = [int(np.argmax(w))]
        while len(blk) < k:
            c2 = [x for x in pts if x not in blk]
            best2 = max(c2, key=lambda x: (gain(x, blk), w[x]))
            blk.append(best2)
        blk.sort()
        add_block(blk)
        blocks.append(blk)

    return blocks
