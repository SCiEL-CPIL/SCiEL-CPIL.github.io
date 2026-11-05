import time
import math
import random
import numpy as np

def construct():
    start = time.time()
    DEADLINE = start + 85.0
    n = 256

    # --- Build all APs (length >= 2) ---
    aps = []
    for d in range(1, n):
        for r in range(min(d, n)):
            idx = list(range(r, n, d))
            if len(idx) >= 2:
                aps.append(idx)
    A = len(aps)

    max_len = max(len(a) for a in aps)
    ap_arr = np.full((A, max_len), -1, dtype=np.int32)
    ap_len = np.zeros(A, dtype=np.int32)
    for ai, a in enumerate(aps):
        ap_arr[ai, :len(a)] = np.array(a, dtype=np.int32)
        ap_len[ai] = len(a)

    ap_membership = [[] for _ in range(n)]
    for ai, a in enumerate(aps):
        for pos in a:
            ap_membership[pos].append(ai)

    def update_ap(ai, signs):
        L = ap_len[ai]
        vals = signs[ap_arr[ai, :L]]
        cum = np.concatenate(([0], np.cumsum(vals)))
        min_pref = np.minimum.accumulate(cum)
        max_pref = np.maximum.accumulate(cum)
        return int(np.max(cum - min_pref)), int(np.min(cum - max_pref))

    def init_sums(signs):
        maxs = np.empty(A, dtype=np.int32)
        mins = np.empty(A, dtype=np.int32)
        for ai in range(A):
            maxs[ai], mins[ai] = update_ap(ai, signs)
        return maxs, mins

    def get_disc(maxs, mins):
        return int(max(max(np.max(maxs), -np.min(mins)), 1))

    def eval_disc(signs):
        maxs, mins = init_sums(signs)
        return get_disc(maxs, mins)

    def find_worst(maxs, mins, k=3):
        absd = np.maximum(maxs, -mins)
        k = min(k, A)
        return np.argsort(-absd)[:k]

    def offending_positions(ai, signs, maxs, mins):
        L = ap_len[ai]
        ap = ap_arr[ai, :L]
        vals = signs[ap]
        cum = np.concatenate(([0], np.cumsum(vals)))
        min_pref = np.minimum.accumulate(cum)
        max_pref = np.maximum.accumulate(cum)
        cand_max = cum[1:] - min_pref[:-1]
        cand_min = cum[1:] - max_pref[:-1]
        if maxs[ai] >= -mins[ai]:
            j = int(np.argmax(cand_max))
            end = j + 1
            st = int(np.argmin(cum[:end]))
        else:
            j = int(np.argmin(cand_min))
            end = j + 1
            st = int(np.argmax(cum[:end]))
        return list(ap[st:end])

    # --- Phase 0: candidates ---
    cands = []
    cands.append(np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]))
    cands.append(np.array([1 if i % 2 == 0 else -1 for i in range(n)]))
    for bs in (2, 4, 8, 16):
        arr = np.empty(n, dtype=np.int32)
        for i in range(n):
            blk = (i // bs) % 2
            pos = i % bs
            arr[i] = 1 if (pos < bs // 2) == (blk == 0) else -1
        cands.append(arr)
    for s in range(30):
        r = random.Random(s)
        cands.append(np.array([r.choice((-1, 1)) for _ in range(n)]))
    for s in range(20):
        r = random.Random(500 + s)
        arr = np.array([r.choice((-1, 1)) for _ in range(n)])
        while arr.sum() > 0:
            i = r.randint(0, n - 1)
            if arr[i] == 1:
                arr[i] = -1
                break
        cands.append(arr)

    best = cands[0].copy()
    best_disc = eval_disc(best)
    for c in cands[1:]:
        if time.time() > DEADLINE - 30:
            break
        d = eval_disc(c)
        if d < best_disc:
            best_disc = d
            best = c.copy()

    def do_flip(idx, signs, maxs, mins):
        old = signs[idx]
        signs[idx] = -old
        for ai in ap_membership[idx]:
            maxs[ai], mins[ai] = update_ap(ai, signs)
        return old

    def undo_flip(idx, old, signs, maxs, mins):
        signs[idx] = old
        for ai in ap_membership[idx]:
            maxs[ai], mins[ai] = update_ap(ai, signs)

    def repair_search(signs, maxs, mins, cur, rng, max_iters, t0, cool, cutoff):
        temp = t0
        ni = 0
        it = 0
        while time.time() < cutoff and ni < max_iters:
            it += 1
            if it % 4 == 0:
                idx = rng.randint(0, n - 1)
            else:
                worst = find_worst(maxs, mins, k=1)
                ai = int(worst[0])
                pos_list = offending_positions(ai, signs, maxs, mins)
                if not pos_list:
                    idx = rng.randint(0, n - 1)
                else:
                    extreme = maxs[ai] if maxs[ai] >= -mins[ai] else -mins[ai]
                    sgn = 1 if extreme > 0 else -1
                    pref = [p for p in pos_list if signs[p] == sgn]
                    pool = pref if pref else pos_list
                    idx = rng.choice(pool)
            old = do_flip(idx, signs, maxs, mins)
            nd = get_disc(maxs, mins)
            if nd < cur:
                cur = nd
                ni = 0
            else:
                delta = nd - cur
                if delta == 0:
                    cur = nd
                    ni += 1
                elif temp > 0.05 and rng.random() < math.exp(-delta / temp):
                    cur = nd
                    ni += 1
                else:
                    undo_flip(idx, old, signs, maxs, mins)
                    ni += 1
            temp *= cool
        return cur

    def targeted_polish(signs, maxs, mins, cur, max_rounds, cutoff):
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < cutoff:
            improved = False
            rounds += 1
            worst = find_worst(maxs, mins, k=4)
            for ai in worst:
                if time.time() > cutoff:
                    break
                pos_list = offending_positions(ai, signs, maxs, mins)
                if not pos_list:
                    continue
                best_flip = -1
                best_new = cur
                for p in pos_list:
                    old = do_flip(p, signs, maxs, mins)
                    nd = get_disc(maxs, mins)
                    undo_flip(p, old, signs, maxs, mins)
                    if nd < best_new:
                        best_new = nd
                        best_flip = p
                if best_flip >= 0:
                    old = do_flip(best_flip, signs, maxs, mins)
                    cur = get_disc(maxs, mins)
                    improved = True
            if not improved:
                for idx in range(n):
                    if time.time() > cutoff:
                        break
                    old = do_flip(idx, signs, maxs, mins)
                    nd = get_disc(maxs, mins)
                    undo_flip(idx, old, signs, maxs, mins)
                    if nd < cur:
                        old = do_flip(idx, signs, maxs, mins)
                        cur = get_disc(maxs, mins)
                        improved = True
                        break
        return cur

    # --- Phase 1: main repair search ---
    signs = best.copy()
    maxs, mins = init_sums(signs)
    cur = get_disc(maxs, mins)
    if cur < best_disc:
        best_disc = cur
        best = signs.copy()
    rng = random.Random(20240501)
    cur = repair_search(signs, maxs, mins, cur, rng, 4000, 2.0, 0.997, DEADLINE - 12)
    if cur < best_disc:
        best_disc = cur
        best = signs.copy()

    # --- Phase 2: polish best ---
    signs = best.copy()
    maxs, mins = init_sums(signs)
    cur = get_disc(maxs, mins)
    cur = targeted_polish(signs, maxs, mins, cur, 30, DEADLINE - 4)
    if cur < best_disc:
        best_disc = cur
        best = signs.copy()

    # --- Phase 3: restarts ---
    for rs in range(6):
        if time.time() > DEADLINE - 8:
            break
        signs = best.copy()
        r2 = random.Random(7777 + rs)
        for _ in range(8):
            i = r2.randint(0, n - 1)
            signs[i] = -signs[i]
        maxs, mins = init_sums(signs)
        cur = get_disc(maxs, mins)
        cur = repair_search(signs, maxs, mins, cur, r2, 1500, 1.5, 0.997, DEADLINE - 6)
        maxs, mins = init_sums(signs)
        cur = get_disc(maxs, mins)
        cur = targeted_polish(signs, maxs, mins, cur, 20, DEADLINE - 3)
        if cur < best_disc:
            best_disc = cur
            best = signs.copy()

    return [int(x) for x in best]
