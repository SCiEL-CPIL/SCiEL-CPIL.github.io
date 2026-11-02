import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    N = q ** n

    # ---- Precompute neighbors (Hamming ball of `radius`) as index arrays ----
    def word_to_idx(word):
        idx = 0
        for k in range(n):
            idx = idx * q + word[k]
        return idx

    # base word = all zeros
    base = [0] * n
    nbs = [0]
    if radius == 1:
        for pos in range(n):
            for val in range(1, q):
                w = base[:]; w[pos] = val
                nbs.append(word_to_idx(w))
    else:
        import itertools
        for d in range(1, radius + 1):
            for positions in itertools.combinations(range(n), d):
                for vals in itertools.product(range(1, q), repeat=d):
                    w = base[:]; changed = False
                    for pos, val in zip(positions, vals):
                        if val != w[pos]:
                            w[pos] = val; changed = True
                    if changed:
                        nbs.append(word_to_idx(w))
    nbs = np.array(sorted(set(nbs)), dtype=np.int64)

    # For each word index, its neighbor indices = (i + nbs) mod N
    idx = np.arange(N, dtype=np.int64)
    neighbor_cols = (idx[:, None] + nbs[None, :]) % N          # shape (N, M)
    M = nbs.shape[0]

    # coverage count matrix: cover[i, j] = 1 if j in ball(i)
    cover = np.zeros((N, N), dtype=np.int32)
    cover[np.arange(N)[:, None], neighbor_cols] = 1

    def verify(code):
        if not code:
            return N == 0
        cc = cover[code].sum(axis=0)
        return bool((cc > 0).all())

    def greedy(seed=None):
        rng = np.random.default_rng(seed)
        covered = np.zeros(N, dtype=np.uint8)
        code = []
        rem = N
        while rem > 0:
            unc = (1 - covered).astype(np.int32)
            scores = cover @ unc
            mx = scores.max()
            if mx == 0:
                break
            cand = np.where(scores == mx)[0]
            c = int(rng.choice(cand)) if cand.size > 1 else int(cand[0])
            code.append(c)
            new = cover[c] & (1 - covered)
            covered |= cover[c]
            rem -= int(new.sum())
            if time.time() > deadline - 20:
                break
        return code

    def local_search(code):
        code = list(code)
        cc = cover[code].sum(axis=0)
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                if time.time() > deadline - 15:
                    return code
                c = code[i]
                m = cover[c]
                if cc[m > 0].min() >= 2:
                    cc -= m
                    code.pop(i)
                    changed = True
                    break
        # swaps
        for _ in range(120):
            if time.time() > deadline - 15:
                break
            c = int(np.random.randint(N))
            if c in code:
                continue
            for i in range(len(code)):
                cj = code[i]
                m1 = cover[cj]; m2 = cover[c]
                if ((cc == 1) & m1 & (1 - m2)).any():
                    continue
                cc = cc - m1 + m2
                code[i] = c
                break
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                if time.time() > deadline - 15:
                    return code
                c = code[i]
                m = cover[c]
                if cc[m > 0].min() >= 2:
                    cc -= m
                    code.pop(i)
                    changed = True
                    break
        return code

    def anneal(code):
        code = list(code)
        cc = cover[code].sum(axis=0)
        best = code[:]; best_cc = cc.copy(); best_len = len(code)
        T = 1.0
        for it in range(3000):
            if time.time() > deadline - 12:
                break
            op = np.random.randint(3)
            if op == 0:  # remove
                if len(code) <= best_len:
                    continue
                i = int(np.random.randint(len(code)))
                c = code[i]; m = cover[c]
                if ((cc == 1) & m).any():
                    continue
                cc -= m; code.pop(i)
            elif op == 1:  # swap
                c = int(np.random.randint(N))
                if c in code:
                    continue
                i = int(np.random.randint(len(code)))
                cj = code[i]; m1 = cover[cj]; m2 = cover[c]
                lost = ((cc == 1) & m1 & (1 - m2)).sum()
                gained = (m2 & (cc == 0)).sum()
                delta = gained - lost
                if delta >= 0 or np.random.rand() < np.exp(delta / max(T, 1e-6)):
                    cc = cc - m1 + m2; code[i] = c
            else:  # add
                c = int(np.random.randint(N))
                if c in code:
                    continue
                gained = (cover[c] & (cc == 0)).sum()
                delta = gained - 1
                if delta >= 0 or np.random.rand() < np.exp(delta / max(T, 1e-6)):
                    cc += cover[c]; code.append(c)
            if len(code) < best_len:
                best = code[:]; best_cc = cc.copy(); best_len = len(code)
            T *= 0.999
        return best

    # ---- known 73-word construction (q=3, n=6, radius=1) ----
    def known73():
        if q != 3 or n != 6 or radius != 1:
            return None
        base = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        res = []
        for h in base:
            for t0 in range(3):
                for t1 in range(3):
                    res.append(h + [t0, t1])
        return res

    # ---- search ----
    best = None; best_size = N + 1

    def consider(code):
        nonlocal best, best_size
        code = local_search(code)
        if code and verify(code) and len(code) < best_size:
            best = code; best_size = len(code)
            return True
        return False

    # deterministic greedy (seed 0)
    consider(greedy(0))
    # anneal from that
    consider(anneal(best if best else greedy(0)))

    rng = np.random.default_rng(12345)
    for it in range(4000):
        if time.time() > deadline - 25:
            break
        if best is not None and best_size <= 73:
            break
        code = greedy(int(rng.integers(0, 2**32)))
        consider(code)
        if time.time() < deadline - 12:
            consider(anneal(best if best else code))

    # fallback
    if best is None or not verify(best):
        k = known73()
        if k is not None and verify(k):
            return k
        if best is not None:
            return best
    if best is not None:
        return best
    return []
