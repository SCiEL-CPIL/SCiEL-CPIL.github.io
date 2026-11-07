import numpy as np
import time
import random
import math

N = 128

def _is_valid(mask):
    """True iff no three distinct members a,b,c have a+b,a+c,b+c all in mask."""
    m2 = np.zeros(2 * N + 1, dtype=bool)
    m2[:N + 1] = mask
    elems = np.nonzero(mask)[0]
    k = len(elems)
    if k < 3:
        return True
    for i in range(k):
        a = elems[i]
        for j in range(i + 1, k):
            b = elems[j]
            s = a + b
            if s > N or not mask[s]:
                continue
            cands = elems
            ac = a + cands
            bc = b + cands
            ok = m2[ac] & m2[bc]
            ok[cands == a] = False
            ok[cands == b] = False
            if ok.any():
                return False
    return True

def _greedy_fill(mask):
    """Add any missing element if it keeps the set valid (monotone size increase)."""
    changed = True
    while changed:
        changed = False
        for v in range(1, N + 1):
            if mask[v]:
                continue
            mask[v] = True
            if _is_valid(mask):
                changed = True
            else:
                mask[v] = False
    return mask

def _initial(seed):
    rng = random.Random(seed)
    mask = np.zeros(N + 1, dtype=bool)
    mask[N // 2 + 1:] = True          # sum-free upper half baseline
    small = list(range(1, N // 2 + 1))
    rng.shuffle(small)
    for v in small:
        mask[v] = True
        if not _is_valid(mask):
            mask[v] = False
    _greedy_fill(mask)
    return mask

def _sa(mask, deadline, seed):
    rng = random.Random(seed)
    cur = mask.copy()
    cur_size = int(cur.sum())
    best = cur.copy()
    best_size = cur_size
    T = 6.0
    while time.time() < deadline:
        in_set = np.nonzero(cur)[0]
        out_set = np.nonzero(~cur)[0]
        out_set = out_set[out_set >= 1]
        if len(in_set) == 0 or len(out_set) == 0:
            break
        r = rng.random()
        new = None
        delta = 0
        if r < 0.30:
            # simple add
            x = int(rng.choice(out_set))
            new = cur.copy()
            new[x] = True
            delta = 1
            if not _is_valid(new):
                new = None
        elif r < 0.55:
            # smart add: insert x; if invalid, evict one blocker y
            x = int(rng.choice(out_set))
            new = cur.copy()
            new[x] = True
            if _is_valid(new):
                delta = 1
            else:
                found = False
                lst = in_set.tolist()
                rng.shuffle(lst)
                for y in lst[:40]:
                    new2 = new.copy()
                    new2[y] = False
                    if _is_valid(new2):
                        new = new2
                        delta = 0
                        found = True
                        break
                if not found:
                    new = None
        elif r < 0.80:
            # remove (to escape local optima)
            x = int(rng.choice(in_set))
            new = cur.copy()
            new[x] = False
            delta = -1
        else:
            # swap
            x = int(rng.choice(out_set))
            y = int(rng.choice(in_set))
            new = cur.copy()
            new[x] = True
            new[y] = False
            delta = 0
            if not _is_valid(new):
                new = None
        if new is None:
            continue
        if delta >= 0 or rng.random() < math.exp(-delta / T):
            cur = new
            cur_size += delta
            if cur_size > best_size:
                best_size = cur_size
                best = cur.copy()
        T = max(0.5, T * 0.9995)
    return best, best_size

def _fallback():
    m = np.zeros(N + 1, dtype=bool)
    m[N // 2 + 1:] = True
    return [int(v) for v in np.nonzero(m)[0]]

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 82
        best = None
        for seed in range(30):
            if time.time() > deadline - 15:
                break
            m = _initial(seed)
            sub = min(deadline - 15, time.time() + 25)
            m2, _ = _sa(m, sub, 1000 + seed)
            _greedy_fill(m2)
            if best is None or m2.sum() > best.sum():
                best = m2.copy()
        if best is None:
            best = _initial(0)
        _greedy_fill(best)
        res = [int(v) for v in np.nonzero(best)[0]]
        # sanity: ensure validity & types before returning
        chk = np.zeros(N + 1, dtype=bool)
        for v in res:
            chk[v] = True
        if not _is_valid(chk):
            return _fallback()
        return res
    except Exception:
        return _fallback()
