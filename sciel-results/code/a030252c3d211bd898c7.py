"""Size-efficient sorting network on 17 wires: prune + repair search."""
import time
import numpy as np

N = 17


def _build_seed():
    # Odd-even transposition network (136 comparators).
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _verify_fast(comp):
    """Zero-one principle over all 2^N Boolean inputs. Returns True iff sorted."""
    m = len(comp)
    if m == 0:
        return True
    a = np.fromiter((i for i, _ in comp), dtype=np.int32, count=m)
    b = np.fromiter((j for _, j in comp), dtype=np.int32, count=m)
    total = 1 << N
    last = np.arange(total, dtype=np.int32)
    for k in range(m):
        ai = a[k]
        bi = b[k]
        va = (last >> ai) & 1
        vb = (last >> bi) & 1
        mn = va & vb
        mx = va | vb
        last = np.where(va != vb, last ^ (1 << ai) ^ (1 << bi), last)
        del va, vb, mn, mx
    return bool((last == np.arange(total, dtype=np.int32)).all())


def _verify_slow(comp):
    for mask in range(1 << N):
        v = list(mask & 1 for _ in range(N))
        for i, j in comp:
            if v[i] > v[j]:
                v[i], v[j] = v[j], v[i]
        if v != sorted(v):
            return False
    return True


def _greedy_insert_and_prune(comp, deadline):
    comp = list(comp)
    cur_len = len(comp)
    # Repair: insert missing comparators in a structured order.
    order = ([(i, i + 1) for i in range(N - 1)] * N +
             [(i, i + 2) for i in range(N - 2)] * N +
             [(i, i + 3) for i in range(N - 3)] * N +
             [(i, j) for j in range(N) for i in range(j)])
    for c in order:
        if time.time() > deadline:
            break
        if _verify_fast(comp):
            break
        comp.append(c)
    # Prune: remove redundant comparators.
    progress = True
    while progress and time.time() < deadline:
        progress = False
        for idx in range(len(comp) - 1, -1, -1):
            if time.time() > deadline:
                break
            if len(comp) <= cur_len:
                break
            cand = comp[:idx] + comp[idx + 1:]
            if _verify_fast(cand):
                comp = cand
                cur_len = len(comp)
                progress = True
    return comp


def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill
    comp = _build_seed()
    # Sanity: seed must sort.
    if not _verify_fast(comp):
        return comp
    # Phase 1: prune redundant comparators from the seed.
    comp = _greedy_insert_and_prune(comp, deadline)
    # Phase 2: if time remains, attempt further pruning on a shuffled copy
    # to escape local minima.
    if time.time() < deadline - 5.0:
        import random
        rnd = random.Random(12345)
        best = comp
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            trial = list(best)
            rnd.shuffle(trial)
            trial = _greedy_insert_and_prune(trial, deadline - 5.0)
            if len(trial) < len(best) and _verify_fast(trial):
                best = trial
        comp = best
    return comp
