import time
import random
from itertools import combinations

DEADLINE_MARGIN = 8.0  # seconds before the 150s wall-clock kill

def _now():
    return time.time()

def _independence_number(triples, points=16):
    # exact: largest S with no triple fully contained in S.
    # SOS DP over 2^16.
    n = points
    size = 1 << n
    bad = bytearray(size)
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        bad[m] = 1
    # propagate: if any triple contained in S, mark S bad
    for i in range(n):
        bit = 1 << i
        for S in range(size):
            if S & bit:
                if bad[S ^ bit]:
                    bad[S] = 1
    best = 0
    for S in range(size):
        if not bad[S]:
            pc = S.bit_count()
            if pc > best:
                best = pc
    return best

def _greedy_maximal_independent_set(triples, points=16, rng=None):
    n = points
    avail = list(range(n))
    if rng is not None:
        rng.shuffle(avail)
    chosen = []
    chosen_set = set()
    for p in avail:
        ok = True
        for t in triples:
            if all(x in chosen_set or x == p for x in t):
                ok = False
                break
        if ok:
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _random_greedy_packing(rng, points=16):
    n = points
    triples = []
    all_triples = list(combinations(range(n), 3))
    rng.shuffle(all_triples)
    used_pairs = set()
    for t in all_triples:
        pairs = list(combinations(t, 2))
        if any(p in used_pairs for p in pairs):
            continue
        for p in pairs:
            used_pairs.add(p)
        triples.append(list(t))
    return triples

def _try_add_triple_breaking_iset(triples, iset, rng, points=16):
    n = points
    iset_set = set(iset)
    cands = []
    for a in iset:
        for b in range(n):
            if b == a:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                cands.append((a, b, c))
    rng.shuffle(cands)
    seen = set()
    for t in cands:
        if t in seen:
            continue
        seen.add(t)
        ok = True
        for ex in triples:
            if len(set(t) & set(ex)) > 1:
                ok = False
                break
        if ok:
            return list(t)
    return None

def construct():
    start = _now()
    deadline = start + 150.0 - DEADLINE_MARGIN
    rng = random.Random(12345)

    n = 16
    best_triples = None
    best_alpha = n + 1

    def consider(triples):
        nonlocal best_triples, best_alpha
        a = _independence_number(triples, n)
        if a < best_alpha:
            best_alpha = a
            best_triples = [list(t) for t in triples]

    # Phase 1: many random greedy packings
    attempts = 0
    while _now() < deadline and attempts < 300:
        attempts += 1
        triples = _random_greedy_packing(rng, n)
        consider(triples)
        # local search: try to break independent sets
        for _ in range(150):
            if _now() > deadline:
                break
            iset = _greedy_maximal_independent_set(triples, n, rng)
            if len(iset) <= best_alpha:
                continue
            new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
            if new_t is not None:
                triples.append(new_t)
                consider(triples)
            else:
                # swap: remove a random triple, then try to add
                if triples:
                    idx = rng.randrange(len(triples))
                    removed = triples.pop(idx)
                    new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
                    if new_t is not None:
                        triples.append(new_t)
                        consider(triples)
                    else:
                        triples.append(removed)
                break

    # Phase 2: targeted improvement on best
    if best_triples is not None and _now() < deadline:
        triples = [list(t) for t in best_triples]
        for _ in range(800):
            if _now() > deadline:
                break
            iset = _greedy_maximal_independent_set(triples, n, rng)
            if len(iset) <= best_alpha:
                break
            new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
            if new_t is not None:
                triples.append(new_t)
                consider(triples)
            else:
                improved = False
                for _ in range(15):
                    if not triples:
                        break
                    idx = rng.randrange(len(triples))
                    removed = triples.pop(idx)
                    new_t = _try_add_triple_breaking_iset(triples, iset, rng, n)
                    if new_t is not None:
                        triples.append(new_t)
                        consider(triples)
                        improved = True
                        break
                    else:
                        triples.append(removed)
                if not improved:
                    break

    if best_triples is None:
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    return best_triples
