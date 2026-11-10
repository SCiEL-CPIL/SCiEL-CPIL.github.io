"""Linear triple systems on 16 points with small independence number."""
import random
import time
import itertools

N = 16
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
NT = len(ALL_TRIPLES)
ALL_PAIRS = list(itertools.combinations(range(N), 2))
PAIR_TO_IDX = {p: i for i, p in enumerate(ALL_PAIRS)}
# precompute, for each triple, the list of its 3 pair-indices
TRIPLE_PAIRS = [
    [PAIR_TO_IDX[(a, b)], PAIR_TO_IDX[(a, c)], PAIR_TO_IDX[(b, c)]]
    for (a, b, c) in ALL_TRIPLES
]
# point masks
POINT_MASK = [1 << i for i in range(N)]
TRIPLE_MASK = [ (1 << a) | (1 << b) | (1 << c) for (a, b, c) in ALL_TRIPLES ]
FULL_MASK = (1 << N) - 1


def _indep_number(triples):
    """Exact size of the largest subset of points containing no triple.
    triples: list of (a,b,c). Returns int in [0, N].
    Uses bitmask + recursive branch-and-bound over points."""
    if not triples:
        return N
    masks = [TRIPLE_MASK[ALL_TRIPLES.index(t)] if t in _TRIPLE_SET else _mask_of(t) for t in triples]
    # dedupe masks
    masks = list(set(masks))
    m = len(masks)
    # point -> which masks contain it (for pruning)
    point_masks = [[] for _ in range(N)]
    for i, mk in enumerate(masks):
        for p in range(N):
            if mk & (1 << p):
                point_masks[p].append(i)

    best = [0]

    def dfs(pos, chosen_mask, count, remaining_masks):
        # remaining_masks: set of mask-indices still fully available (subset of chosen_mask)
        if count + (N - pos) <= best[0]:
            return
        if not remaining_masks:
            if count > best[0]:
                best[0] = count
            return
        if pos == N:
            if count > best[0]:
                best[0] = count
            return
        # branch: exclude point pos
        dfs(pos + 1, chosen_mask, count, remaining_masks)
        # branch: include point pos (only if no mask becomes fully satisfied)
        pm = point_masks[pos]
        new_rem = None
        ok = True
        for mi in pm:
            if mi in remaining_masks:
                # check if all points of mask mi are now in chosen
                mk = masks[mi]
                if (chosen_mask | (1 << pos)) & mk == mk:
                    ok = False
                    break
        if ok:
            if new_rem is None:
                new_rem = set(remaining_masks)
            dfs(pos + 1, chosen_mask | (1 << pos), count + 1, new_rem)

    dfs(0, 0, 0, set(range(m)))
    return best[0]


# cache masks by triple tuple for speed
_TRIPLE_SET = set(ALL_TRIPLES)
_MASK_CACHE = {}
def _mask_of(t):
    if t not in _MASK_CACHE:
        _MASK_CACHE[t] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    return _MASK_CACHE[t]


def _indep_number_fast(triples):
    if not triples:
        return N
    masks = []
    seen = set()
    for t in triples:
        mk = _mask_of(t)
        if mk not in seen:
            seen.add(mk)
            masks.append(mk)
    m = len(masks)
    point_masks = [[] for _ in range(N)]
    for i, mk in enumerate(masks):
        for p in range(N):
            if mk & (1 << p):
                point_masks[p].append(i)
    best = [0]

    def dfs(pos, chosen_mask, count, remaining_masks):
        if count + (N - pos) <= best[0]:
            return
        if pos == N:
            if count > best[0]:
                best[0] = count
            return
        # exclude
        dfs(pos + 1, chosen_mask, count, remaining_masks)
        # include
        ok = True
        for mi in point_masks[pos]:
            if mi in remaining_masks:
                if (chosen_mask | (1 << pos)) & masks[mi] == masks[mi]:
                    ok = False
                    break
        if ok:
            dfs(pos + 1, chosen_mask | (1 << pos), count + 1, remaining_masks)

    dfs(0, 0, 0, set(range(m)))
    return best[0]


def _find_max_iset(triples):
    """Return one maximum independent set (as a set of points)."""
    if not triples:
        return set(range(N))
    masks = []
    seen = set()
    for t in triples:
        mk = _mask_of(t)
        if mk not in seen:
            seen.add(mk)
            masks.append(mk)
    m = len(masks)
    point_masks = [[] for _ in range(N)]
    for i, mk in enumerate(masks):
        for p in range(N):
            if mk & (1 << p):
                point_masks[p].append(i)
    best_set = [set()]

    def dfs(pos, chosen, chosen_mask, remaining_masks, count):
        if count + (N - pos) <= len(best_set[0]):
            return
        if pos == N:
            if count > len(best_set[0]):
                best_set[0] = set(chosen)
            return
        dfs(pos + 1, chosen, chosen_mask, remaining_masks, count)
        ok = True
        for mi in point_masks[pos]:
            if mi in remaining_masks:
                if (chosen_mask | (1 << pos)) & masks[mi] == masks[mi]:
                    ok = False
                    break
        if ok:
            chosen.append(pos)
            dfs(pos + 1, chosen, chosen_mask | (1 << pos), count + 1, remaining_masks)
            chosen.pop()

    dfs(0, [], 0, set(range(m)), 0)
    return best_set[0]


def _is_linear(triples):
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            if len(set(triples[i]) & set(triples[j])) > 1:
                return False
    return True


def _greedy_packing(seed_offset):
    rng = random.Random(1000 + seed_offset)
    used_pairs = set()
    triples = []
    # try to build a near-maximal packing: repeatedly add a triple using 3 unused pairs
    # choose triple that uses "new" pairs and balances point degrees
    point_deg = [0] * N
    remaining = list(range(NT))
    rng.shuffle(remaining)
    for idx in remaining:
        t = ALL_TRIPLES[idx]
        pidx = TRIPLE_PAIRS[idx]
        if any(p in used_pairs for p in pidx):
            continue
        # accept
        for p in pidx:
            used_pairs.add(p)
        triples.append(t)
        for x in t:
            point_deg[x] += 1
    return triples


def _try_add_breaking(triples, iset, rng, used_pairs, used_pairs_set):
    """Try to add a triple that is a subset of iset and is linear. Return (new_triple, ok)."""
    cands = list(itertools.combinations(sorted(iset), 3))
    rng.shuffle(cands)
    for t in cands:
        pidx = [PAIR_TO_IDX[(t[0], t[1])], PAIR_TO_IDX[(t[0], t[2])], PAIR_TO_IDX[(t[1], t[2])]]
        if any(p in used_pairs for p in pidx):
            continue
        # linear check vs existing (pair check suffices)
        ok = True
        for j, et in enumerate(triples):
            if len(set(t) & set(et)) > 1:
                ok = False
                break
        if ok:
            return t, True
    return None, False


def _local_search(triples, deadline, rng):
    best = triples[:]
    best_score = _indep_number_fast(best)
    cur = triples[:]
    cur_score = best_score

    while time.time() < deadline - 1.0:
        # find a maximum independent set
        iset = _find_max_iset(cur)
        if len(iset) < cur_score:
            # recompute
            cur_score = _indep_number_fast(cur)
        # try to add a triple inside iset
        used_pairs = set()
        for t in cur:
            used_pairs.add(PAIR_TO_IDX[(t[0], t[1])])
            used_pairs.add(PAIR_TO_IDX[(t[0], t[2])])
            used_pairs.add(PAIR_TO_IDX[(t[1], t[2])])
        nt, ok = _try_add_breaking(cur, iset, rng, used_pairs, used_pairs)
        if ok:
            cur.append(nt)
            ns = _indep_number_fast(cur)
            if ns < cur_score:
                cur_score = ns
                if ns < best_score:
                    best_score = ns
                    best = cur[:]
            else:
                cur.pop()
        # occasionally try a pair-swap: remove one triple, try to add a different one
        if len(cur) > 3 and rng.random() < 0.5:
            idx = rng.randrange(len(cur))
            old = cur.pop(idx)
            # try to add a random linear triple
            improved = False
            for _ in range(15):
                cand = tuple(sorted(rng.sample(range(N), 3)))
                ok = True
                for et in cur:
                    if len(set(cand) & set(et)) > 1:
                        ok = False
                        break
                if ok:
                    cur.append(cand)
                    ns = _indep_number_fast(cur)
                    if ns < cur_score:
                        cur_score = ns
                        if ns < best_score:
                            best_score = ns
                            best = cur[:]
                        improved = True
                    else:
                        cur.pop()
                    break
            if not improved:
                cur.append(old)
    return best, best_score


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill
    rng = random.Random(7)

    best_triples = None
    best_score = N + 1

    # Phase 1: greedy packings with several seeds
    for s in range(20):
        if time.time() > deadline - 25:
            break
        tr = _greedy_packing(s)
        sc = _indep_number_fast(tr)
        if sc < best_score:
            best_score = sc
            best_triples = tr[:]

    if best_triples is None:
        best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        best_score = _indep_number_fast(best_triples)

    # Phase 2: local search from the best seed
    if time.time() < deadline - 10:
        best_triples, best_score = _local_search(best_triples, deadline, rng)

    # Phase 3: a few more local searches from other good seeds
    for s in range(1, 8):
        if time.time() > deadline - 15:
            break
        tr = _greedy_packing(s)
        sc = _indep_number_fast(tr)
        if sc >= best_score:
            continue
        tr, sc = _local_search(tr, deadline, rng)
        if sc < best_score:
            best_score = sc
            best_triples = tr[:]

    if not _is_linear(best_triples):
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return [list(t) for t in best_triples]
