"""Constructive algebraic design for linear triple systems on 16 points."""
import time
import random
from itertools import combinations

# Hard wall-clock safety margin (evaluator kills at 150s; keep margin).
_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)


def _is_valid(triples):
    """Any two distinct triples share at most one point."""
    for i in range(len(triples)):
        a = set(triples[i])
        for j in range(i + 1, len(triples)):
            if len(a & set(triples[j])) > 1:
                return False
    return True


def _independence_number(triples):
    """Largest subset of {0..15} containing no triple (as a set)."""
    n = 16
    triple_sets = [frozenset(t) for t in triples]
    best = 0
    # Brute force over subsets is 2^16 = 65536, cheap.
    # A subset S is independent iff it contains no triple set.
    # We want max |S| with no triple fully contained.
    # Equivalently, complement of a vertex cover of the 3-uniform hypergraph.
    # Use inclusion over subsets: check all 2^16 subsets for independence.
    # Represent independent test: S independent iff for all T in triple_sets, T not subset S.
    for mask in range(1 << n):
        size = bin(mask).count("1")
        if size <= best:
            continue
        ok = True
        for T in triple_sets:
            # T subset of S?
            tm = 0
            for x in T:
                tm |= (1 << x)
            if (mask & tm) == tm:
                ok = False
                break
        if ok:
            best = size
    return best


def _greedy_cover(triples, seed):
    """Greedy: build a valid linear system maximizing coverage of points."""
    rng = random.Random(seed)
    order = list(range(len(triples)))
    rng.shuffle(order)
    chosen = []
    chosen_sets = []
    used = [0] * 16
    for idx in order:
        t = triples[idx]
        ts = set(t)
        ok = True
        for cs in chosen_sets:
            if len(ts & cs) > 1:
                ok = False
                break
        if ok:
            chosen.append(list(t))
            chosen_sets.append(ts)
            for x in ts:
                used[x] += 1
    return chosen


def _score_of(triples):
    if not _is_valid(triples):
        return None
    return _independence_number(triples)


def _candidate_triples():
    """Generate a rich pool of candidate triples from algebraic structure."""
    tris = set()
    # All triples of the form {a, b, a+b} in GF(2)^4 (affine lines through 0 shifted).
    # In GF(2)^4, a+b is the third point of the line {a, b, a+b} since a+b+(a+b)=0? No.
    # Actually the 2-point line is {a, a+v}. For 3-point we use "arithmetic" in Z16 or conics.
    # Use Z16 arithmetic progressions: {a, a+d, a+2d} mod 16, d != 0, a+2d != a.
    for a in range(16):
        for d in range(1, 16):
            t = (a, (a + d) % 16, (a + 2 * d) % 16)
            if len(set(t)) == 3:
                tris.add(tuple(sorted(t)))
    # GF(2)^4 "conic" type: fix basis, take {x, x+e_i, x+e_j} patterns via XOR.
    # Represent points as 4-bit ints. For each pair of distinct nonzero vectors u,v,
    # and each point x, consider {x, x^u, x^v} if all distinct.
    for x in range(16):
        for u in range(1, 16):
            for v in range(u + 1, 16):
                a, b, c = x, x ^ u, x ^ v
                if len({a, b, c}) == 3:
                    tris.add(tuple(sorted((a, b, c))))
    return list(tris)


def _refine(triples, seed, budget):
    """Local search: try swapping one triple out for another to reduce independence number."""
    rng = random.Random(seed)
    cur = list(triples)
    cur_score = _score_of(cur)
    if cur_score is None:
        return triples, None
    best = cur
    best_score = cur_score
    pool = _candidate_triples()
    pool_sets = [set(t) for t in pool]
    while budget() > 0.5:
        improved = False
        for i in range(len(cur)):
            # try removing cur[i] and adding a random pool triple
            for _ in range(20):
                cand = rng.choice(pool)
                if cand in [tuple(t) for t in cur]:
                    continue
                trial = [t for j, t in enumerate(cur) if j != i]
                cset = set(cand)
                ok = True
                for t in trial:
                    if len(cset & set(t)) > 1:
                        ok = False
                        break
                if not ok:
                    continue
                trial.append(list(cand))
                s = _score_of(trial)
                if s is not None and s < best_score:
                    best = trial
                    best_score = s
                    cur = trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_score


def construct():
    start = time.time()
    deadline = 145.0
    pool = _candidate_triples()
    print("pool size:", len(pool))

    best_triples = None
    best_score = None

    # Phase 1: greedy coverage with several seeds.
    for seed in range(200):
        if time.time() - start > deadline - 5:
            break
        cand = _greedy_cover(pool, seed)
        s = _score_of(cand)
        if s is None:
            continue
        if best_score is None or s < best_score:
            best_score = s
            best_triples = cand
            print("seed", seed, "score", s, "n", len(cand))

    # Phase 2: local refinement on the best.
    if best_triples is not None:
        def budget():
            return deadline - (time.time() - start)
        refined, rs = _refine(best_triples, 12345, budget)
        if rs is not None and (best_score is None or rs < best_score):
            best_score = rs
            best_triples = refined
            print("refined score", rs)

    if best_triples is None:
        # Fallback: trivial valid system.
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return best_triples


if __name__ == "__main__":
    res = construct()
    print("valid:", _is_valid(res))
    print("independence number:", _independence_number(res))
