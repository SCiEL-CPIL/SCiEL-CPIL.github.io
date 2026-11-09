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
    # Precompute bitmask for each triple
    triple_masks = []
    for t in triples:
        m = 0
        for x in t:
            m |= (1 << x)
        triple_masks.append(m)
    
    best = 0
    for mask in range(1 << n):
        size = bin(mask).count("1")
        if size <= best:
            continue
        ok = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                ok = False
                break
        if ok:
            best = size
    return best


def _independence_number_fast(triples):
    """Faster independence number using bitmask and early termination."""
    n = 16
    # Precompute bitmask for each triple
    triple_masks = []
    for t in triples:
        m = 0
        for x in t:
            m |= (1 << x)
        triple_masks.append(m)
    
    # Sort triples by mask to potentially find conflicts faster
    triple_masks.sort(key=lambda m: bin(m).count("1"))
    
    best = 0
    # Check larger subsets first (from 15 down to 0)
    for size in range(n, best, -1):
        # Check all subsets of this size
        for mask in range(1 << n):
            if bin(mask).count("1") != size:
                continue
            ok = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    ok = False
                    break
            if ok:
                return size
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


def _algebraic_design():
    """
    Construct a structured design using GF(2)^4 geometry.
    
    In GF(2)^4, we can create triples that form affine planes or use
    quadratic forms to ensure good coverage properties.
    
    Strategy: Use triples from parallel classes of affine lines, ensuring
    that the hypergraph is as uniform as possible.
    """
    triples = []
    
    # GF(2)^4 has 16 points. We can view them as 4-bit vectors.
    # An affine line in GF(2)^4 has exactly 2 points (since char 2),
    # so we need a different approach for 3-uniform hypergraphs.
    
    # Instead, use the fact that any 3 points in GF(2)^4 either form
    # an affine line (if one is sum of other two) or not.
    
    # Create triples using a structured approach:
    # For each "direction" (nonzero vector v), and each "base" point,
    # create triples {x, x+v, x+w} where w is another direction.
    
    # Better: Use a near-field or quadratic construction.
    
    # Approach: Partition 16 points into 4 groups of 4, then create
    # triples that cross groups in a balanced way.
    
    # Group 1: 0,1,2,3
    # Group 2: 4,5,6,7
    # Group 3: 8,9,10,11
    # Group 4: 12,13,14,15
    
    # Create triples with one point from each of 3 groups
    groups = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # For each combination of 3 groups, create balanced triples
    for g1_idx, g2_idx, g3_idx in combinations(range(4), 3):
        g1, g2, g3 = groups[g1_idx], groups[g2_idx], groups[g3_idx]
        # Create 4 triples, each using different elements
        for i in range(4):
            a = g1[i % 4]
            b = g2[(i + 1) % 4]
            c = g3[(i + 2) % 4]
            t = tuple(sorted((a, b, c)))
            if len(set(t)) == 3:
                triples.append(list(t))
    
    # Add some within-group triples for additional coverage
    for g in groups:
        for i in range(4):
            t = tuple(sorted((g[i], g[(i+1) % 4], g[(i+2) % 4])))
            if len(set(t)) == 3:
                triples.append(list(t))
    
    # Deduplicate
    seen = set()
    unique_triples = []
    for t in triples:
        key = tuple(sorted(t))
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)
    
    return unique_triples


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

    # Phase 0: Try the algebraic design first
    alg_triples = _algebraic_design()
    if alg_triples:
        s = _score_of(alg_triples)
        if s is not None:
            best_score = s
            best_triples = alg_triples
            print("algebraic design score:", s, "n:", len(alg_triples))

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
