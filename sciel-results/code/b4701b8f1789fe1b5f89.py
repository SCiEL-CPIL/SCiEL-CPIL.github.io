"""CSP-style backtracking with forward checking to build a linear 3-graph on 16 points
with small independence number, then local-search to reduce alpha."""
import time
import random
from itertools import combinations

DEADLINE = 140.0  # safety margin under 150s kill

def calc_alpha(triples):
    """Independence number: largest set of points containing no triple."""
    n = 16
    # For each triple, it's "hit" if all 3 points are in the set.
    # We want largest set S with no triple fully inside S.
    # Equivalent: complement of a hitting set of all triples.
    # Brute force over subsets from large to small.
    all_points = set(range(n))
    triple_sets = [tuple(sorted(t)) for t in triples]
    # Try sizes from n down
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            s = set(subset)
            ok = True
            for t in triple_sets:
                if t[0] in s and t[1] in s and t[2] in s:
                    ok = False
                    break
            if ok:
                return size
    return 0

def is_linear(triples):
    """Check any two triples share at most one point."""
    for i in range(len(triples)):
        si = set(triples[i])
        for j in range(i + 1, len(triples)):
            sj = set(triples[j])
            if len(si & sj) > 1:
                return False
    return True

def max_triples_bound(n=16):
    """Max number of triples in a linear 3-graph on n points.
    Each pair of points can be in at most one triple.
    Number of pairs = C(n,2) = 120. Each triple uses 3 pairs.
    So at most 120/3 = 40 triples."""
    return n * (n - 1) // 6

def build_linear_graph_backtrack(seed=0, time_limit=10.0):
    """Backtracking: add triples one at a time, maintaining linearity.
    Order triples to cover many pairs, then check independence number."""
    rng = random.Random(seed)
    start = time.time()
    n = 16
    all_pairs = list(combinations(range(n), 2))
    # Shuffle for variety
    rng.shuffle(all_pairs)
    
    # State: set of used pairs, list of triples
    used_pairs = set()
    triples = []
    
    # Precompute: for each pair, which triples could contain it
    # We'll greedily pick triples that cover 3 unused pairs
    # But to be systematic, use backtracking with a heuristic
    
    # Heuristic: pick triple that maximizes new pairs covered
    # Represent as: for each pair, count how many times it's in a candidate triple
    # Simpler: iterative greedy with randomization + backtracking on stuck
    
    # Actually, let's do a randomized greedy with multiple restarts
    # and pick the one with best alpha
    best_triples = None
    best_alpha = 17
    
    for attempt in range(200):
        if time.time() - start > time_limit:
            break
        rng2 = random.Random(seed * 1000 + attempt)
        used_pairs2 = set()
        triples2 = []
        
        # Build candidate triples: all C(16,3) = 560
        all_triples = list(combinations(range(n), 3))
        rng2.shuffle(all_triples)
        
        # Greedy: pick triple if all 3 pairs are unused
        # Prioritize triples that cover "rare" pairs
        # Track pair frequency in remaining candidates
        for t in all_triples:
            p1, p2, p3 = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
            if p1 not in used_pairs2 and p2 not in used_pairs2 and p3 not in used_pairs2:
                used_pairs2.add(p1)
                used_pairs2.add(p2)
                used_pairs2.add(p3)
                triples2.append(list(t))
        
        if len(triples2) > len(triples) if best_triples is None else len(triples2) >= len(triples):
            triples = triples2
            used_pairs = used_pairs2
            if len(triples2) >= max_triples_bound(n):
                break
    
    return triples

def local_search_reduce_alpha(triples, time_limit=20.0, seed=0):
    """Try to reduce independence number by modifying the graph.
    Operations: add a triple (if linear), remove a triple, swap triples."""
    rng = random.Random(seed)
    start = time.time()
    n = 16
    
    # Current state
    current_triples = [list(t) for t in triples]
    current_alpha = calc_alpha(current_triples)
    best_triples = [list(t) for t in current_triples]
    best_alpha = current_alpha
    
    # Build pair-to-triple map for fast linearity check
    def pair_map(trs):
        m = {}
        for t in trs:
            for i in range(3):
                for j in range(i+1, 3):
                    p = (min(t[i], t[j]), max(t[i], t[j]))
                    m[p] = m.get(p, 0) + 1
        return m
    
    pm = pair_map(current_triples)
    
    all_triples = list(combinations(range(n), 3))
    
    for iteration in range(10000):
        if time.time() - start > time_limit:
            break
        # Try to add a random triple that maintains linearity
        t = rng.choice(all_triples)
        p1, p2, p3 = (min(t[0], t[1]), max(t[0], t[1])), (min(t[0], t[2]), max(t[0], t[2])), (min(t[1], t[2]), max(t[1], t[2]))
        if pm.get(p1, 0) == 0 and pm.get(p2, 0) == 0 and pm.get(p3, 0) == 0:
            current_triples.append(list(t))
            pm[p1] = pm.get(p1, 0) + 1
            pm[p2] = pm.get(p2, 0) + 1
            pm[p3] = pm.get(p3, 0) + 1
            new_alpha = calc_alpha(current_triples)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = [list(x) for x in current_triples]
            # Keep the addition (it doesn't hurt linearity, may help alpha)
        else:
            # Try removing a random triple to free up pairs, then re-add differently
            if current_triples:
                idx = rng.randrange(len(current_triples))
                removed = current_triples.pop(idx)
                for i in range(3):
                    for j in range(i+1, 3):
                        p = (min(removed[i], removed[j]), max(removed[i], removed[j]))
                        pm[p] = pm.get(p, 0) - 1
                        if pm[p] <= 0:
                            del pm[p]
                # Try to add a different triple
                for _ in range(5):
                    t2 = rng.choice(all_triples)
                    q1, q2, q3 = (min(t2[0], t2[1]), max(t2[0], t2[1])), (min(t2[0], t2[2]), max(t2[0], t2[2])), (min(t2[1], t2[2]), max(t2[1], t2[2]))
                    if pm.get(q1, 0) == 0 and pm.get(q2, 0) == 0 and pm.get(q3, 0) == 0:
                        current_triples.append(list(t2))
                        pm[q1] = pm.get(q1, 0) + 1
                        pm[q2] = pm.get(q2, 0) + 1
                        pm[q3] = pm.get(q3, 0) + 1
                        break
    
    return best_triples, best_alpha

def construct():
    start = time.time()
    
    # Phase 1: Build a dense linear graph via randomized greedy
    best_triples = None
    best_alpha = 17
    
    for seed in range(50):
        if time.time() - start > DEADLINE - 30:
            break
        triples = build_linear_graph_backtrack(seed=seed, time_limit=1.5)
        if triples:
            a = calc_alpha(triples)
            if a < best_alpha:
                best_alpha = a
                best_triples = triples
    
    if best_triples is None:
        # Fallback: seed program
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Phase 2: Local search to reduce alpha
    remaining = DEADLINE - (time.time() - start) - 5
    if remaining > 10:
        for seed in range(20):
            if time.time() - start > DEADLINE - 5:
                break
            lt, la = local_search_reduce_alpha(best_triples, time_limit=max(5, remaining / 20 - 2), seed=seed)
            if la < best_alpha:
                best_alpha = la
                best_triples = lt
    
    # Verify linearity
    if not is_linear(best_triples):
        # Fallback
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
