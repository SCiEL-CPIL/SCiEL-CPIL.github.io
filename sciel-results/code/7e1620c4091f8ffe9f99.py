"""Linear triple systems on 16 points with small independence number.
Strategy: fast bitset-based alpha calculation, deterministic combinatorial seeds,
and local search with efficient updates to push alpha as low as possible (target 5)."""
import time
import random
from itertools import combinations

DEADLINE = 135.0  # safety margin under 150s kill

def calc_alpha_fast(triples):
    """Fast independence number using bitset superset marking.
    Returns the largest size of a subset with no triple fully contained."""
    n = 16
    # A set S is independent iff no triple is a subset of S.
    # Equivalent: S is NOT a superset of any triple.
    # We can mark all subsets that contain at least one triple (non-independent),
    # then find the largest unmarked subset.
    # But 2^16 = 65536, so we can do it directly.
    
    # For each triple, mark all supersets as non-independent.
    # Represent subsets as bitmasks.
    non_independent = [False] * (1 << n)
    
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        # All supersets of mask are non-independent
        # Iterate over all supersets of mask
        # A superset of mask has all bits of mask set, plus any subset of remaining bits
        remaining_mask = ((1 << n) - 1) ^ mask
        # Enumerate all subsets of remaining_mask and OR with mask
        sub = remaining_mask
        while True:
            non_independent[mask | sub] = True
            if sub == 0:
                break
            sub = (sub - 1) & remaining_mask
    
    # Find largest unmarked subset
    for size in range(n, 0, -1):
        # Check all subsets of this size
        for mask in range(1 << n):
            if mask.bit_count() == size and not non_independent[mask]:
                return size
    return 0

def calc_alpha_triples(triples, n=16):
    """Alternative: iterate subsets from large to small, check no triple fully inside."""
    triple_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in triples]
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            smask = 0
            for x in subset:
                smask |= (1 << x)
            ok = True
            for tm in triple_masks:
                if (smask & tm) == tm:
                    ok = False
                    break
            if ok:
                return size
    return 0

def is_linear(triples):
    """Check any two triples share at most one point."""
    pair_count = {}
    for t in triples:
        for i in range(3):
            for j in range(i + 1, 3):
                p = (min(t[i], t[j]), max(t[i], t[j]))
                pair_count[p] = pair_count.get(p, 0) + 1
                if pair_count[p] > 1:
                    return False
    return True

def build_sts4_like(n=16, seed=0):
    """Build a linear 3-graph using affine geometry over F4 x F4 (16 points).
    F4 = {0, 1, a, b} where a^2 = a + 1, b = a^2.
    Lines in AG(2,4): 21 lines of 4 points each.
    We can take triples from these lines or use a different construction.
    
    Alternative: use the known (16, 40, 4) design or similar.
    Let's use a randomized greedy with good heuristics."""
    rng = random.Random(seed)
    all_triples = list(combinations(range(n), 3))
    rng.shuffle(all_triples)
    
    used_pairs = set()
    triples = []
    
    # Greedy: pick triples that cover 3 unused pairs
    # Use a priority: prefer triples with more "rare" pairs
    for t in all_triples:
        p1 = (min(t[0], t[1]), max(t[0], t[1]))
        p2 = (min(t[0], t[2]), max(t[0], t[2]))
        p3 = (min(t[1], t[2]), max(t[1], t[2]))
        if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            triples.append(list(t))
    
    return triples

def local_search(triples, time_limit=20.0, seed=0):
    """Local search to reduce independence number.
    Operations: add triple (if linear), remove triple, swap."""
    rng = random.Random(seed)
    start = time.time()
    n = 16
    
    current_triples = [list(t) for t in triples]
    current_alpha = calc_alpha_fast(current_triples)
    best_triples = [list(t) for t in current_triples]
    best_alpha = current_alpha
    
    # Pair-to-count map for fast linearity check
    def build_pair_map(trs):
        m = {}
        for t in trs:
            for i in range(3):
                for j in range(i + 1, 3):
                    p = (min(t[i], t[j]), max(t[i], t[j]))
                    m[p] = m.get(p, 0) + 1
        return m
    
    pm = build_pair_map(current_triples)
    all_triples = list(combinations(range(n), 3))
    
    iteration = 0
    while time.time() - start < time_limit:
        iteration += 1
        op = rng.random()
        
        if op < 0.4:
            # Try to add a random triple
            t = rng.choice(all_triples)
            p1 = (min(t[0], t[1]), max(t[0], t[1]))
            p2 = (min(t[0], t[2]), max(t[0], t[2]))
            p3 = (min(t[1], t[2]), max(t[1], t[2]))
            if pm.get(p1, 0) == 0 and pm.get(p2, 0) == 0 and pm.get(p3, 0) == 0:
                current_triples.append(list(t))
                pm[p1] = pm.get(p1, 0) + 1
                pm[p2] = pm.get(p2, 0) + 1
                pm[p3] = pm.get(p3, 0) + 1
                new_alpha = calc_alpha_fast(current_triples)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = [list(x) for x in current_triples]
        elif op < 0.7:
            # Try to remove a random triple
            if current_triples:
                idx = rng.randrange(len(current_triples))
                removed = current_triples.pop(idx)
                for i in range(3):
                    for j in range(i + 1, 3):
                        p = (min(removed[i], removed[j]), max(removed[i], removed[j]))
                        pm[p] = pm.get(p, 0) - 1
                        if pm[p] <= 0:
                            del pm[p]
                new_alpha = calc_alpha_fast(current_triples)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = [list(x) for x in current_triples]
        else:
            # Swap: remove one, add another
            if current_triples:
                idx = rng.randrange(len(current_triples))
                removed = current_triples.pop(idx)
                for i in range(3):
                    for j in range(i + 1, 3):
                        p = (min(removed[i], removed[j]), max(removed[i], removed[j]))
                        pm[p] = pm.get(p, 0) - 1
                        if pm[p] <= 0:
                            del pm[p]
                # Try to add a different triple
                added = False
                for _ in range(10):
                    t2 = rng.choice(all_triples)
                    q1 = (min(t2[0], t2[1]), max(t2[0], t2[1]))
                    q2 = (min(t2[0], t2[2]), max(t2[0], t2[2]))
                    q3 = (min(t2[1], t2[2]), max(t2[1], t2[2]))
                    if pm.get(q1, 0) == 0 and pm.get(q2, 0) == 0 and pm.get(q3, 0) == 0:
                        current_triples.append(list(t2))
                        pm[q1] = pm.get(q1, 0) + 1
                        pm[q2] = pm.get(q2, 0) + 1
                        pm[q3] = pm.get(q3, 0) + 1
                        added = True
                        break
                if added:
                    new_alpha = calc_alpha_fast(current_triples)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = [list(x) for x in current_triples]
    
    return best_triples, best_alpha

def construct():
    start = time.time()
    
    # Phase 1: Build dense linear graphs via randomized greedy
    best_triples = None
    best_alpha = 17
    
    for seed in range(200):
        if time.time() - start > DEADLINE - 40:
            break
        triples = build_sts4_like(n=16, seed=seed)
        if triples:
            a = calc_alpha_fast(triples)
            if a < best_alpha:
                best_alpha = a
                best_triples = triples
                if best_alpha <= 5:
                    break
    
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    # Phase 2: Local search to reduce alpha
    remaining = DEADLINE - (time.time() - start) - 5
    if remaining > 10:
        for seed in range(50):
            if time.time() - start > DEADLINE - 5:
                break
            lt, la = local_search(best_triples, time_limit=max(3, remaining / 50 - 1), seed=seed)
            if la < best_alpha:
                best_alpha = la
                best_triples = lt
                if best_alpha <= 5:
                    break
    
    # Verify linearity
    if not is_linear(best_triples):
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
