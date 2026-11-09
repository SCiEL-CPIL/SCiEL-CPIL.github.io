"""
Construct a linear triple system on 16 points with small independence number.

Strategy:
- Use the 4-bit vector space GF(2)^4 representation of points 0..15.
- Generate triples from affine lines (3 collinear points in a 1D subspace over GF(2))
  and additional structured triples to maximize coverage.
- Use a greedy/local-search approach to build a linear hypergraph (any two triples share at most one point)
  that minimizes the independence number.
- The independence number is the largest subset of points containing no triple entirely.

We use a time-bounded local search with a deadline guard.
"""

import time
import random
import itertools

DEADLINE_SECONDS = 110  # safety margin before 120s kill

def _elapsed(start_time):
    return time.time() - start_time

def _is_linear(triples):
    """Check that any two triples share at most one point."""
    n = len(triples)
    for i in range(n):
        si = set(triples[i])
        for j in range(i + 1, n):
            if len(si & set(triples[j])) > 1:
                return False
    return True

def _independence_number(triples, n=16):
    """
    Compute the independence number: largest subset of {0..n-1} containing no triple.
    A subset S is independent if no triple is fully contained in S.
    We check all subsets by size descending, but that's 2^16 = 65536 which is fine.
    """
    # Build a set of frozensets for fast lookup
    triple_set = set()
    for t in triples:
        triple_set.add(frozenset(t))
    
    # Try largest subsets first
    for size in range(n, 0, -1):
        for subset in itertools.combinations(range(n), size):
            s = set(subset)
            # Check if any triple is fully in s
            independent = True
            for t in triples:
                if t[0] in s and t[1] in s and t[2] in s:
                    independent = False
                    break
            if independent:
                return size
    return 0

def _count_covered(triples, n=16):
    """Count how many points appear in at least one triple."""
    covered = set()
    for t in triples:
        covered.update(t)
    return len(covered)

def _generate_affine_lines():
    """
    Generate all affine lines in GF(2)^4.
    An affine line is {a, a+v, a+2v} but in GF(2), 2v=0, so we need 3 distinct points.
    Actually in GF(2)^4, a 1D affine subspace has only 2 points: {a, a+v}.
    So affine lines of size 3 don't exist in GF(2)^4 in the usual sense.
    
    Instead, let's use a different algebraic structure.
    Use the Fano plane-like or use specific combinatorial designs.
    
    Alternative: use the 4x4 grid structure. Points are (i,j) for i,j in 0..3.
    Define triples as:
    - Rows: but rows have 4 points, not 3.
    
    Let me use a different approach: generate candidate triples from structured patterns
    and use greedy selection with local search.
    """
    # Let's just generate a pool of candidate triples
    candidates = []
    # All C(16,3) = 560 triples
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                candidates.append([a, b, c])
    return candidates

def _greedy_build(candidates, n=16, target_indep=None):
    """
    Greedily add triples that maintain linearity and reduce independence number.
    """
    triples = []
    # Track point-pair usage to ensure linearity
    pair_count = {}
    
    for t in candidates:
        # Check linearity: no pair already used in another triple
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        ok = True
        for p in pairs:
            key = (min(p), max(p))
            if key in pair_count:
                ok = False
                break
        if not ok:
            continue
        triples.append(t)
        for p in pairs:
            key = (min(p), max(p))
            pair_count[key] = pair_count.get(key, 0) + 1
    
    return triples

def _local_search(triples, candidates, n=16, start_time=None, max_iter=5000):
    """
    Local search: try swapping or adding/removing triples to reduce independence number.
    """
    if start_time is None:
        start_time = time.time()
    
    triples = list(triples)
    current_indep = _independence_number(triples, n)
    best_indep = current_indep
    best_triples = list(triples)
    
    # Track pair usage
    def build_pair_map(triples):
        pm = {}
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    key = (min(t[i], t[j]), max(t[i], t[j]))
                    pm[key] = pm.get(key, 0) + 1
        return pm
    
    pair_map = build_pair_map(triples)
    
    iter_count = 0
    while iter_count < max_iter:
        if _elapsed(start_time) > DEADLINE_SECONDS - 5:
            break
        iter_count += 1
        
        # Try adding a random triple
        if random.random() < 0.3 and len(triples) < 100:
            t = random.choice(candidates)
            # Check linearity
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            ok = True
            for p in pairs:
                key = (min(p), max(p))
                if key in pair_map:
                    ok = False
                    break
            if ok:
                new_triples = triples + [t]
                new_indep = _independence_number(new_triples, n)
                if new_indep < current_indep:
                    triples = new_triples
                    current_indep = new_indep
                    if current_indep < best_indep:
                        best_indep = current_indep
                        best_triples = list(triples)
                    pair_map = build_pair_map(triples)
        
        # Try removing a random triple
        elif random.random() < 0.2 and len(triples) > 5:
            idx = random.randint(0, len(triples) - 1)
            removed = triples[idx]
            new_triples = triples[:idx] + triples[idx+1:]
            new_indep = _independence_number(new_triples, n)
            if new_indep < current_indep:
                triples = new_triples
                current_indep = new_indep
                if current_indep < best_indep:
                    best_indep = current_indep
                    best_triples = list(triples)
                pair_map = build_pair_map(triples)
            # else keep removed out (don't add back)
        
        # Try swapping: remove one, add one
        else:
            if len(triples) > 0:
                idx = random.randint(0, len(triples) - 1)
                removed = triples[idx]
                # Try to find a replacement
                for _ in range(20):
                    t = random.choice(candidates)
                    if t == removed:
                        continue
                    # Check linearity with remaining triples
                    remaining = triples[:idx] + triples[idx+1:]
                    pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
                    ok = True
                    # Build pair map for remaining
                    rem_pm = build_pair_map(remaining)
                    for p in pairs:
                        key = (min(p), max(p))
                        if key in rem_pm:
                            ok = False
                            break
                    if not ok:
                        continue
                    new_triples = remaining + [t]
                    new_indep = _independence_number(new_triples, n)
                    if new_indep < current_indep:
                        triples = new_triples
                        current_indep = new_indep
                        if current_indep < best_indep:
                            best_indep = current_indep
                            best_triples = list(triples)
                        pair_map = build_pair_map(triples)
                        break
    
    return best_triples, best_indep

def construct():
    start_time = time.time()
    
    # Generate candidate triples
    candidates = _generate_affine_lines()
    random.shuffle(candidates)
    
    # Greedy build
    triples = _greedy_build(candidates)
    
    # Local search
    triples, indep = _local_search(triples, candidates, start_time=start_time, max_iter=10000)
    
    # Final check
    if not _is_linear(triples):
        # Fallback to simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return triples

# Quick self-test
if __name__ == "__main__":
    t = construct()
    print(f"Number of triples: {len(t)}")
    print(f"Linear: {_is_linear(t)}")
    print(f"Independence number: {_independence_number(t)}")
