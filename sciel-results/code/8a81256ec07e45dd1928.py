"""Linear triple systems on 16 points with small independence number."""
import time
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START

def _validate(triples):
    """Check that all triples are valid and the system is linear."""
    if not triples:
        return True
    seen = set()
    for t in triples:
        if len(t) != 3:
            return False
        if any(p < 0 or p > 15 for p in t):
            return False
        if len(set(t)) != 3:
            return False
        key = tuple(sorted(t))
        if key in seen:
            return False
        seen.add(key)
    # Check linearity: any two triples share at most one point
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            inter = set(triples[i]) & set(triples[j])
            if len(inter) > 1:
                return False
    return True

def _independence_number(triples):
    """Compute the independence number: largest set of points containing no triple."""
    n = 16
    # A set S is independent if it contains no triple entirely.
    # Equivalently, S is NOT independent if there exists a triple t with t ⊆ S.
    # We want the max |S| such that no triple is contained in S.
    # This is equivalent to: min vertex cover of the 3-uniform hypergraph.
    # alpha = 16 - tau, where tau is the minimum vertex cover size.
    # But we can also compute directly: for each subset, check if it's independent.
    # That's 2^16 = 65536 subsets, each check is O(num_triples).
    # Better: a set S is independent iff for every triple t, t is not a subset of S.
    # i.e., for every triple t, at least one point of t is NOT in S.
    # So S is independent iff S is a hitting set complement... wait.
    # S contains no triple means: for all t in triples, t ∩ S ≠ t, i.e., t ⊄ S.
    # Equivalently, the complement C = V \ S hits every triple: C ∩ t ≠ ∅ for all t.
    # So alpha = 16 - tau where tau = min vertex cover.
    
    # Direct approach: iterate over all subsets from largest to smallest.
    # For a subset of size k, check if it's independent.
    # A subset S of size k is independent if no triple is fully contained in S.
    
    # Use bitmasks for speed
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    # Try subsets from size 16 down to 0
    # For each size k, check all C(16,k) subsets
    # A subset S (as bitmask) is independent if for all tm in triple_masks: (tm & S) != tm
    # i.e., no triple is fully contained in S.
    
    from itertools import combinations as _comb
    for k in range(16, -1, -1):
        for subset in _comb(range(16), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    independent = False
                    break
            if independent:
                return k
    return 0

def _greedy_construct(n_points=16, n_iter=500, seed=42, deadline=None):
    """Greedy construction of linear triple system with small independence number."""
    rng = random.Random(seed)
    best_alpha = 16
    best_triples = []
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Build a linear triple system greedily
        # Start with all possible triples, shuffle, add if linear
        all_triples = list(combinations(range(n_points), 3))
        rng.shuffle(all_triples)
        
        triples = []
        point_triple_count = [0] * n_points  # how many triples each point is in
        pair_used = set()  # pairs that are already in a triple (for linearity)
        
        for t in all_triples:
            t_set = set(t)
            # Check linearity: no pair in t is already used by another triple
            # Actually, linearity means any two triples share at most 1 point.
            # So for a new triple t, it's OK if it shares 0 or 1 point with each existing triple.
            # Equivalent: no pair from t has appeared in any previous triple.
            ok = True
            for a, b in combinations(t, 2):
                if (a, b) in pair_used or (b, a) in pair_used:
                    ok = False
                    break
            if ok:
                triples.append(list(t))
                for a, b in combinations(t, 2):
                    pair_used.add((a, b))
                    pair_used.add((b, a))
                for p in t:
                    point_triple_count[p] += 1
        
        # Compute independence number
        alpha = _independence_number(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
    
    return best_triples, best_alpha

def _local_search(triples, n_points=16, max_iter=200, deadline=None):
    """Try to improve by swapping triples or adding/removing."""
    if not triples:
        return triples, _independence_number(triples)
    
    best_alpha = _independence_number(triples)
    current = [list(t) for t in triples]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a random valid triple
        all_triples = list(combinations(range(n_points), 3))
        rng = random.Random(it * 7919 + 13)
        rng.shuffle(all_triples)
        
        for t in all_triples:
            if len(current) > 200:  # limit size
                break
            t_set = set(t)
            ok = True
            for existing in current:
                inter = t_set & set(existing)
                if len(inter) > 1:
                    ok = False
                    break
            if ok:
                current.append(list(t))
                new_alpha = _independence_number(current)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    improved = True
                else:
                    current.pop()
        
        # Try removing a triple that might help (unlikely but check)
        if not improved:
            # Try a random swap: remove one triple, add another
            if len(current) > 0:
                idx = rng.randrange(len(current))
                removed = current.pop(idx)
                for t in all_triples[:50]:
                    t_set = set(t)
                    if t_set == set(removed):
                        continue
                    ok = True
                    for existing in current:
                        inter = t_set & set(existing)
                        if len(inter) > 1:
                            ok = False
                            break
                    if ok:
                        current.append(list(t))
                        new_alpha = _independence_number(current)
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            improved = True
                        else:
                            current.pop()
                            current.insert(idx, removed)
                            break
                else:
                    current.insert(idx, removed)
    
    return current, best_alpha

def construct():
    """Construct a linear triple system on 16 points minimizing independence number."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Phase 1: Greedy construction with multiple restarts
    best_triples, best_alpha = _greedy_construct(
        n_points=16, 
        n_iter=300, 
        seed=12345,
        deadline=deadline
    )
    
    if time.time() > deadline:
        return best_triples
    
    # Phase 2: Local search improvement
    improved_triples, improved_alpha = _local_search(
        best_triples,
        n_points=16,
        max_iter=100,
        deadline=deadline
    )
    
    if improved_alpha < best_alpha:
        best_triples = improved_triples
        best_alpha = improved_alpha
    
    # Phase 3: More greedy restarts with different strategies
    for seed in [7, 42, 99, 2024, 31415]:
        if time.time() > deadline:
            break
        t, a = _greedy_construct(
            n_points=16,
            n_iter=50,
            seed=seed,
            deadline=deadline
        )
        if a < best_alpha:
            best_triples = t
            best_alpha = a
    
    if time.time() > deadline:
        return best_triples
    
    # Final local search
    final_triples, final_alpha = _local_search(
        best_triples,
        n_points=16,
        max_iter=50,
        deadline=deadline
    )
    
    if final_alpha < best_alpha:
        best_triples = final_triples
    
    return best_triples

# EVOLVE-BLOCK-END
