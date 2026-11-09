"""Linear triple systems on 16 points with small independence number."""
import random
import time

# EVOLVE-BLOCK-START

def _is_valid(triples):
    """Check that any two triples share at most one point."""
    for i in range(len(triples)):
        si = set(triples[i])
        for j in range(i + 1, len(triples)):
            sj = set(triples[j])
            if len(si & sj) > 1:
                return False
    return True

def _independence_number(triples):
    """Compute the largest subset of {0..15} containing no triple."""
    n = 16
    # Build a set of frozensets for fast lookup
    triple_set = set()
    for t in triples:
        triple_set.add(frozenset(t))
    
    # Check all subsets from largest to smallest
    # 2^16 = 65536 subsets, very fast
    best = 0
    for mask in range(1, 1 << n):
        # Count bits
        if bin(mask).count('1') <= best:
            continue
        # Extract subset
        subset = set()
        for i in range(n):
            if mask & (1 << i):
                subset.add(i)
        # Check if subset contains any triple
        has_triple = False
        for t in triple_set:
            if t.issubset(subset):
                has_triple = True
                break
        if not has_triple:
            best = len(subset)
    return best

def _greedy_construct(seed, max_triples=40):
    """Greedy randomized construction of a linear triple system."""
    random.seed(seed)
    triples = []
    # Track pair usage: pair -> count
    pair_count = {}
    
    # Generate all possible triples
    all_triples = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                all_triples.append((a, b, c))
    
    random.shuffle(all_triples)
    
    for t in all_triples:
        if len(triples) >= max_triples:
            break
        # Check if all pairs in t are unused
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        ok = True
        for p in pairs:
            if pair_count.get(p, 0) >= 1:
                ok = False
                break
        if ok:
            triples.append(list(t))
            for p in pairs:
                pair_count[p] = pair_count.get(p, 0) + 1
    
    return triples

def _local_search(triples, seed, time_limit=5.0):
    """Local search: try swapping out a triple for a better one."""
    random.seed(seed)
    start_time = time.time()
    current = [list(t) for t in triples]
    current_indep = _independence_number(current)
    
    # Build pair_count for current
    pair_count = {}
    for t in current:
        a, b, c = t
        for p in [(a,b), (a,c), (b,c)]:
            pair_count[p] = pair_count.get(p, 0) + 1
    
    # Generate candidate triples not in current
    current_set = set(frozenset(t) for t in current)
    candidates = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                t = (a, b, c)
                if frozenset(t) not in current_set:
                    candidates.append(t)
    
    while time.time() - start_time < time_limit:
        # Try to swap: remove a random triple, add a candidate
        idx = random.randint(0, len(current) - 1)
        old_triple = current[idx]
        old_set = frozenset(old_triple)
        
        # Free up pairs
        for p in [(old_triple[0], old_triple[1]), (old_triple[0], old_triple[2]), (old_triple[1], old_triple[2])]:
            pair_count[p] -= 1
            if pair_count[p] == 0:
                del pair_count[p]
        
        # Find a valid candidate
        found = False
        for _ in range(50):  # Try up to 50 random candidates
            if not candidates:
                break
            cand = random.choice(candidates)
            a, b, c = cand
            pairs = [(a, b), (a, c), (b, c)]
            ok = True
            for p in pairs:
                if pair_count.get(p, 0) >= 1:
                    ok = False
                    break
            if ok:
                # Add candidate
                current[idx] = list(cand)
                for p in pairs:
                    pair_count[p] = pair_count.get(p, 0) + 1
                new_indep = _independence_number(current)
                if new_indep < current_indep:
                    current_indep = new_indep
                    found = True
                else:
                    # Revert
                    for p in pairs:
                        pair_count[p] -= 1
                        if pair_count[p] == 0:
                            del pair_count[p]
                    current[idx] = list(old_triple)
                    for p in [(old_triple[0], old_triple[1]), (old_triple[0], old_triple[2]), (old_triple[1], old_triple[2])]:
                        pair_count[p] = pair_count.get(p, 0) + 1
                break
        
        if not candidates:
            break
    
    return current, current_indep

def construct():
    deadline = time.time() + 130  # 130s to leave margin for 150s limit
    
    best_triples = None
    best_indep = 16  # worst case
    
    # Phase 1: Multiple greedy constructions
    seeds = range(200)
    for seed in seeds:
        if time.time() > deadline:
            break
        triples = _greedy_construct(seed)
        if not triples:
            continue
        indep = _independence_number(triples)
        if indep < best_indep:
            best_indep = indep
            best_triples = [list(t) for t in triples]
    
    # Phase 2: Local search on best and some good candidates
    if best_triples is not None:
        if time.time() < deadline - 10:
            result, indep = _local_search(best_triples, seed=42, time_limit=10)
            if indep < best_indep:
                best_indep = indep
                best_triples = result
    
    # Phase 3: More local search with different seeds if time permits
    if best_triples is not None and time.time() < deadline - 5:
        for ls_seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 2:
                break
            result, indep = _local_search(best_triples, seed=ls_seed, time_limit=5)
            if indep < best_indep:
                best_indep = indep
                best_triples = result
    
    if best_triples is None:
        # Fallback: seed program
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples

# EVOLVE-BLOCK-END
