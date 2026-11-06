import time
import random
import itertools
from functools import lru_cache

def _independence_number(triples):
    """Compute the independence number: largest subset with no complete triple."""
    n = 16
    triple_set = set()
    for t in triples:
        triple_set.add(tuple(sorted(t)))
    
    # We want the largest S such that no triple is fully contained in S
    # Check from size 16 down
    for size in range(n, 0, -1):
        found = False
        for subset in itertools.combinations(range(n), size):
            s = set(subset)
            has_triple = False
            for t in triple_set:
                if t[0] in s and t[1] in s and t[2] in s:
                    has_triple = True
                    break
            if not has_triple:
                found = True
                break
        if found:
            return size
    return 0


def _is_linear(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _can_add(triples, new_triple):
    """Check if adding new_triple maintains linearity."""
    new_set = set(new_triple)
    for t in triples:
        common = new_set & set(t)
        if len(common) > 1:
            return False
    return True


def _pair_count(triples):
    """Count distinct pairs covered by triples."""
    pairs = set()
    for t in triples:
        for a, b in itertools.combinations(t, 2):
            pairs.add((min(a, b), max(a, b)))
    return len(pairs)


def _greedy_build(deadline):
    """Greedy construction: add triples that maximize new pairs while maintaining linearity."""
    triples = []
    all_pairs = set(itertools.combinations(range(16), 2))
    
    while time.time() < deadline:
        best_triple = None
        best_new_pairs = -1
        # Try all possible triples not already present
        existing = set(tuple(sorted(t)) for t in triples)
        for a in range(16):
            for b in range(a + 1, 16):
                for c in range(b + 1, 16):
                    t = (a, b, c)
                    if t in existing:
                        continue
                    if not _can_add(triples, t):
                        continue
                    # Count new pairs
                    new_pairs = 0
                    for x, y in itertools.combinations(t, 2):
                        p = (min(x, y), max(x, y))
                        if p not in all_pairs:
                            new_pairs += 1
                    if new_pairs > best_new_pairs:
                        best_new_pairs = new_pairs
                        best_triple = t
        if best_triple is None:
            break
        triples.append(list(best_triple))
        for x, y in itertools.combinations(best_triple, 2):
            all_pairs.discard((min(x, y), max(x, y)))
    
    return triples


def _local_search(triples, deadline):
    """Local search: try swapping triples to improve pair coverage and reduce independence number."""
    best_triples = list(triples)
    best_pairs = _pair_count(best_triples)
    
    while time.time() < deadline:
        improved = False
        # Try removing one triple and adding a different one
        for remove_idx in range(len(best_triples)):
            candidate = best_triples[:remove_idx] + best_triples[remove_idx + 1:]
            # Try adding various triples
            existing = set(tuple(sorted(t)) for t in candidate)
            for a in range(16):
                for b in range(a + 1, 16):
                    for c in range(b + 1, 16):
                        t = (a, b, c)
                        if t in existing:
                            continue
                        if not _can_add(candidate, t):
                            continue
                        new_triples = candidate + [list(t)]
                        new_pairs = _pair_count(new_triples)
                        if new_pairs > best_pairs:
                            best_triples = new_triples
                            best_pairs = new_pairs
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_triples


def _random_greedy(deadline, rng):
    """Randomized greedy: randomly choose from top candidates."""
    triples = []
    all_pairs = set(itertools.combinations(range(16), 2))
    
    while time.time() < deadline:
        best_triple = None
        best_new_pairs = -1
        candidates = []
        existing = set(tuple(sorted(t)) for t in triples)
        for a in range(16):
            for b in range(a + 1, 16):
                for c in range(b + 1, 16):
                    t = (a, b, c)
                    if t in existing:
                        continue
                    if not _can_add(triples, t):
                        continue
                    new_pairs = 0
                    for x, y in itertools.combinations(t, 2):
                        p = (min(x, y), max(x, y))
                        if p not in all_pairs:
                            new_pairs += 1
                    if new_pairs > best_new_pairs:
                        best_new_pairs = new_pairs
                        candidates = [t]
                    elif new_pairs == best_new_pairs:
                        candidates.append(t)
        
        if not candidates:
            break
        
        # Pick randomly from top candidates
        best_triple = rng.choice(candidates)
        triples.append(list(best_triple))
        for x, y in itertools.combinations(best_triple, 2):
            all_pairs.discard((min(x, y), max(x, y)))
    
    return triples


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    rng = random.Random(42)
    
    best_triples = None
    best_indep = 17  # Start high
    best_pairs = -1
    
    # Phase 1: Greedy construction
    triples = _greedy_build(deadline)
    if triples:
        indep = _independence_number(triples)
        pairs = _pair_count(triples)
        if indep < best_indep or (indep == best_indep and pairs > best_pairs):
            best_indep = indep
            best_pairs = pairs
            best_triples = list(triples)
    
    if time.time() >= deadline:
        return best_triples if best_triples else []
    
    # Phase 2: Local search on the greedy result
    if best_triples:
        improved = _local_search(best_triples, deadline)
        indep = _independence_number(improved)
        pairs = _pair_count(improved)
        if indep < best_indep or (indep == best_indep and pairs > best_pairs):
            best_indep = indep
            best_pairs = pairs
            best_triples = improved
    
    if time.time() >= deadline:
        return best_triples if best_triples else []
    
    # Phase 3: Multiple randomized greedy attempts
    attempts = 0
    while time.time() < deadline and attempts < 20:
        attempts += 1
        rng_seed = rng.randint(0, 10**9)
        rng2 = random.Random(rng_seed)
        triples = _random_greedy(deadline, rng2)
        if triples:
            # Local search
            improved = _local_search(triples, deadline)
            indep = _independence_number(improved)
            pairs = _pair_count(improved)
            if indep < best_indep or (indep == best_indep and pairs > best_pairs):
                best_indep = indep
                best_pairs = pairs
                best_triples = improved
    
    return best_triples if best_triples else []
