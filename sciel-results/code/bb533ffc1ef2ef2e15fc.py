"""Fresh design: bitmask independence + greedy construction + local search."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _independence_number(triples):
    """Compute independence number using bitmask. Returns the size of the largest
    subset of {0..15} containing no triple."""
    n = 16
    full = (1 << n) - 1
    # covered[mask] = True if mask contains at least one triple
    covered = [False] * (1 << n)
    for t in triples:
        a, b, c = t
        tri_mask = (1 << a) | (1 << b) | (1 << c)
        # Enumerate all supersets of tri_mask
        remaining = full ^ tri_mask
        sub = remaining
        while True:
            covered[tri_mask | sub] = True
            if sub == 0:
                break
            sub = (sub - 1) & remaining
    # Find the largest independent set
    best = 0
    for mask in range(1 << n):
        if not covered[mask]:
            cnt = bin(mask).count('1')
            if cnt > best:
                best = cnt
    return best

def _can_add(triples, new_triple):
    """Check if new_triple can be added while maintaining linearity."""
    a, b, c = new_triple
    new_set = {a, b, c}
    for t in triples:
        if t[0] in new_set and t[1] in new_set:
            return False
        if t[0] in new_set and t[2] in new_set:
            return False
        if t[1] in new_set and t[2] in new_set:
            return False
    return True

def _greedy_build(rng):
    """Greedy construction of a linear triple system."""
    all_triples = list(combinations(range(16), 3))
    rng.shuffle(all_triples)
    triples = []
    for t in all_triples:
        if _can_add(triples, t):
            triples.append([t[0], t[1], t[2]])
    return triples

def _local_search(triples, rng, time_limit):
    """Local search: try to replace triples to reduce independence number."""
    best_triples = [list(t) for t in triples]
    best_alpha = _independence_number(best_triples)
    current_triples = [list(t) for t in best_triples]
    current_alpha = best_alpha
    start = time.time()
    
    all_triples_list = list(combinations(range(16), 3))
    
    while time.time() - start < time_limit:
        # Pick a random triple to potentially replace
        if not current_triples:
            break
        idx = rng.randrange(len(current_triples))
        old_triple = tuple(current_triples[idx])
        
        # Try replacements
        found_improvement = False
        for _ in range(30):
            new_triple = rng.choice(all_triples_list)
            if new_triple == old_triple:
                continue
            remaining = current_triples[:idx] + current_triples[idx+1:]
            if _can_add(remaining, new_triple):
                new_set = remaining + [[new_triple[0], new_triple[1], new_triple[2]]]
                alpha = _independence_number(new_set)
                if alpha < current_alpha:
                    current_triples = new_set
                    current_alpha = alpha
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_triples = [list(t) for t in current_triples]
                    found_improvement = True
                    break
                elif alpha == current_alpha:
                    # Accept with some probability for exploration
                    if rng.random() < 0.2:
                        current_triples = new_set
                        found_improvement = True
                        break
        
        # Try removal
        if not found_improvement and len(current_triples) > 1:
            removed = current_triples[:idx] + current_triples[idx+1:]
            alpha = _independence_number(removed)
            if alpha < current_alpha:
                current_triples = removed
                current_alpha = alpha
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_triples = [list(t) for t in current_triples]
    
    return best_triples, best_alpha

def construct():
    start_time = time.time()
    deadline = 145.0
    
    rng = random.Random(42)
    
    best_triples = None
    best_alpha = 16
    
    # Phase 1: Multiple greedy constructions
    for seed in range(30):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        rng_local = random.Random(seed)
        triples = _greedy_build(rng_local)
        alpha = _independence_number(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
    
    # Phase 2: Local search on best
    if best_triples is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 3:
            best_triples, best_alpha = _local_search(best_triples, rng, remaining)
    
    # Phase 3: More greedy + local search if time allows
    if best_triples is not None:
        elapsed = time.time() - start_time
        for seed in range(20):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            rng_local = random.Random(1000 + seed)
            triples = _greedy_build(rng_local)
            alpha = _independence_number(triples)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples
                remaining = deadline - (time.time() - start_time) - 2
                if remaining > 3:
                    best_triples, best_alpha = _local_search(best_triples, rng_local, remaining)
    
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
