"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
import numpy as np

def _independence_number(triples, n=16):
    """Compute the largest subset of points containing no triple using bitmasks."""
    if not triples:
        return n
    # Precompute triple bitmasks
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Check sizes from n down to 1
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            # Check if any triple is fully contained
            independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    independent = False
                    break
            if independent:
                return size
    return 0

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def _all_f24_lines():
    """Generate all triples {a,b,c} in F2^4 with a^b^c=0 (affine lines through origin).
    These are the 85 lines of the projective geometry PG(3,2)."""
    triples = []
    seen = set()
    for a in range(1, 16):
        for b in range(a+1, 16):
            c = a ^ b
            if c == 0 or c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                triples.append(list(t))
    return triples

def _all_affine_lines_f24():
    """Generate all affine lines in F2^4: for each direction d and point p,
    the line is {p, p^d}. But that's only 2 points. Instead, use 2D affine subspaces.
    A 2D affine subspace has 4 points {p, p^u, p^v, p^u^v}.
    We form 4 triples from each 4-point set (all 4 ways to choose 3 of 4).
    But we need to ensure linearity (no pair repeated).
    
    Actually, let's use a different approach: generate all triples {a,b,c} where
    a,b,c are distinct and the triple is an affine line in some 2D subspace.
    In F2^4, any 3 distinct points a,b,c form a valid triple for our purposes
    as long as no pair is shared. The F2^4 structure gives us a natural pool.
    
    Let's generate all possible triples and filter for linearity greedily."""
    return list(combinations(range(16), 3))

def _greedy_construct(deadline):
    """Greedy construction: start with F2^4 lines, then greedily add triples
    that most reduce the independence number."""
    triples = []
    pair_used = set()
    
    # Phase 1: Add all F2^4 lines (85 triples on points 1-15)
    f24_lines = _all_f24_lines()
    for t in f24_lines:
        t = tuple(sorted(t))
        pairs = [(t[0],t[1]), (t[0],t[2]), (t[1],t[2])]
        if all(p not in pair_used for p in pairs):
            triples.append(list(t))
            pair_used.update(pairs)
    
    # Phase 2: Greedily add triples involving point 0 and others
    # Try all triples with point 0
    candidates_with_0 = []
    for b in range(1, 16):
        for c in range(b+1, 16):
            t = (0, b, c)
            pairs = [(0,b), (0,c), (b,c)]
            if all(p not in pair_used for p in pairs):
                candidates_with_0.append(t)
    
    # Sort by some heuristic: prefer triples that hit more "gaps"
    # For now, just add them all if linear
    for t in candidates_with_0:
        if time.time() > deadline:
            break
        pairs = [(t[0],t[1]), (t[0],t[2]), (t[1],t[2])]
        if all(p not in pair_used for p in pairs):
            triples.append(list(t))
            pair_used.update(pairs)
    
    # Phase 3: Try to add more triples (without 0) that are still linear
    all_triples = list(combinations(range(16), 3))
    rng = random.Random(42)
    rng.shuffle(all_triples)
    
    for t in all_triples:
        if time.time() > deadline:
            break
        t = tuple(sorted(t))
        pairs = [(t[0],t[1]), (t[0],t[2]), (t[1],t[2])]
        if all(p not in pair_used for p in pairs):
            triples.append(list(t))
            pair_used.update(pairs)
    
    return triples

def _greedy_minimize_indep(triples, deadline):
    """Given a set of linear triples, try to reduce independence number
    by strategically adding more triples."""
    triples = [list(t) for t in triples]
    pair_used = set()
    for t in triples:
        t = tuple(sorted(t))
        pairs = [(t[0],t[1]), (t[0],t[2]), (t[1],t[2])]
        pair_used.update(pairs)
    
    current_indep = _independence_number(triples, 16)
    
    # Generate all possible additional triples
    all_possible = []
    for t in combinations(range(16), 3):
        t = tuple(sorted(t))
        pairs = [(t[0],t[1]), (t[0],t[2]), (t[1],t[2])]
        if all(p not in pair_used for p in pairs):
            all_possible.append(t)
    
    # Try adding each candidate and check if it reduces independence
    # To be efficient, only try a subset
    rng = random.Random(99)
    rng.shuffle(all_possible)
    
    for t in all_possible:
        if time.time() > deadline:
            break
        new_triples = triples + [list(t)]
        new_indep = _independence_number(new_triples, 16)
        if new_indep < current_indep:
            current_indep = new_indep
            triples = new_triples
            pairs = [(t[0],t[1]), (t[0],t[2]), (t[1],t[2])]
            pair_used.update(pairs)
    
    return triples, current_indep

def _hill_climb_swap(triples, deadline, seed=0):
    """Try to swap out triples to reduce independence number."""
    rng = random.Random(seed)
    triples = [list(t) for t in triples]
    pair_used = set()
    for t in triples:
        t = tuple(sorted(t))
        pairs = [(t[0],t[1]), (t[0],t[2]), (t[1],t[2])]
        pair_used.update(pairs)
    
    current_indep = _independence_number(triples, 16)
    best_triples = [list(t) for t in triples]
    best_indep = current_indep
    
    # Generate candidate replacements
    all_candidates = list(combinations(range(16), 3))
    
    iterations = 0
    max_iterations = 5000
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        # Pick a random triple to potentially replace
        idx = rng.randint(0, len(triples)-1)
        old_t = tuple(sorted(triples[idx]))
        
        # Try to find a replacement that improves independence
        for _ in range(5):
            new_t = tuple(sorted(rng.choice(all_candidates)))
            # Check linearity with current set (minus the old triple)
            new_pairs = [(new_t[0],new_t[1]), (new_t[0],new_t[2]), (new_t[1],new_t[2])]
            old_pairs = [(old_t[0],old_t[1]), (old_t[0],old_t[2]), (old_t[1],old_t[2])]
            
            # Check if new triple's pairs conflict with any other triple
            conflict = False
            for p in new_pairs:
                if p in pair_used and p not in old_pairs:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Build new set
            new_triples = triples[:idx] + [list(new_t)] + triples[idx+1:]
            new_indep = _independence_number(new_triples, 16)
            
            if new_indep < best_indep:
                # Accept the swap
                # Remove old pairs, add new pairs
                for p in old_pairs:
                    pair_used.discard(p)
                for p in new_pairs:
                    pair_used.add(p)
                triples = new_triples
                best_triples = [list(t) for t in triples]
                best_indep = new_indep
                current_indep = new_indep
                break
    
    return best_triples, best_indep

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Phase 1: Deterministic algebraic construction
    triples = _greedy_construct(deadline)
    if not _is_linear(triples):
        # Fallback
        triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    current_indep = _independence_number(triples, 16)
    best_triples = [list(t) for t in triples]
    best_indep = current_indep
    
    # Phase 2: Greedy addition of more triples
    if time.time() < deadline:
        improved, indep = _greedy_minimize_indep(best_triples, deadline)
        if indep < best_indep:
            best_indep = indep
            best_triples = improved
    
    # Phase 3: Hill climbing with swaps
    if time.time() < deadline:
        improved, indep = _hill_climb_swap(best_triples, deadline, seed=123)
        if indep < best_indep:
            best_indep = indep
            best_triples = improved
    
    # Phase 4: Another round of greedy + swap with different seed
    if time.time() < deadline:
        improved, indep = _greedy_minimize_indep(best_triples, deadline)
        if indep < best_indep:
            best_indep = indep
            best_triples = improved
    
    if time.time() < deadline:
        improved, indep = _hill_climb_swap(best_triples, deadline, seed=456)
        if indep < best_indep:
            best_indep = indep
            best_triples = improved
    
    if best_triples is None or not best_triples:
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
