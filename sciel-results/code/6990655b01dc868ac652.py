import time
import math
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    triples.append((a, b, c))
    return triples

def _build_conflict_graph(triples):
    """Build a graph where edge (x,y) means x and y appear together in some forbidden triple.
    Actually, we need triple-level conflicts. Let's build for each element, the list of triples it's in."""
    element_triples = defaultdict(list)
    for i, (a, b, c) in enumerate(triples):
        element_triples[a].append(i)
        element_triples[b].append(i)
        element_triples[c].append(i)
    return element_triples

def _check_valid_fast(S_set, triples, element_triples):
    """Check validity by iterating over triples and checking if all three are in S."""
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    triples = _precompute_triples(N)
    element_triples = _build_conflict_graph(triples)
    
    # Strategy 1: Find the largest valid contiguous suffix {K, K+1, ..., 180}
    best_suffix = None
    for K in range(1, N + 1):
        S = set(range(K, N + 1))
        if _check_valid_fast(S, triples, element_triples):
            best_suffix = S
            break
    
    if best_suffix is None:
        best_suffix = set(range(1, N + 1))
    
    best_set = set(best_suffix)
    best_size = len(best_set)
    
    # Strategy 2: Greedy add with conflict-aware ordering
    if time.time() < deadline - 5.0:
        candidates = list(range(1, N + 1))
        # Sort by number of conflicts (fewer conflicts = better to add first)
        candidates.sort(key=lambda x: len(element_triples[x]))
        random.shuffle(candidates)  # Add randomness
        
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if x not in best_set:
                new_set = best_set | {x}
                if _check_valid_fast(new_set, triples, element_triples):
                    best_set = new_set
                    best_size += 1
    
    # Strategy 3: Local search - swap operations
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current = list(best_set)
            random.shuffle(current)
            
            for x_remove in current:
                if time.time() > deadline - 3.0:
                    break
                reduced_set = best_set - {x_remove}
                
                # Try to add elements not in reduced_set, sorted by fewest conflicts
                available = [y for y in range(1, N + 1) if y not in reduced_set]
                available.sort(key=lambda x: len(element_triples[x]))
                random.shuffle(available)
                
                for x_add in available:
                    if time.time() > deadline - 3.0:
                        break
                    if x_add == x_remove:
                        continue
                    new_set = reduced_set | {x_add}
                    if _check_valid_fast(new_set, triples, element_triples):
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_size = len(best_set)
                            improved = True
                            break
                    elif len(new_set) == best_size:
                        # Same size, but maybe different composition could help later
                        # Accept with some probability for diversity
                        if random.random() < 0.1:
                            best_set = new_set
                            improved = True
                            break
    
    # Strategy 4: Multiple random restarts with greedy construction
    if time.time() < deadline - 10.0:
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            
            S = set()
            elements = list(range(1, N + 1))
            # Bias towards larger numbers (they have fewer conflicts)
            elements.sort(key=lambda x: (-x, random.random()))
            
            for x in elements:
                if time.time() > deadline - 3.0:
                    break
                new_set = S | {x}
                if _check_valid_fast(new_set, triples, element_triples):
                    S = new_set
            
            if len(S) > best_size:
                best_set = S
                best_size = len(best_set)
    
    # Strategy 5: Final polish - try to add any remaining elements
    if time.time() < deadline - 2.0:
        for x in range(1, N + 1):
            if time.time() > deadline - 1.0:
                break
            if x not in best_set:
                new_set = best_set | {x}
                if _check_valid_fast(new_set, triples, element_triples):
                    best_set = new_set
                    best_size += 1
    
    return list(best_set)

# EVOLVE-BLOCK-END
