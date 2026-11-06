import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    
    # For each triple, find all K(2,2,2) configurations it participates in
    # A triple (a,b,c) is in a K(2,2,2) with pairs {a,a'}, {b,b'}, {c,c'}
    # where a' != a, b' != b, c' != c, all distinct
    
    # Precompute: for each triple, the set of "partner" triples that would complete K(2,2,2)
    # If we have triple (a,b,c), a K(2,2,2) with pairs {a,a'},{b,b'},{c,c'} 
    # requires all 8: (a,b,c),(a,b,c'),(a,b',c),(a,b',c'),(a',b,c),(a',b,c'),(a',b',c),(a',b',c')
    
    # Strategy: Greedy add triples, maintaining that no K(2,2,2) is complete
    # For efficiency, track for each potential K(2,2,2) how many of its 8 triples are present
    
    # Number of ways to choose 3 disjoint pairs from 16 points:
    # Choose 6 points: C(16,6) = 8008, then partition into 3 pairs: 15 ways
    # Total K(2,2,2) configs: 8008 * 15 = 120120
    
    # For each K(2,2,2) config, track count of present triples
    # This is too many to track individually. Instead, use a different approach.
    
    # Better: Greedy construction with backtracking-free approach
    # Sort triples by some heuristic, add if it doesn't complete any K(2,2,2)
    
    # For a triple t=(a,b,c), it completes a K(2,2,2) if there exist a',b',c' 
    # such that all other 7 transversal triples are already present.
    
    # Precompute for each triple, the list of "completing" configurations
    # Triple (a,b,c) completes K(2,2,2) with pairs {a,a'},{b,b'},{c,c'} 
    # if (a,b,c'),(a,b',c),(a,b',c'),(a',b,c),(a',b,c'),(a',b',c),(a',b',c') are all present
    
    # This is expensive to check. Let's use a different strategy.
    
    # Use simulated annealing:
    # State: set of triples
    # Move: add a triple, or remove a triple, or swap
    # Energy: -number_of_triples + penalty * number_of_violations
    
    # First, get a good initial solution using greedy
    
    def count_violations(triple_set_local):
        """Count K(2,2,2) violations. Expensive, use for validation."""
        violations = 0
        # For each set of 6 points, check all 15 pairings
        for six in combinations(range(n), 6):
            s = set(six)
            if not s.issubset(triple_set_local.__class__ and set()):
                pass
            # Check if all 6 points are "covered" in some sense - skip, just check all
            # For 15 pairings of 6 points into 3 pairs
            a, b, c, d, e, f = six
            pairings = [
                ({a,b},{c,d},{e,f}),
                ({a,b},{c,e},{d,f}),
                ({a,b},{c,f},{d,e}),
                ({a,c},{b,d},{e,f}),
                ({a,c},{b,e},{d,f}),
                ({a,c},{b,f},{d,e}),
                ({a,d},{b,c},{e,f}),
                ({a,d},{b,e},{c,f}),
                ({a,d},{b,f},{c,e}),
                ({a,e},{b,c},{d,f}),
                ({a,e},{b,d},{c,f}),
                ({a,e},{b,f},{c,d}),
                ({a,f},{b,c},{d,e}),
                ({a,f},{b,d},{c,e}),
                ({a,f},{b,e},{c,d}),
            ]
            for p1, p2, p3 in pairings:
                # Check if all 8 transversal triples are in triple_set_local
                all_present = True
                for x in p1:
                    for y in p2:
                        for z in p3:
                            t = tuple(sorted((x, y, z)))
                            if t not in triple_set_local:
                                all_present = False
                                break
                        if not all_present:
                            break
                    if not all_present:
                        break
                if all_present:
                    violations += 1
        return violations
    
    # Faster violation check for incremental updates
    # For adding triple t=(a,b,c): check if it completes any K(2,2,2)
    # It completes with a',b',c' if all 7 other transversal triples are present
    
    def would_complete_k222(t, triple_set_local):
        """Check if adding triple t would complete any K(2,2,2)."""
        a, b, c = t
        # For each choice of a' != a, b' != b, c' != c, all distinct from a,b,c and each other
        # Check if the other 7 transversal triples are all present
        for a_prime in range(n):
            if a_prime == a:
                continue
            for b_prime in range(n):
                if b_prime in (a, b, a_prime):
                    continue
                for c_prime in range(n):
                    if c_prime in (a, b, c, a_prime, b_prime):
                        continue
                    # Check all 7 other transversal triples
                    other_triples = [
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime),
                    ]
                    all_present = True
                    for ot in other_triples:
                        if tuple(sorted(ot)) not in triple_set_local:
                            all_present = False
                            break
                    if all_present:
                        return True
        return False
    
    # Greedy construction
    random.seed(42)
    triple_list = list(all_triples)
    random.shuffle(triple_list)
    
    current_set = set()
    for t in triple_list:
        if not would_complete_k222(t, current_set):
            current_set.add(t)
        if time.time() > deadline - 30:
            break
    
    best_set = set(current_set)
    best_count = len(current_set)
    
    # Local search: try to add more triples
    # Try each missing triple, see if it can be added (may need to remove others)
    
    # Simulated annealing
    if time.time() < deadline - 20:
        temp = 1.0
        cooling = 0.999
        current_set = set(best_set)
        
        # Build list of missing triples
        missing = [t for t in all_triples if t not in current_set]
        present = list(current_set)
        
        iteration = 0
        while time.time() < deadline - 5:
            iteration += 1
            # Random move: add a random missing triple, or remove a random present triple
            if random.random() < 0.7 and missing:
                # Try to add a triple
                idx = random.randint(0, len(missing) - 1)
                t = missing[idx]
                if not would_complete_k222(t, current_set):
                    current_set.add(t)
                    missing.pop(idx)
                    present.append(t)
                    if len(current_set) > best_count:
                        best_count = len(current_set)
                        best_set = set(current_set)
                else:
                    # Would create violation; accept with probability exp(-delta/temp)
                    # delta = 1 (one violation created)
                    if random.random() < pow(2.718281828, -1.0 / max(temp, 0.01)):
                        current_set.add(t)
                        missing.pop(idx)
                        present.append(t)
            elif present:
                # Remove a random present triple
                idx = random.randint(0, len(present) - 1)
                t = present[idx]
                current_set.remove(t)
                present.pop(idx)
                missing.append(t)
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
    
    return [list(t) for t in best_set]
