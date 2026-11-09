import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    n = 16
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # For each triple, precompute which "forbidden configurations" it participates in
    # A forbidden configuration is a set of 3 disjoint pairs where all 8 transversal triples are present
    # When we add a triple, we need to check if it completes any such configuration
    
    # Strategy: Greedy with backtracking-like approach
    # We'll use a randomized greedy approach with multiple restarts
    
    # For efficiency, let's think about the constraint differently:
    # A set of 3 disjoint pairs (x1,y1), (x2,y2), (x3,y3) creates a forbidden config
    # if all 8 triples from {x1,y1} x {x2,y2} x {x3,y3} are in our system.
    
    # When adding triple T = {a,b,c}, it could be part of a forbidden config if:
    # There exist disjoint pairs such that T is one of the 8 transversals
    # and the other 7 transversals are already in our system.
    
    # Let's precompute for each triple, what other 7 triples would need to be present
    # to form a forbidden config with it.
    
    # Actually, let's use a different approach: 
    # Maintain a set of "dangerous" partial configurations.
    
    # Simpler approach: For each new triple, check all possible sets of 3 disjoint pairs
    # that could use this triple as one of the 8 transversals.
    
    # For triple (a,b,c), it can be a transversal of pairs (x1,y1),(x2,y2),(x3,y3)
    # where {x1,x2,x3} or {x1,x2,y3} etc. = {a,b,c}
    # This means the three pairs must be formed from elements that include a,b,c
    
    # Let's think: if (a,b,c) is a transversal, then each of a,b,c must come from a different pair.
    # So we need to assign a to one pair, b to another, c to the third.
    # The pairs are disjoint and each pair has 2 elements.
    # So the 6 elements of the 3 pairs are: {a,b,c} plus 3 more elements (one from each pair).
    
    # For each way to assign a,b,c to the three pairs, and for each choice of the 3 additional elements,
    # we can check if the other 7 transversals are in our system.
    
    # This is getting complex. Let's use a simpler greedy with constraint checking.
    
    def check_if_valid(system_set, triple):
        """Check if adding triple to system_set creates a forbidden configuration."""
        a, b, c = triple
        # For each permutation of assigning a,b,c to the three pair positions
        for perm in itertools.permutations([a, b, c]):
            # perm[0] goes to pair 1, perm[1] to pair 2, perm[2] to pair 3
            # We need to choose the other element for each pair
            # The other elements must be distinct from each other and from a,b,c
            available = [x for x in range(n) if x not in (a, b, c)]
            for d in available:
                for e in available:
                    if e <= d:
                        continue
                    for f in available:
                        if f <= e:
                            continue
                        # Pairs are (perm[0], d), (perm[1], e), (perm[2], f)
                        # Check if all 8 transversals are in system_set or equal to triple
                        # The 8 transversals:
                        # (perm[0], perm[1], perm[2]) - this is our triple (in some order)
                        # (perm[0], perm[1], f)
                        # (perm[0], e, perm[2])
                        # (perm[0], e, f)
                        # (d, perm[1], perm[2])
                        # (d, perm[1], f)
                        # (d, e, perm[2])
                        # (d, e, f)
                        
                        transversals = [
                            (perm[0], perm[1], perm[2]),
                            (perm[0], perm[1], f),
                            (perm[0], e, perm[2]),
                            (perm[0], e, f),
                            (d, perm[1], perm[2]),
                            (d, perm[1], f),
                            (d, e, perm[2]),
                            (d, e, f)
                        ]
                        
                        # Check if all 8 are in system_set (the triple itself is being added,
                        # so we check if the other 7 are already in system_set)
                        all_present = True
                        for t in transversals:
                            t_set = frozenset(t)
                            if t_set == frozenset(triple):
                                continue  # This is the triple we're adding
                            if t_set not in system_set:
                                all_present = False
                                break
                        if all_present:
                            return False  # Would create forbidden config
        return True
    
    best_system = None
    best_count = 0
    
    # Try multiple random orderings
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 5:
            break
            
        # Shuffle the triples
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        system_set = set()
        system_list = []
        
        for triple in shuffled:
            if time.time() > deadline - 5:
                break
            if check_if_valid(system_set, triple):
                system_set.add(frozenset(triple))
                system_list.append(list(triple))
        
        if len(system_list) > best_count:
            best_count = len(system_list)
            best_system = system_list[:]
        
        # Early termination if we've used too much time
        if time.time() > deadline - 10:
            break
    
    if best_system is None:
        # Fallback: return the seed solution
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_system
