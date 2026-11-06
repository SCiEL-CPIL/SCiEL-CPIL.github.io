import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_system = None
    best_size = 0
    
    # Precompute all possible triples
    all_triples = list(combinations(range(16), 3))
    
    # Shuffle for randomness
    random.shuffle(all_triples)
    
    # We will try to build a system greedily, checking constraints as we go.
    # The constraint is: no three disjoint pairs have all eight transversal triples present.
    # Let pairs be (a,b), (c,d), (e,f) with all 6 points distinct.
    # The 8 transversal triples are formed by picking one point from each pair.
    # e.g., (a,c,e), (a,c,f), (a,d,e), (a,d,f), (b,c,e), (b,c,f), (b,d,e), (b,d,f).
    # If all 8 are in the system, it's invalid.
    
    # To check this efficiently, we can maintain a set of triples.
    # When adding a new triple T = (x, y, z), we need to check if it completes any forbidden configuration.
    # A forbidden configuration is defined by 3 disjoint pairs.
    # If we add (x,y,z), it could be part of such a configuration.
    # The other 7 triples in the configuration would have to already exist.
    
    # Efficient check:
    # For a new triple (x,y,z), consider all pairs (p1, p2) disjoint from {x,y,z}.
    # Actually, the structure is 3 pairs. Let the pairs be P1, P2, P3.
    # If (x,y,z) is in the system, it picks one from each pair.
    # So {x,y,z} must contain exactly one element from each of the 3 pairs.
    # This means the 3 pairs are a partition of 6 points, and {x,y,z} is a transversal.
    # The other 7 transversals must be in the system.
    
    # How to check if adding (x,y,z) creates a violation?
    # A violation exists if there is a set of 3 disjoint pairs { (a,b), (c,d), (e,f) } such that
    # all 8 transversals are in the system.
    # If we just added (x,y,z), then (x,y,z) must be one of the 8.
    # The other 7 must have been added previously.
    
    # We can check:
    # For any two other points u, v (distinct from x,y,z), if (x,u,v) is in system, and ... no, that's not right.
    # The 8 triples are determined by the 3 pairs.
    # If (x,y,z) is a transversal, the pairs must be:
    # Pair 1 contains x (and some other point x')
    # Pair 2 contains y (and some other point y')
    # Pair 3 contains z (and some other point z')
    # The 8 triples are all combinations of {x,x'}, {y,y'}, {z,z'}.
    # So if we assume (x,y,z) is in the system, we need to find x', y', z' such that
    # (x',y,z), (x,y',z), (x,y,z') are in the system? No.
    # The triples are:
    # (x,y,z) - just added
    # (x',y,z)
    # (x,y',z)
    # (x,y,z')
    # (x',y',z)
    # (x',y,z')
    # (x,y',z')
    # (x',y',z')
    
    # So, to check if adding (x,y,z) creates a violation, we can iterate over all possible
    # "complementary" points x', y', z'.
    # x' is a point not in {x,y,z}.
    # y' is a point not in {x,y,z,x'}.
    # z' is a point not in {x,y,z,x',y'}.
    
    # There are 13 choices for x', 12 for y', 11 for z'. Total 13*12*11 = 1716 checks.
    # Each check involves looking up 7 triples in a set.
    # This is fast enough for a greedy approach.
    
    def check_violation(triple, system_set):
        x, y, z = triple
        for x_prime in range(16):
            if x_prime in (x, y, z):
                continue
            for y_prime in range(16):
                if y_prime in (x, y, z, x_prime):
                    continue
                for z_prime in range(16):
                    if z_prime in (x, y, z, x_prime, y_prime):
                        continue
                    
                    # Check if the other 7 transversals are in system_set
                    # We need to check if these 7 are in the system:
                    # (x_prime, y, z)
                    # (x, y_prime, z)
                    # (x, y, z_prime)
                    # (x_prime, y_prime, z)
                    # (x_prime, y, z_prime)
                    # (x, y_prime, z_prime)
                    # (x_prime, y_prime, z_prime)
                    
                    t1 = tuple(sorted((x_prime, y, z)))
                    t2 = tuple(sorted((x, y_prime, z)))
                    t3 = tuple(sorted((x, y, z_prime)))
                    t4 = tuple(sorted((x_prime, y_prime, z)))
                    t5 = tuple(sorted((x_prime, y, z_prime)))
                    t6 = tuple(sorted((x, y_prime, z_prime)))
                    t7 = tuple(sorted((x_prime, y_prime, z_prime)))
                    
                    if (t1 in system_set and t2 in system_set and t3 in system_set and 
                        t4 in system_set and t5 in system_set and t6 in system_set and 
                        t7 in system_set):
                        return True
        return False

    # Greedy construction
    system = []
    system_set = set()
    
    for triple in all_triples:
        if time.time() > deadline:
            break
            
        if not check_violation(triple, system_set):
            system.append(list(triple))
            system_set.add(triple)
            
        if len(system) > best_size:
            best_size = len(system)
            best_system = [t[:] for t in system]
            
        # If we've exhausted all triples or hit deadline, stop
        if len(system) == len(all_triples):
            break

    # If we didn't find anything better than the star, return the star
    star_system = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    if best_size < len(star_system):
        return star_system
    else:
        return best_system

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
