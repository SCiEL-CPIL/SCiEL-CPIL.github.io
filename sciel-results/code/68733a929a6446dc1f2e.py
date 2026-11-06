import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # Precompute all possible triples
    all_triples = list(combinations(range(16), 3))
    n_all = len(all_triples)
    
    # We will try multiple random orders and take the best result
    best_system = None
    best_size = 0
    
    # The constraint: no three disjoint pairs have all eight transversal triples present.
    # For 3 disjoint pairs P1={a,a'}, P2={b,b'}, P3={c,c'}, the 8 transversals are
    # all triples (x,y,z) where x in P1, y in P2, z in P3.
    
    # Efficient violation check:
    # When adding triple T=(x,y,z), a violation occurs if there exist x',y',z'
    # (all distinct from each other and from x,y,z) such that all 8 triples
    # from {x,x'}x{y,y'}x{z,z'} are in the system.
    # Since T just got added, we need the other 7 to already be present.
    
    def check_violation(triple, system_set):
        x, y, z = triple
        # For each possible "partner" x' for x, y' for y, z' for z
        # such that all 6 points are distinct
        for x_prime in range(16):
            if x_prime == x or x_prime == y or x_prime == z:
                continue
            for y_prime in range(16):
                if y_prime == x or y_prime == y or y_prime == z or y_prime == x_prime:
                    continue
                for z_prime in range(16):
                    if (z_prime == x or z_prime == y or z_prime == z or 
                        z_prime == x_prime or z_prime == y_prime):
                        continue
                    
                    # Check if the other 7 transversals are in system_set
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
    
    # Strategy: Try multiple random greedy constructions with different shuffles
    # Also try a "conflict-avoiding" greedy that prioritizes triples with fewer conflicts
    
    # Precompute for each triple, the number of potential conflicts (how many 
    # x',y',z' combinations would complete a K(2,2,2) if this triple were added)
    # This is a static property: for triple (x,y,z), the number of (x',y',z') 
    # with all 6 distinct is 13*12*11 = 1716. This is the same for all triples.
    # So that doesn't help differentiate.
    
    # Better: use a scoring function that considers how "central" a triple is.
    # Triples that share many points with others are more likely to create conflicts.
    
    # Let's try a different approach: local search / hill climbing
    # Start with a random valid system, then try to add triples or swap.
    
    # Actually, let's first try multiple random greedy runs with different strategies.
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Strategy 1: Random shuffle greedy
        if attempt % 2 == 0:
            order = list(all_triples)
            random.shuffle(order)
        else:
            # Strategy 2: Sort by some heuristic - e.g., prefer triples with 
            # lower-index points first (more structured)
            order = list(all_triples)
            random.shuffle(order)
        
        system = []
        system_set = set()
        
        for triple in order:
            if time.time() > deadline:
                break
            
            if not check_violation(triple, system_set):
                system.append(list(triple))
                system_set.add(triple)
        
        if len(system) > best_size:
            best_size = len(system)
            best_system = [t[:] for t in system]
        
        # If we've found a very good solution, we can stop early
        if best_size > 300:
            break
    
    # Now try to improve via local search: add triples that were rejected
    if best_system is not None and time.time() < deadline:
        current_system = [list(t) for t in best_system]
        current_set = set(tuple(t) for t in current_system)
        
        # Find triples not in the system
        rejected = [t for t in all_triples if t not in current_set]
        random.shuffle(rejected)
        
        improved = False
        for triple in rejected:
            if time.time() > deadline:
                break
            if not check_violation(triple, current_set):
                current_system.append(list(triple))
                current_set.add(triple)
                improved = True
        
        if len(current_system) > best_size:
            best_size = len(current_system)
            best_system = current_system
    
    # Try a different approach: start with a structured system and add triples
    if time.time() < deadline - 20:
        # Start with all triples containing point 0 (star)
        star = [tuple(t) for t in all_triples if 0 in t]
        star_set = set(star)
        star_system = [list(t) for t in star]
        
        # Try to add triples not containing 0
        candidates = [t for t in all_triples if 0 not in t]
        random.shuffle(candidates)
        
        for triple in candidates:
            if time.time() > deadline:
                break
            if not check_violation(triple, star_set):
                star_system.append(list(triple))
                star_set.add(triple)
        
        if len(star_system) > best_size:
            best_size = len(star_system)
            best_system = star_system
    
    # Final: try to add any remaining triples
    if best_system is not None and time.time() < deadline:
        final_set = set(tuple(t) for t in best_system)
        remaining = [t for t in all_triples if t not in final_set]
        random.shuffle(remaining)
        
        for triple in remaining:
            if time.time() > deadline:
                break
            if not check_violation(triple, final_set):
                best_system.append(list(triple))
                final_set.add(triple)
        
        best_size = len(best_system)
    
    # Ensure we have a valid fallback
    if best_system is None or best_size == 0:
        star_system = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        return star_system
    
    return best_system

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
