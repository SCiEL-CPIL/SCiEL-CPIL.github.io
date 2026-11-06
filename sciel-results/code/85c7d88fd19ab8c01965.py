import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # Precompute all possible triples
    all_triples = list(combinations(range(16), 3))
    n_all = len(all_triples)
    
    # For each triple, precompute the list of "conflict sets" - 
    # the other 7 triples that would complete a K(2,2,2)
    # For triple (x,y,z), a conflict is determined by (x',y',z') all distinct from x,y,z and each other
    # The 7 other transversals are:
    # (x',y,z), (x,y',z), (x,y,z'), (x',y',z), (x',y,z'), (x,y',z'), (x',y',z')
    
    # Precompute for each triple the list of conflict signatures
    # A conflict signature is a frozenset of 7 triples
    # But that's too memory intensive. Instead, we'll use a different approach.
    
    # Better approach: For each triple T=(x,y,z), precompute all (x',y',z') tuples
    # where all 6 points are distinct. For each such tuple, the 7 other transversals
    # form a "required set". T can be added only if NOT all 7 are present.
    
    # Even better: Use a counter-based approach.
    # For each triple, maintain a count of how many "conflict groups" are "satisfied" 
    # (i.e., all 7 other transversals are present). If count > 0, adding this triple 
    # would create a violation.
    
    # Actually, let's think differently. The constraint is:
    # For any 6 distinct points {a,a',b,b',c,c'}, at most 7 of the 8 transversals 
    # (x,y,z) with x in {a,a'}, y in {b,b'}, z in {c,c'} can be present.
    
    # When we add triple T=(x,y,z), it participates in multiple such 6-point groups.
    # Specifically, for each choice of x' != x,y,z, y' != x,y,z,x', z' != x,y,z,x',y',
    # the group {x,x',y,y',z,z'} has T as one of its 8 transversals.
    # Adding T creates a violation iff for some such group, the other 7 transversals 
    # are all already present.
    
    # Efficient check: Precompute for each triple T the list of "witness" sets.
    # A witness for T is a set of 7 triples. T causes a violation iff any witness 
    # is fully contained in the current system.
    
    # Number of witnesses per triple: 13*12*11 = 1716. Too many to check naively.
    
    # Alternative: Use a different data structure.
    # For each pair of points (p,q), track which triples contain both p and q.
    # When checking if adding T=(x,y,z) violates, we need to check if there exist
    # x',y',z' such that all 7 other transversals are present.
    
    # Let's use a smarter check:
    # For T=(x,y,z), a violation exists iff there exist x',y',z' (all distinct from 
    # x,y,z and each other) such that:
    # (x',y,z) in S, (x,y',z) in S, (x,y,z') in S, (x',y',z) in S,
    # (x',y,z') in S, (x,y',z') in S, (x',y',z') in S
    
    # We can iterate over x' (13 choices), then for each x', iterate over y' (12 choices),
    # and check if the remaining conditions can be satisfied.
    
    # Actually, let's precompute for each ordered pair (p,q) the set of points r such that
    # (p,q,r) is in the system. This is a 16x16 array of sets.
    
    # Let's use a bitset approach for speed.
    # For each pair (i,j) with i<j, maintain a 16-bit mask of points k such that (i,j,k) is in S.
    
    def check_violation_fast(triple, pair_masks):
        """
        Check if adding triple would create a K(2,2,2) violation.
        pair_masks[i][j] is a bitmask of points k such that sorted triple (i,j,k) is in system.
        """
        x, y, z = triple
        # Ensure x < y < z for canonical form
        # Actually, let's just work with the given triple
        
        for x_prime in range(16):
            if x_prime == x or x_prime == y or x_prime == z:
                continue
            for y_prime in range(16):
                if y_prime in (x, y, z, x_prime):
                    continue
                # Check if (x_prime, y, z) is in system
                a, b, c = sorted((x_prime, y, z))
                if not (pair_masks[a][b] >> c) & 1:
                    continue
                # Check if (x, y_prime, z) is in system
                a, b, c = sorted((x, y_prime, z))
                if not (pair_masks[a][b] >> c) & 1:
                    continue
                # Check if (x_prime, y_prime, z) is in system
                a, b, c = sorted((x_prime, y_prime, z))
                if not (pair_masks[a][b] >> c) & 1:
                    continue
                
                # Now we need to find z_prime such that:
                # (x, y, z_prime), (x_prime, y, z_prime), (x, y_prime, z_prime), (x_prime, y_prime, z_prime) are all in S
                # And z_prime not in {x, y, z, x_prime, y_prime}
                
                # For each candidate z_prime, check all 4 conditions
                for z_prime in range(16):
                    if z_prime in (x, y, z, x_prime, y_prime):
                        continue
                    # Check (x, y, z_prime)
                    a, b, c = sorted((x, y, z_prime))
                    if not (pair_masks[a][b] >> c) & 1:
                        continue
                    # Check (x_prime, y, z_prime)
                    a, b, c = sorted((x_prime, y, z_prime))
                    if not (pair_masks[a][b] >> c) & 1:
                        continue
                    # Check (x, y_prime, z_prime)
                    a, b, c = sorted((x, y_prime, z_prime))
                    if not (pair_masks[a][b] >> c) & 1:
                        continue
                    # Check (x_prime, y_prime, z_prime)
                    a, b, c = sorted((x_prime, y_prime, z_prime))
                    if not (pair_masks[a][b] >> c) & 1:
                        continue
                    return True
        return False
    
    def add_triple(triple, system, system_set, pair_masks):
        """Add triple to system and update pair_masks."""
        x, y, z = triple
        a, b, c = sorted((x, y, z))
        pair_masks[a][b] |= (1 << c)
        system_set.add(triple)
        system.append(list(triple))
    
    # Strategy: Multiple random greedy runs with the fast check
    best_system = None
    best_size = 0
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Initialize pair_masks
        pair_masks = [[0] * 16 for _ in range(16)]
        system = []
        system_set = set()
        
        # Random order
        order = list(all_triples)
        random.shuffle(order)
        
        for triple in order:
            if time.time() > deadline:
                break
            if not check_violation_fast(triple, pair_masks):
                add_triple(triple, system, system_set, pair_masks)
        
        if len(system) > best_size:
            best_size = len(system)
            best_system = [t[:] for t in system]
        
        # Early stop if we've done well
        if best_size > 350:
            break
    
    # Try to improve by adding remaining triples
    if best_system is not None and time.time() < deadline:
        # Rebuild pair_masks for best_system
        pair_masks = [[0] * 16 for _ in range(16)]
        current_set = set()
        for t in best_system:
            x, y, z = t
            a, b, c = sorted((x, y, z))
            pair_masks[a][b] |= (1 << c)
            current_set.add(tuple(t))
        
        # Try to add remaining triples
        remaining = [t for t in all_triples if t not in current_set]
        random.shuffle(remaining)
        
        for triple in remaining:
            if time.time() > deadline:
                break
            if not check_violation_fast(triple, pair_masks):
                add_triple(triple, best_system, current_set, pair_masks)
        
        best_size = len(best_system)
    
    # Fallback
    if best_system is None or best_size == 0:
        star_system = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        return star_system
    
    return best_system

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
