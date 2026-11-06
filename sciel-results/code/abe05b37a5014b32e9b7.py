import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s limit

    # Precompute all possible triples as sorted tuples
    all_triples = [tuple(t) for t in combinations(range(16), 3)]
    n_all = len(all_triples)
    
    # Triple index mapping for O(1) lookup
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute for each triple the set of "conflict witnesses"
    # A witness is a set of 7 triples that, together with the candidate, form K(2,2,2)
    # For triple T=(x,y,z), a witness is determined by x',y',z' all distinct from x,y,z and each other
    # The 7 other transversals are:
    # (x',y,z), (x,y',z), (x,y,z'), (x',y',z), (x',y,z'), (x,y',z'), (x',y',z')
    
    # We'll use a counter-based approach:
    # For each triple T, count how many witness sets are "fully satisfied" (all 7 other triples present)
    # T can be added only if count == 0
    
    # Actually, let's use a more efficient approach:
    # Maintain for each triple a bitmask of which witnesses are "close" to being satisfied
    # But that's complex. Let's use the direct check but optimize it.
    
    # Precompute for each pair (i,j) with i<j, the list of triples containing this pair
    pair_to_triples = {}
    for i in range(16):
        for j in range(i+1, 16):
            pair_to_triples[(i,j)] = []
    
    for t in all_triples:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = sorted((t[i], t[j]))
                pair_to_triples[(a,b)].append(t)
    
    # For fast membership testing, maintain a set of current triples
    # And for each pair, a set of the third points in triples containing that pair
    
    def check_violation(triple, present_set, pair_thirds):
        """
        Check if adding triple would create a K(2,2,2) violation.
        present_set: set of triples currently in system
        pair_thirds: dict from (a,b) to set of c such that (a,b,c) is in system (a<b)
        """
        x, y, z = triple
        
        # Try all x', y', z' distinct from x,y,z and each other
        for x_prime in range(16):
            if x_prime in (x, y, z):
                continue
            for y_prime in range(16):
                if y_prime in (x, y, z, x_prime):
                    continue
                for z_prime in range(16):
                    if z_prime in (x, y, z, x_prime, y_prime):
                        continue
                    
                    # Check all 7 other transversals
                    # (x_prime, y, z)
                    a, b, c = sorted((x_prime, y, z))
                    if (a, b, c) not in present_set:
                        continue
                    # (x, y_prime, z)
                    a, b, c = sorted((x, y_prime, z))
                    if (a, b, c) not in present_set:
                        continue
                    # (x, y, z_prime)
                    a, b, c = sorted((x, y, z_prime))
                    if (a, b, c) not in present_set:
                        continue
                    # (x_prime, y_prime, z)
                    a, b, c = sorted((x_prime, y_prime, z))
                    if (a, b, c) not in present_set:
                        continue
                    # (x_prime, y, z_prime)
                    a, b, c = sorted((x_prime, y, z_prime))
                    if (a, b, c) not in present_set:
                        continue
                    # (x, y_prime, z_prime)
                    a, b, c = sorted((x, y_prime, z_prime))
                    if (a, b, c) not in present_set:
                        continue
                    # (x_prime, y_prime, z_prime)
                    a, b, c = sorted((x_prime, y_prime, z_prime))
                    if (a, b, c) not in present_set:
                        continue
                    return True
        return False

    # Better approach: use bitmasks for pairs
    # pair_mask[i][j] is a 16-bit mask where bit k is set if (i,j,k) is in system, i<j
    
    def check_violation_fast(triple, pair_masks):
        """
        Check if adding triple would create a K(2,2,2) violation.
        pair_masks[i][j] is a 16-bit mask for i<j, bit k set if (i,j,k) in system.
        """
        x, y, z = triple
        
        for x_prime in range(16):
            if x_prime in (x, y, z):
                continue
            for y_prime in range(16):
                if y_prime in (x, y, z, x_prime):
                    continue
                # Check (x_prime, y, z)
                a, b = sorted((x_prime, y))
                c = z if z != a and z != b else (x if x != a and x != b else None)
                # This is getting messy. Let me just use sorted tuple lookup.
                
                # Actually, let me precompute a 3D array: present[i][j][k] for i<j<k
                # That's simpler.
        
        return False

    # Let me use a 3D boolean array for O(1) lookup
    # present[i][j][k] = True if triple (i,j,k) with i<j<k is in system
    
    def check_violation_3d(triple, present):
        """
        Check if adding triple would create a K(2,2,2) violation.
        present is a 16x16x16 boolean array where present[i][j][k] is True 
        only for i<j<k and the triple is in the system.
        """
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
                    
                    # Helper to check if triple (a,b,c) is in system
                    def in_system(a, b, c):
                        i, j, k = sorted((a, b, c))
                        return present[i][j][k]
                    
                    if not in_system(x_prime, y, z):
                        continue
                    if not in_system(x, y_prime, z):
                        continue
                    if not in_system(x, y, z_prime):
                        continue
                    if not in_system(x_prime, y_prime, z):
                        continue
                    if not in_system(x_prime, y, z_prime):
                        continue
                    if not in_system(x, y_prime, z_prime):
                        continue
                    if not in_system(x_prime, y_prime, z_prime):
                        continue
                    return True
        return False

    # This is still slow. Let me think of a better approach.
    # 
    # Key insight: For a fixed triple T=(x,y,z), a violation occurs iff there exist
    # x', y', z' such that all 7 other transversals are present.
    # 
    # We can restructure: For each pair (x', y'), track which z' values have
    # (x', y', z') in the system. Similarly for other pairs.
    #
    # Actually, let me just use the random greedy approach but with multiple restarts
    # and a more efficient check. The check above is O(13*12*11) = 1716 checks per triple,
    # each with 7 lookups. That's ~12k operations per triple check. For 560 triples,
    # that's ~7M operations per greedy pass. Should be fine in Python with numpy.
    
    import numpy as np
    
    # Use numpy for the present array for faster access
    def construct_with_numpy():
        start_time = time.time()
        deadline = start_time + 100.0
        
        all_triples = [tuple(t) for t in combinations(range(16), 3)]
        
        # present[i][j][k] for i<j<k
        # Use a set for simplicity, it's fast enough
        
        best_system = None
        best_size = 0
        
        num_restarts = 50
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            present = set()
            system = []
            
            order = all_triples[:]
            random.shuffle(order)
            
            for triple in order:
                if time.time() > deadline:
                    break
                x, y, z = triple
                violation = False
                
                for x_prime in range(16):
                    if violation:
                        break
                    if x_prime in (x, y, z):
                        continue
                    for y_prime in range(16):
                        if violation:
                            break
                        if y_prime in (x, y, z, x_prime):
                            continue
                        for z_prime in range(16):
                            if z_prime in (x, y, z, x_prime, y_prime):
                                continue
                            
                            # Check 7 transversals
                            t1 = tuple(sorted((x_prime, y, z)))
                            t2 = tuple(sorted((x, y_prime, z)))
                            t3 = tuple(sorted((x, y, z_prime)))
                            t4 = tuple(sorted((x_prime, y_prime, z)))
                            t5 = tuple(sorted((x_prime, y, z_prime)))
                            t6 = tuple(sorted((x, y_prime, z_prime)))
                            t7 = tuple(sorted((x_prime, y_prime, z_prime)))
                            
                            if all(t in present for t in (t1,t2,t3,t4,t5,t6,t7)):
                                violation = True
                                break
                
                if not violation:
                    present.add(triple)
                    system.append(triple)
            
            if len(system) > best_size:
                best_size = len(system)
                best_system = system[:]
        
        # Try to improve best system by adding more triples
        if best_system and time.time() < deadline:
            present = set(best_system)
            remaining = [t for t in all_triples if t not in present]
            random.shuffle(remaining)
            
            for triple in remaining:
                if time.time() > deadline:
                    break
                x, y, z = triple
                violation = False
                
                for x_prime in range(16):
                    if violation:
                        break
                    if x_prime in (x, y, z):
                        continue
                    for y_prime in range(16):
                        if violation:
                            break
                        if y_prime in (x, y, z, x_prime):
                            continue
                        for z_prime in range(16):
                            if z_prime in (x, y, z, x_prime, y_prime):
                                continue
                            
                            t1 = tuple(sorted((x_prime, y, z)))
                            t2 = tuple(sorted((x, y_prime, z)))
                            t3 = tuple(sorted((x, y, z_prime)))
                            t4 = tuple(sorted((x_prime, y_prime, z)))
                            t5 = tuple(sorted((x_prime, y, z_prime)))
                            t6 = tuple(sorted((x, y_prime, z_prime)))
                            t7 = tuple(sorted((x_prime, y_prime, z_prime)))
                            
                            if all(t in present for t in (t1,t2,t3,t4,t5,t6,t7)):
                                violation = True
                                break
                
                if not violation:
                    present.add(triple)
                    best_system.append(triple)
        
        if best_system is None:
            return [[0, a, b] for a in range(1, 16) for b in range(a+1, 16)]
        
        return [list(t) for t in best_system]
    
    return construct_with_numpy()
