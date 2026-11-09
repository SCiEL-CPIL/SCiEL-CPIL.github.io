import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_n = 0
    best_edges = []

    # Try n=47 first, then 46, then 45...
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        if min_deg_needed < 0:
            min_deg_needed = 0
            
        found = False
        
        # Adjust restarts and probability based on n
        # For n=47, min_deg=8. p ~ 0.25
        # For n=46, min_deg=7. p ~ 0.25
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 100 + target_n)
            
            # M[i] is a bitmask of neighbors of i
            M = [0] * target_n
            deg = [0] * target_n
            
            # Initialize random graph
            # Target average degree slightly above min_deg_needed to have slack
            # p = (min_deg_needed + 3) / target_n
            p = (min_deg_needed + 3.0) / target_n
            if p > 0.5: p = 0.5 # Cap it
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p:
                        M[u] |= (1 << v)
                        M[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
            
            # Check initial validity
            # If min_deg is already too low, we might need to add edges immediately
            # But let's run the repair loop.
            
            success = False
            stuck_count = 0
            
            for step in range(10000): # Max steps
                if time.time() > deadline:
                    break
                    
                # 1. Check if we are done
                if min(deg) >= min_deg_needed:
                    # Verify C4-free
                    c4_free = True
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if (M[i] & M[j]).bit_count() > 1:
                                c4_free = False
                                break
                        if not c4_free:
                            break
                    
                    if c4_free:
                        success = True
                        break
                
                # 2. If min degree is too low, try to add an edge
                if min(deg) < min_deg_needed:
                    low_vertices = [i for i in range(target_n) if deg[i] < min_deg_needed]
                    u = random.choice(low_vertices)
                    
                    # Find a valid v to add
                    added = False
                    # Shuffle candidates for diversity
                    candidates = [v for v in range(target_n) if v != u and not (M[u] & (1 << v))]
                    random.shuffle(candidates)
                    
                    for v in candidates:
                        if can_add(u, v, M):
                            # Add edge
                            M[u] |= (1 << v)
                            M[v] |= (1 << u)
                            deg[u] += 1
                            deg[v] += 1
                            added = True
                            break
                            
                    if not added:
                        # Can't add any edge to low vertex. Stuck?
                        # Maybe we need to remove an edge elsewhere to create space?
                        # For now, restart if this happens too often
                        stuck_count += 1
                        if stuck_count > 100:
                            break
                        continue
                else:
                    stuck_count = 0
                    
                # 3. If min degree is OK, but C4s exist, remove edges
                # Find a C4
                c4_found = False
                bad_u = -1
                bad_v = -1
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        common = M[i] & M[j]
                        if common.bit_count() > 1:
                            c4_found = True
                            bad_u = i
                            bad_v = j
                            break
                    if c4_found:
                        break
                
                if not c4_found:
                    # No C4s, but min degree check failed?
                    # This shouldn't happen if we check min(deg) first, 
                    # unless min(deg) >= min_deg_needed but we didn't break?
                    # Ah, if min(deg) >= min_deg_needed and c4_free, we break in step 1.
                    # So if we are here, min(deg) < min_deg_needed OR c4 exists.
                    # If min(deg) >= min_deg_needed and c4 exists, we are in this branch.
                    # If min(deg) < min_deg_needed, we are in the previous branch.
                    # Wait, if min(deg) >= min_deg_needed and c4 exists, we go to step 3.
                    # If min(deg) < min_deg_needed, we go to step 2.
                    # So if we are here, min(deg) >= min_deg_needed and c4 exists.
                    pass

                if c4_found:
                    # We have a C4 involving bad_u, bad_v.
                    # Common neighbors: C = M[bad_u] & M[bad_v]
                    C = M[bad_u] & M[bad_v]
                    # We need to reduce |C| to <= 1.
                    # We can remove edge (bad_u, c) or (bad_v, c) for c in C.
                    # Or remove edge (bad_u, bad_v) if it exists? No, C4 is u-c1-v-c2-u.
                    # The edges are (u, c1), (c1, v), (v, c2), (c2, u).
                    # Removing any of these breaks this specific C4.
                    # But we need to break ALL C4s.
                    
                    # Strategy: Pick a random common neighbor c from C.
                    # Try to remove (bad_u, c) or (bad_v, c).
                    # Check if removal keeps deg >= min_deg_needed.
                    
                    # Get list of bits in C
                    c_list = []
                    mask = C
                    while mask:
                        lsb = mask & -mask
                        c = lsb.bit_length() - 1
                        c_list.append(c)
                        mask ^= lsb
                    
                    random.shuffle(c_list)
                    
                    removed = False
                    for c in c_list:
                        # Try removing (bad_u, c)
                        if deg[bad_u] > min_deg_needed:
                            # Check if removing (bad_u, c) is safe?
                            # It reduces deg[bad_u] and deg[c].
                            # It might create new issues? No, removing edges doesn't create C4s.
                            # So it's always "safe" for C4-freeness.
                            # Only constraint is degree.
                            M[bad_u] &= ~(1 << c)
                            M[c] &= ~(1 << bad_u)
                            deg[bad_u] -= 1
                            deg[c] -= 1
                            removed = True
                            break
                            
                        # Try removing (bad_v, c)
                        if deg[bad_v] > min_deg_needed:
                            M[bad_v] &= ~(1 << c)
                            M[c] &= ~(1 << bad_v)
                            deg[bad_v] -= 1
                            deg[c] -= 1
                            removed = True
                            break
                            
                    if not removed:
                        # Cannot remove any edge without dropping degree below min.
                        # Stuck.
                        stuck_count += 1
                        if stuck_count > 100:
                            break
                        continue
                else:
                    # No C4 found? But we didn't break in step 1.
                    # This implies min(deg) < min_deg_needed was false, so min(deg) >= min_deg_needed.
                    # And c4_found is false.
                    # So it should have broken in step 1.
                    # Unless there's a bug.
                    break

            if success:
                # Final verification
                c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (M[i] & M[j]).bit_count() > 1:
                            c4_free = False
                            break
                    if not c4_free:
                        break
                
                if c4_free and min(deg) >= min_deg_needed:
                    final_edges = []
                    for i in range(target_n):
                        mask = M[i]
                        while mask:
                            lsb = mask & -mask
                            j = lsb.bit_length() - 1
                            if i < j:
                                final_edges.append([i, j])
                            mask ^= lsb
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break # Break restart loop
        
        if best_n == 47:
            break
        if found:
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: n=39, empty graph is valid (min deg 0 >= 0)
        return {'n': 39, 'edges': []}

def can_add(u, v, M):
    if M[u] & (1 << v):
        return False
    common = M[u] & M[v]
    if common.bit_count() > 1:
        return False
    # Check C4 creation
    # D2 logic from previous program was good.
    # But we don't have D2 here.
    # We can calculate it on the fly?
    # N(u) & (Union of N(w) for w in N(v))
    # This is O(deg(u)*deg(v)) bit ops.
    # For n=47, deg~10, 100 ops. Fine.
    
    # Union of N(w) for w in N(v)
    union_v = 0
    mask_v = M[v]
    while mask_v:
        lsb = mask_v & -mask_v
        w = lsb.bit_length() - 1
        union_v |= M[w]
        mask_v ^= lsb
        
    if M[u] & union_v:
        return False
        
    union_u = 0
    mask_u = M[u]
    while mask_u:
        lsb = mask_u & -mask_u
        w = lsb.bit_length() - 1
        union_u |= M[w]
        mask_u ^= lsb
        
    if M[v] & union_u:
        return False
        
    return True
