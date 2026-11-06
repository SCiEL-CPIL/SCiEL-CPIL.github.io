import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110.0

    def elapsed():
        return time.time() - start_time

    def build_paley(p):
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return p, orientation

    def verify(tournament, n):
        # Check if every 4-set has a common dominator
        # tournament is a list of bits for pairs (i,j) i<j
        # Build adjacency matrix for speed
        adj = np.zeros((n, n), dtype=bool)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if tournament[idx] == 0:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
                idx += 1
        
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            found = False
            for u in range(n):
                if u in quad:
                    continue
                if adj[u, a] and adj[u, b] and adj[u, c] and adj[u, d]:
                    found = True
                    break
            if not found:
                return False
        return True

    # Initial construction
    p_init, orientation = build_paley(67)
    n_orig = p_init
    adj_orig = np.zeros((n_orig, n_orig), dtype=bool)
    idx = 0
    for i in range(n_orig):
        for j in range(i+1, n_orig):
            if orientation[idx] == 0:
                adj_orig[i, j] = True
            else:
                adj_orig[j, i] = True
            idx += 1

    active = list(range(n_orig))
    
    while elapsed() < DEADLINE:
        m = len(active)
        if m < 5:
            break
            
        # Map active indices to 0..m-1 for local processing
        # But we can just work with the original indices in active list
        # We need to determine which vertices in 'active' are critical.
        
        # A vertex v is critical if it is the unique dominator of some 4-set S in active.
        critical = [False] * m
        
        # Iterate over all 4-sets in active
        # To speed up, we can break early if time is running out
        quads = list(combinations(range(m), 4))
        
        # Optimization: Process quads in chunks or just iterate
        # For each quad, find dominators in active.
        
        for quad_idx in range(len(quads)):
            if elapsed() > DEADLINE - 5: # Safety margin
                break
                
            a_idx, b_idx, c_idx, d_idx = quads[quad_idx]
            a = active[a_idx]
            b = active[b_idx]
            c = active[c_idx]
            d = active[d_idx]
            
            dominators = []
            for i_idx in range(m):
                if i_idx == a_idx or i_idx == b_idx or i_idx == c_idx or i_idx == d_idx:
                    continue
                u = active[i_idx]
                if adj_orig[u, a] and adj_orig[u, b] and adj_orig[u, c] and adj_orig[u, d]:
                    dominators.append(i_idx)
                    if len(dominators) > 1:
                        break # More than one dominator, no vertex is unique
            
            if len(dominators) == 1:
                critical[dominators[0]] = True
                
        # If all are critical, we are at a local minimum (or stuck)
        if all(critical):
            break
            
        # Remove a non-critical vertex
        # It's better to remove vertices that are "least" critical?
        # Or just the first one found.
        # Removing more vertices is better for the score.
        # We can remove ALL non-critical vertices at once?
        # No, removing one might make another non-critical (or critical).
        # Actually, if v is non-critical, it means it's not the UNIQUE dominator of any 4-set.
        # If we remove v, do we create new unique dominators?
        # If S had dominators {v, u}, and we remove v, u becomes unique.
        # So u becomes critical.
        # So we should remove them one by one, or carefully.
        # However, if we remove all non-critical vertices, we might invalidate the condition
        # for sets that had {v, u} as dominators. u would become unique, which is fine,
        # BUT we removed v, so u is still there.
        # Wait. If S had dominators {v, u}. v is non-critical (so it's not unique for ANY set).
        # u is non-critical.
        # If we remove v, S still has u.
        # If we remove u, S still has v.
        # If we remove BOTH, S has NO dominator.
        # So we cannot remove all non-critical vertices at once.
        
        # Greedy removal: pick one non-critical vertex.
        removable_idx = -1
        for i in range(m):
            if not critical[i]:
                removable_idx = i
                break
                
        if removable_idx == -1:
            break
            
        active.pop(removable_idx)
        
    # Construct final orientation
    n_final = len(active)
    final_orientation = []
    for i in range(n_final):
        for j in range(i+1, n_final):
            orig_i = active[i]
            orig_j = active[j]
            if adj_orig[orig_i, orig_j]:
                final_orientation.append(0)
            else:
                final_orientation.append(1)
                
    # Final verification to be safe
    if not verify(final_orientation, n_final):
        # Fallback to full Paley
        return {'n': 67, 'orientation': orientation}

    return {'n': n_final, 'orientation': final_orientation}
