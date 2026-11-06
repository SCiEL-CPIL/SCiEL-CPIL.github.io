import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Construct Paley 67
    p = 67
    squares = {x * x % p for x in range(1, p)}
    
    # masks[i] is bitmask of j such that i -> j
    masks = [0] * p
    # doms[i] is bitmask of j such that j -> i
    doms = [0] * p
    
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                # i -> j
                masks[i] |= (1 << j)
                doms[j] |= (1 << i)
            else:
                # j -> i
                masks[j] |= (1 << i)
                doms[i] |= (1 << j)
                
    n = p
    
    # Helper to count violations
    def count_violations(masks, doms, n):
        violations = 0
        # Iterate over all 4-combinations
        # Optimization: use local vars
        d = doms
        for a in range(n):
            da = d[a]
            for b in range(a + 1, n):
                dab = da & d[b]
                if dab == 0: continue
                for c in range(b + 1, n):
                    dabc = dab & d[c]
                    if dabc == 0: continue
                    for d_idx in range(c + 1, n):
                        dabcd = dabc & d[d_idx]
                        # Check if any dominator is outside {a,b,c,d}
                        # mask of S
                        s_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d_idx)
                        if (dabcd & ~s_mask) == 0:
                            violations += 1
        return violations

    # Helper to remove vertex v
    def remove_vertex(masks, doms, n, v):
        new_n = n - 1
        new_masks = [0] * new_n
        new_doms = [0] * new_n
        
        # Mapping from old index to new index
        # old i -> new i if i < v, i-1 if i > v
        # We just rebuild by iterating old vertices except v
        
        old_to_new = {}
        k = 0
        for i in range(n):
            if i != v:
                old_to_new[i] = k
                k += 1
                
        for old_i in range(n):
            if old_i == v: continue
            new_i = old_to_new[old_i]
            
            # Build new masks[new_i]
            m = 0
            d = 0
            old_mask = masks[old_i]
            old_dom = doms[old_i]
            
            # Iterate over all old j != old_i, j != v
            # Actually, we can just iterate new indices and map back?
            # Easier: iterate old j, if j != v and j != old_i, map to new_j
            for old_j in range(n):
                if old_j == v or old_j == old_i: continue
                new_j = old_to_new[old_j]
                # Check edge old_i -> old_j
                if (old_mask >> old_j) & 1:
                    m |= (1 << new_j)
                else:
                    # old_j -> old_i
                    d |= (1 << new_j)
                    
            new_masks[new_i] = m
            new_doms[new_i] = d
            
        return new_masks, new_doms, new_n

    # Helper to flip edge u, v (u, v in 0..n-1)
    def flip_edge(masks, doms, u, v):
        # Toggle u -> v to v -> u
        # If u -> v currently:
        #   masks[u] has v, doms[v] has u
        #   masks[v] no u, doms[u] no v
        # After flip:
        #   masks[u] no v, doms[v] no u
        #   masks[v] has u, doms[u] has v
        
        bit_u = 1 << u
        bit_v = 1 << v
        
        # Check current state
        if (masks[u] >> v) & 1:
            # u -> v
            masks[u] &= ~bit_v
            doms[v] &= ~bit_u
            masks[v] |= bit_u
            doms[u] |= bit_v
        else:
            # v -> u
            masks[v] &= ~bit_u
            doms[u] &= ~bit_v
            masks[u] |= bit_v
            doms[v] |= bit_u

    # Initial check for n=67
    # It should be 0.
    # v_count = count_violations(masks, doms, n)
    # if v_count != 0: return fallback
    
    best_n = n
    best_masks = masks
    best_doms = doms
    
    # Try to reduce n
    # We will try to go down from 67 to 5
    # But we stop if we can't reduce further
    
    while n > 5:
        if time.time() > deadline - 10: break
        
        reduced = False
        # Try removing a few random vertices
        for attempt in range(5):
            if time.time() > deadline - 5: break
            
            v_to_remove = random.randint(0, n - 1)
            
            new_masks, new_doms, new_n = remove_vertex(masks, doms, n, v_to_remove)
            
            v_count = count_violations(new_masks, new_doms, new_n)
            
            if v_count == 0:
                # Success
                n = new_n
                masks = new_masks
                doms = new_doms
                best_n = n
                best_masks = masks
                best_doms = doms
                reduced = True
                break # Try to reduce further
            
            # Local search on new_n
            # We have v_count violations
            # Try to fix them
            current_masks = new_masks
            current_doms = new_doms
            current_v = v_count
            best_local_masks = new_masks
            best_local_doms = new_doms
            best_local_v = v_count
            
            steps = 0
            max_steps = 50 # Limit local search time
            
            while steps < max_steps and current_v > 0:
                if time.time() > deadline - 5: break
                
                # Pick random edge
                u = random.randint(0, new_n - 1)
                v = random.randint(0, new_n - 1)
                if u == v: continue
                
                # Flip
                flip_edge(current_masks, current_doms, u, v)
                
                # Recount
                new_v = count_violations(current_masks, current_doms, new_n)
                
                if new_v < current_v:
                    current_v = new_v
                    if new_v < best_local_v:
                        best_local_v = new_v
                        best_local_masks = [m for m in current_masks]
                        best_local_doms = [d for d in current_doms]
                else:
                    # Revert
                    flip_edge(current_masks, current_doms, u, v)
                
                steps += 1
                
            if best_local_v == 0:
                # Found a valid tournament on new_n
                n = new_n
                masks = best_local_masks
                doms = best_local_doms
                best_n = n
                best_masks = masks
                best_doms = doms
                reduced = True
                break # Try to reduce further
            else:
                # Failed to reduce, revert to original n
                # masks and doms are still the original valid ones
                pass
        
        if not reduced:
            # Could not reduce further with current strategy
            # Break to save time or try different strategy?
            # For now, break.
            break

    # Construct output
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if (masks[i] >> j) & 1:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': n, 'orientation': orientation}
