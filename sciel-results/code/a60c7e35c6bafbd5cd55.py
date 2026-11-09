import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 140

    # Fallback: Paley(67)
    def make_paley_prime_matrix(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        edge_matrix = [[False] * p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                edge_matrix[i][j] = diff in squares
        return edge_matrix

    def build_orientation_from_matrix(edge_matrix, n):
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                if edge_matrix[i][j]:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def fast_search(n, max_time):
        """
        Fast local search using bitmasks.
        """
        if n < 31:
            return None
        
        # Random initialization
        # out_masks[i] is a bitmask of vertices j such that i -> j
        out_masks = [0] * n
        # We need to represent the tournament state.
        # Since n <= 90, we can use Python integers (arbitrary precision) or NumPy uint64 if n<=64.
        # Let's use Python integers for simplicity and safety up to n=90.
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    out_masks[i] |= (1 << j)
                    out_masks[j] |= (1 << i) # Wait, tournament is antisymmetric.
                    # If i->j, then j does NOT point to i.
                    # So out_masks[i] has bit j. out_masks[j] does NOT have bit i.
                else:
                    out_masks[j] |= (1 << i)
                    # out_masks[i] does NOT have bit j.

        # Correction:
        out_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    out_masks[i] |= (1 << j)
                else:
                    out_masks[j] |= (1 << i)
        
        def count_bad():
            bad = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            # Check if any dominates the other 3
                            # i dominates: i->j, i->k, i->l
                            if (out_masks[i] & (1<<j) & (1<<k) & (1<<l)) or \
                               (out_masks[j] & (1<<i) & (1<<k) & (1<<l)) or \
                               (out_masks[k] & (1<<i) & (1<<j) & (1<<l)) or \
                               (out_masks[l] & (1<<i) & (1<<j) & (1<<k)):
                                continue
                            bad += 1
            return bad

        current_bad = count_bad()
        if current_bad == 0:
            return out_masks

        start_ls = time.time()
        
        # Precompute list of all pairs for random selection?
        # Or just random u, v.
        
        while time.time() - start_ls < max_time:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            # Check if flip improves
            # Affected sets: {u, v, a, b}
            # Current state of u->v:
            old_u_v = bool(out_masks[u] & (1 << v))
            
            delta = 0
            # Iterate over a < b, a,b != u,v
            # To be fast, we can iterate over all a, b and skip u,v
            # But checking 2000 sets per step is fast.
            
            for a in range(n):
                if a == u or a == v: continue
                for b in range(a+1, n):
                    if b == u or b == v: continue
                    
                    # Check if {u,v,a,b} is bad currently
                    is_bad_now = not (
                        (out_masks[u] & (1<<v) & (1<<a) & (1<<b)) or
                        (out_masks[v] & (1<<u) & (1<<a) & (1<<b)) or
                        (out_masks[a] & (1<<u) & (1<<v) & (1<<b)) or
                        (out_masks[b] & (1<<u) & (1<<v) & (1<<a))
                    )
                    
                    # Simulate flip
                    # New u->v is not old_u_v
                    new_u_v = not old_u_v
                    new_v_u = old_u_v # v->u is opposite of u->v? 
                    # Wait. In a tournament, if u->v, then v does not point to u.
                    # out_masks[u] has bit v. out_masks[v] does NOT have bit u.
                    # If we flip:
                    # If old_u_v was True (u->v), new is False (v->u).
                    # So out_masks[u] loses bit v. out_masks[v] gains bit u.
                    # If old_u_v was False (v->u), new is True (u->v).
                    # So out_masks[u] gains bit v. out_masks[v] loses bit u.
                    
                    # Check new bad status
                    # u dominates: new_u_v AND u->a AND u->b
                    u_dom = new_u_v and bool(out_masks[u] & (1<<a)) and bool(out_masks[u] & (1<<b))
                    # v dominates: new_v_u AND v->a AND v->b
                    # Note: v->a is bool(out_masks[v] & (1<<a)). This is unchanged.
                    v_dom = new_v_u and bool(out_masks[v] & (1<<a)) and bool(out_masks[v] & (1<<b))
                    # a dominates: a->u AND a->v AND a->b
                    # a->u is NOT (u->a). u->a is bool(out_masks[u] & (1<<a)). Unchanged.
                    # a->v is NOT (v->a). v->a is bool(out_masks[v] & (1<<a)). Unchanged.
                    # a->b is bool(out_masks[a] & (1<<b)). Unchanged.
                    a_dom = (not bool(out_masks[u] & (1<<a))) and (not bool(out_masks[v] & (1<<a))) and bool(out_masks[a] & (1<<b))
                    # b dominates: b->u AND b->v AND b->a
                    b_dom = (not bool(out_masks[u] & (1<<b))) and (not bool(out_masks[v] & (1<<b))) and bool(out_masks[b] & (1<<a))
                    
                    is_bad_new = not (u_dom or v_dom or a_dom or b_dom)
                    
                    if is_bad_now and not is_bad_new:
                        delta -= 1
                    elif not is_bad_now and is_bad_new:
                        delta += 1
                        # Early exit if delta > 0? 
                        # If we only accept delta <= 0, we can break if delta > 0.
                        # But we might want to accept delta == 0.
                        # So we must calculate full delta if we want to know if it's 0 or positive.
                        # However, if delta becomes > 0, we will reject.
                        # We can break early if delta > 0 to save time, assuming we will reject.
                        # But what if it goes back to 0? Unlikely but possible.
                        # For speed, let's break if delta > 0.
                        break
                if delta > 0:
                    break
            
            if delta <= 0:
                # Accept
                if old_u_v:
                    out_masks[u] &= ~(1 << v)
                    out_masks[v] |= (1 << u)
                else:
                    out_masks[u] |= (1 << v)
                    out_masks[v] &= ~(1 << u)
                
                current_bad += delta
                if current_bad == 0:
                    return out_masks
        
        return None

    # Try smaller n
    # Budget: 30s each for 66, 65, 64
    if time.time() - start_time < deadline - 10:
        res = fast_search(66, 30.0)
        if res:
            # Convert out_masks to orientation list
            orientation = []
            for i in range(66):
                for j in range(i + 1, 66):
                    if res[i] & (1 << j):
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': 66, 'orientation': orientation}

    if time.time() - start_time < deadline - 10:
        res = fast_search(65, 30.0)
        if res:
            orientation = []
            for i in range(65):
                for j in range(i + 1, 65):
                    if res[i] & (1 << j):
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': 65, 'orientation': orientation}

    if time.time() - start_time < deadline - 10:
        res = fast_search(64, 30.0)
        if res:
            orientation = []
            for i in range(64):
                for j in range(i + 1, 64):
                    if res[i] & (1 << j):
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': 64, 'orientation': orientation}

    # Fallback to Paley(67)
    p = 67
    edge_matrix = make_paley_prime_matrix(p)
    orientation = build_orientation_from_matrix(edge_matrix, p)
    return {'n': p, 'orientation': orientation}
