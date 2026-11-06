import time
import numpy as np
import random

def _get_paleyn_orientation(n):
    if n < 2:
        return None
    is_prime = True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            is_prime = False
            break
    if not is_prime:
        return None
    if n % 4 != 3:
        return None
        
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
        
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                orientation.append(0) 
            else:
                orientation.append(1) 
    return orientation

def _orientation_to_in_bits(n, orientation):
    in_bits = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                in_bits[j] |= (1 << i)
            else:
                in_bits[i] |= (1 << j)
            idx += 1
    return in_bits

def _in_bits_to_orientation(n, in_bits):
    orientation = [0] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (in_bits[j] >> i) & 1:
                orientation[idx] = 0
            else:
                orientation[idx] = 1
            idx += 1
    return orientation

def _validate_tournament(n, in_bits):
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d] & ~mask
                    if common == 0:
                        return False
    return True

def _prune_tournament(n, in_bits):
    """
    Iteratively removes vertices that are not unique dominators of any 4-set.
    Returns (new_n, new_in_bits).
    """
    active = list(range(n))
    current_in_bits = in_bits[:]
    current_n = n
    
    while True:
        if not active:
            break
            
        # Calculate unique dominator counts for active vertices
        # unique_count[v] = number of 4-sets in active where v is the ONLY dominator
        unique_count = {v: 0 for v in active}
        
        # Iterate all 4-subsets of active
        # Optimization: Use combinations from itertools? 
        # For n=67, C(67,4) is ~ 700,000. This is fast enough in Python if we don't do it too many times.
        # But we might iterate the while loop several times.
        
        # Let's use a list of active vertices for iteration
        act = active
        len_act = len(act)
        
        # To speed up, we can precompute masks? No, just iterate.
        for i1 in range(len_act):
            a = act[i1]
            for i2 in range(i1 + 1, len_act):
                b = act[i2]
                for i3 in range(i2 + 1, len_act):
                    c = act[i3]
                    for i4 in range(i3 + 1, len_act):
                        d = act[i4]
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        common = current_in_bits[a] & current_in_bits[b] & current_in_bits[c] & current_in_bits[d] & ~mask
                        # common contains dominators from the FULL set, but we only care about active ones?
                        # No, current_in_bits is updated to only reflect edges between active vertices?
                        # If we just remove vertices from the list but keep the bits, the bits for removed vertices are still there.
                        # We must ensure 'common' only counts active vertices.
                        active_mask = 0
                        for v in act:
                            active_mask |= (1 << v)
                        common &= active_mask
                        
                        if common == 0:
                            # This 4-set has no dominator in the active set.
                            # This means the tournament is invalid.
                            # But we started with a valid one and only removed non-essential vertices.
                            # So this should not happen.
                            pass
                        else:
                            # Count bits
                            # If count is 1, that vertex is the unique dominator for this set.
                            cnt = bin(common).count('1')
                            if cnt == 1:
                                v = common.bit_length() - 1
                                if v in unique_count:
                                    unique_count[v] += 1
                            # If cnt > 1, no vertex is the unique dominator for this set.
        
        # Find a vertex with unique_count 0
        removable = None
        for v in active:
            if unique_count[v] == 0:
                removable = v
                break
                
        if removable is None:
            break
            
        # Remove 'removable' from active
        active.remove(removable)
        
        # Update current_in_bits?
        # We don't strictly need to update the bits if we always mask with active_mask.
        # But for efficiency, we could clear the bits.
        # For now, masking is sufficient.

    # Rebuild in_bits for the final active set
    final_active = sorted(active)
    new_n = len(final_active)
    if new_n == 0:
        return 0, []
        
    # Map old indices to new indices
    idx_map = {v: i for i, v in enumerate(final_active)}
    
    new_in_bits = [0] * new_n
    for i in range(new_n):
        old_i = final_active[i]
        for j in range(new_n):
            if i == j:
                continue
            old_j = final_active[j]
            # Check if old_i -> old_j in original tournament
            # i -> j means j is in In[i], so (in_bits[i] >> j) & 1
            # Wait, in_bits[x] is bitmask of vertices pointing TO x.
            # So i -> j means i is in In[j], so (in_bits[j] >> i) & 1
            if (current_in_bits[old_j] >> old_i) & 1:
                new_in_bits[i] |= (1 << j) # j points to i? No.
                # Let's re-verify.
                # in_bits[k] has bit l set if l -> k.
                # We want new_in_bits[i] to have bit j set if final_active[j] -> final_active[i].
                # i.e., old_j -> old_i.
                # This is true if (current_in_bits[old_i] >> old_j) & 1.
                # So:
                pass
                
    # Correct logic:
    # new_in_bits[i] represents vertices pointing to final_active[i].
    # A vertex final_active[j] points to final_active[i] iff old_j -> old_i.
    # old_j -> old_i is true if (current_in_bits[old_i] >> old_j) & 1.
    
    new_in_bits = [0] * new_n
    for i in range(new_n):
        old_i = final_active[i]
        for j in range(new_n):
            if i == j:
                continue
            old_j = final_active[j]
            if (current_in_bits[old_i] >> old_j) & 1:
                new_in_bits[i] |= (1 << j)
                
    return new_n, new_in_bits

def _try_remove_vertex(n, in_bits):
    """
    Tries to remove one vertex from a valid tournament.
    Returns (new_n, new_in_bits) if a vertex can be removed, else (n, in_bits).
    """
    for v in range(n):
        # Check if removing v leaves a valid tournament
        # We need to check all 4-sets in {0..n-1} \ {v}
        # This is expensive: O(n * C(n, 4))
        # But n is ~60, so it's ~ 60 * 400,000 = 24M checks.
        # Each check is a few bit ops. Should be < 1s in Python? Maybe 2-3s.
        # We have 150s. We can try this.
        
        # Optimization: Only check 4-sets that might be affected?
        # Removing v doesn't change the edges between other vertices.
        # So a 4-set S is valid in T\{v} iff S is valid in T AND v is not in S?
        # No. S is a set of 4 vertices from T\{v}.
        # S is valid in T\{v} iff there exists a dominator d in T\{v} such that d->s for all s in S.
        # If S was valid in T with dominator d, and d != v, then S is still valid in T\{v}.
        # If S was valid in T ONLY with dominator v, then S is NOT valid in T\{v} (unless another dominator exists in T\{v}).
        # So we only need to check 4-sets S such that v was a dominator of S in T.
        
        # Let's find all 4-sets S where v is a dominator.
        # v is a dominator of S if v->a, v->b, v->c, v->d for all a,b,c,d in S.
        # This means a,b,c,d are in Out(v).
        # So we need to check all 4-subsets of Out(v).
        # If any such 4-subset has no other dominator in T\{v}, then removing v invalidates it.
        
        out_v = []
        for u in range(n):
            if u == v:
                continue
            # Check if v -> u
            # v -> u means u is in In[v], so (in_bits[v] >> u) & 1
            if (in_bits[v] >> u) & 1:
                out_v.append(u)
                
        # Iterate 4-subsets of out_v
        len_out = len(out_v)
        if len_out < 4:
            continue
            
        valid_removal = True
        for i1 in range(len_out):
            a = out_v[i1]
            for i2 in range(i1 + 1, len_out):
                b = out_v[i2]
                for i3 in range(i2 + 1, len_out):
                    c = out_v[i3]
                    for i4 in range(i3 + 1, len_out):
                        d = out_v[i4]
                        # Check if {a,b,c,d} has a dominator other than v
                        # Dominator must be in T\{v}
                        # Common dominators of {a,b,c,d} in T:
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d] & ~mask
                        # Remove v from common
                        common &= ~(1 << v)
                        if common == 0:
                            # No other dominator
                            valid_removal = False
                            break
                        if not valid_removal:
                            break
                        if not valid_removal:
                            break
                        if not valid_removal:
                            break
                            
        if valid_removal:
            # We can remove v
            # Construct new tournament
            active = [u for u in range(n) if u != v]
            new_n = n - 1
            idx_map = {u: i for i, u in enumerate(active)}
            new_in_bits = [0] * new_n
            for i in range(new_n):
                old_i = active[i]
                for j in range(new_n):
                    if i == j:
                        continue
                    old_j = active[j]
                    if (in_bits[old_i] >> old_j) & 1:
                        new_in_bits[i] |= (1 << j)
            return new_n, new_in_bits
            
    return n, in_bits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Get Paley 67
    orientation = _get_paleyn_orientation(67)
    if orientation is None:
        # Fallback
        return {'n': 67, 'orientation': [0]* (67*66//2)}
        
    n = 67
    in_bits = _orientation_to_in_bits(n, orientation)
    
    # 2. Prune non-essential vertices
    n, in_bits = _prune_tournament(n, in_bits)
    
    # 3. Try to remove more vertices
    # We can try to remove vertices one by one.
    # Each removal takes O(C(Out(v), 4)) time.
    # If we have ~60 vertices, and Out(v) is ~30, C(30,4) ~ 27k.
    # 60 vertices * 27k = 1.6M checks. Very fast.
    
    while time.time() < deadline - 10:
        prev_n = n
        n, in_bits = _try_remove_vertex(n, in_bits)
        if n == prev_n:
            break
            
    # 4. Final check and return
    if n < 5:
        # Should not happen
        return {'n': 5, 'orientation': [0]*10}
        
    final_orientation = _in_bits_to_orientation(n, in_bits)
    
    return {'n': n, 'orientation': final_orientation}
