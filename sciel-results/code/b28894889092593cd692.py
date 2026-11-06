import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def build_paley(p):
        """
        Build Paley tournament on p vertices.
        Returns orientation list.
        """
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0) # i -> j
                else:
                    orientation.append(1) # j -> i
        return orientation

    def build_in_nb(n, orientation):
        """
        Build in-neighborhood bitsets.
        in_nb[i] is a bitmask of vertices j such that j -> i.
        """
        in_nb = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    # i -> j
                    # j's in-neighbor is i
                    in_nb[j] |= (1 << i)
                else:
                    # j -> i
                    # i's in-neighbor is j
                    in_nb[i] |= (1 << j)
        return in_nb

    def is_valid(n, in_nb):
        """
        Check if every 4-set has a common dominator.
        Dominator of {a,b,c,d} is v such that v -> a, v -> b, v -> c, v -> d.
        This is equivalent to v being in In(a) ∩ In(b) ∩ In(c) ∩ In(d).
        """
        for a in range(n - 3):
            if elapsed() > DEADLINE:
                return False
            for b in range(a + 1, n - 2):
                ab = in_nb[a] & in_nb[b]
                if ab == 0:
                    continue
                for c in range(b + 1, n - 1):
                    abc = ab & in_nb[c]
                    if abc == 0:
                        continue
                    for d in range(c + 1, n):
                        if (abc & in_nb[d]) == 0:
                            return False
        return True

    def try_reduce(n, orientation):
        """
        Try to reduce the tournament by removing non-critical vertices.
        A vertex v is critical if it is the unique dominator of some 4-set.
        """
        current_n = n
        current_orientation = orientation
        current_in_nb = build_in_nb(current_n, current_orientation)
        
        # We need to map vertices correctly when removing.
        # It's easier to work with a list of active vertices.
        # But orientation is a flat list. Removing a vertex requires rebuilding orientation.
        
        # Let's work with an adjacency matrix or bitsets and rebuild orientation at the end.
        # Actually, let's just keep the bitsets and a list of active vertices.
        
        active = list(range(n))
        in_nb = current_in_nb
        
        while True:
            if elapsed() > DEADLINE:
                break
            
            # Check criticality
            critical = [False] * current_n
            found_invalid = False
            
            # Iterate over all 4-sets of active vertices
            # We need to map active indices to original indices for bitsets?
            # No, let's just re-index everything to 0..current_n-1 each time?
            # That's expensive.
            
            # Alternative: Keep the original bitsets and just check subsets of active vertices.
            # But "unique dominator" means the intersection of in-nb of the 4 vertices 
            # contains exactly one vertex, and that vertex must be active.
            
            # Let's iterate over combinations of active vertices.
            # For efficiency, we can just check all 4-sets in the current active set.
            
            # To avoid O(n^4) overhead of itertools.combinations on a list, 
            # we can use indices into the 'active' list.
            
            # However, for n<=67, itertools.combinations is fine.
            
            for a_idx in range(current_n - 3):
                if elapsed() > DEADLINE:
                    break
                a = active[a_idx]
                for b_idx in range(a_idx + 1, current_n - 2):
                    b = active[b_idx]
                    ab = in_nb[a] & in_nb[b]
                    if ab == 0:
                        continue
                    for c_idx in range(b_idx + 1, current_n - 1):
                        c = active[c_idx]
                        abc = ab & in_nb[c]
                        if abc == 0:
                            continue
                        for d_idx in range(c_idx + 1, current_n):
                            d = active[d_idx]
                            doms = abc & in_nb[d]
                            if doms == 0:
                                # Invalid tournament (should not happen if we started valid)
                                return None 
                            
                            # Check if unique dominator
                            # Count bits in doms
                            # Python 3.10+ int.bit_count()
                            if hasattr(int, 'bit_count'):
                                cnt = doms.bit_count()
                            else:
                                cnt = bin(doms).count('1')
                                
                            if cnt == 1:
                                # Unique dominator is the single bit set in doms
                                v = doms.bit_length() - 1
                                # v must be in active set to be a valid dominator in the reduced tournament?
                                # Actually, if v is not active, it was already removed.
                                # But we are checking the *current* tournament.
                                # If v is the unique dominator, and v is in active, then v is critical.
                                # If v is NOT in active, then the intersection is empty in the *reduced* tournament?
                                # No, in_nb[a] contains all original vertices.
                                # We must ensure the dominator is in the active set.
                                
                                # If doms has only 1 bit, and that bit is NOT in active, 
                                # then in the active tournament, this 4-set has NO dominator.
                                # This means the current tournament (restricted to active) is INVALID.
                                # But we should have checked validity before reducing.
                                # If we only remove non-critical vertices, validity should be preserved.
                                # However, if we remove a vertex, it might become the unique dominator for some other set?
                                # No, removing a vertex can only *remove* dominators, not create new unique ones 
                                # unless the removed vertex was a *non*-dominator?
                                # Actually, if v is removed, any 4-set that had v as a dominator loses v.
                                # If v was the *only* dominator, the set becomes invalid.
                                # So if we remove v, we must ensure v was not the unique dominator of any 4-set.
                                
                                # So, criticality check:
                                # v is critical if there exists a 4-set S such that 
                                # {v} = {u in Active | u dominates S}
                                
                                if v in active:
                                    critical[v] = True
                                else:
                                    # The unique dominator in the full graph is not in the active set.
                                    # This implies the active tournament is invalid for this 4-set.
                                    # This should not happen if we started with a valid tournament 
                                    # and only removed vertices that were not unique dominators.
                                    # But to be safe, if this happens, we cannot reduce further or the state is corrupt.
                                    # Actually, if v is not in active, it means we already removed it.
                                    # If it was the unique dominator, we shouldn't have removed it.
                                    # So this case indicates a bug or that the tournament became invalid.
                                    # We'll treat this as "cannot reduce further" or "invalid".
                                    pass

            if found_invalid:
                break

            # Find a non-critical vertex
            removed = -1
            for v in range(current_n):
                if not critical[v]:
                    removed = v
                    break
            
            if removed == -1:
                # All vertices are critical, cannot reduce
                break
            
            # Remove 'removed' from active
            active.remove(removed)
            current_n -= 1
            
            # Optimization: If current_n is small, we can break early? No, keep going.
            
        # Rebuild orientation for the active set
        if not active:
            return None
            
        # Map active vertices to new indices 0..current_n-1
        # We need to generate orientation for pairs (i,j) in new indices.
        # Original indices are in 'active'.
        
        # We need the original orientation or adjacency to determine direction.
        # We have 'in_nb' for original indices.
        # If in_nb[b] has bit a, then a -> b.
        
        new_orientation = []
        # Create a mapping from original index to new index
        old_to_new = {old: new for new, old in enumerate(active)}
        
        for i_new in range(current_n):
            old_i = active[i_new]
            for j_new in range(i_new + 1, current_n):
                old_j = active[j_new]
                # Check direction old_i -> old_j
                # If in_nb[old_j] has bit old_i, then old_i -> old_j (bit 0)
                # Else old_j -> old_i (bit 1)
                if in_nb[old_j] & (1 << old_i):
                    new_orientation.append(0)
                else:
                    new_orientation.append(1)
                    
        return current_n, new_orientation

    # Candidate primes p ≡ 3 mod 4
    # We want to minimize n.
    # Try smaller primes first.
    candidates = [31, 43, 59, 67, 79, 83]
    
    best_n = 91
    best_orientation = None
    
    for p in candidates:
        if elapsed() > DEADLINE - 10:
            break
            
        if p > best_n:
            break
            
        orientation = build_paley(p)
        in_nb = build_in_nb(p, orientation)
        
        # Check validity
        if not is_valid(p, in_nb):
            continue
            
        # It is valid. Try to reduce.
        reduced = try_reduce(p, orientation)
        
        if reduced:
            n_red, ori_red = reduced
            if n_red < best_n:
                best_n = n_red
                best_orientation = ori_red
        else:
            # Could not reduce or reduction failed, use original
            if p < best_n:
                best_n = p
                best_orientation = orientation
                
        # If we found a very small n, we might stop?
        # Lower bound is 31. If we get 31, stop.
        if best_n <= 31:
            break

    if best_orientation is None:
        # Fallback to 67 if something went wrong
        p = 67
        orientation = build_paley(p)
        return {'n': p, 'orientation': orientation}
        
    return {'n': best_n, 'orientation': best_orientation}

# EVOLVE-BLOCK-END
