import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    def build_paley(p):
        """Build Paley tournament on p vertices (p must be prime ≡ 3 mod 4)."""
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # Build in-neighborhood bitsets directly
        in_nb = [0] * p
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)  # i -> j
                    in_nb[j] |= (1 << i)
                else:
                    orientation.append(1)  # j -> i
                    in_nb[i] |= (1 << j)
        
        return p, orientation, in_nb
    
    def is_valid_and_critical(p, in_nb):
        """Check validity and find critical vertices. Returns (valid, critical_list)."""
        critical = [False] * p
        valid = True
        
        for a in range(p - 3):
            if elapsed() > DEADLINE:
                return valid, critical
            na = in_nb[a]
            for b in range(a + 1, p - 2):
                ab = na & in_nb[b]
                if ab == 0:
                    continue
                for c in range(b + 1, p - 1):
                    abc = ab & in_nb[c]
                    if abc == 0:
                        continue
                    for d in range(c + 1, p):
                        D = abc & in_nb[d]
                        if D == 0:
                            valid = False
                            return valid, critical
                        # Check if D is a singleton
                        if D & (D - 1) == 0:
                            w = D.bit_length() - 1
                            critical[w] = True
        
        return valid, critical
    
    def remove_vertex(p, orientation, in_nb, w):
        """Remove vertex w and return new n, orientation, in_nb."""
        new_n = p - 1
        new_orientation = []
        new_in_nb = [0] * new_n
        
        # Mapping from old index to new index
        old_to_new = {}
        new_idx = 0
        for i in range(p):
            if i != w:
                old_to_new[i] = new_idx
                new_idx += 1
        
        # Build new in_nb and orientation
        for i_old in range(p):
            if i_old == w:
                continue
            i_new = old_to_new[i_old]
            for j_old in range(i_old + 1, p):
                if j_old == w:
                    continue
                j_new = old_to_new[j_old]
                # Determine direction: check in_nb
                # If i_old is in in_nb[j_old], then i_old -> j_old (bit i_old set in in_nb[j_old])
                if (in_nb[j_old] >> i_old) & 1:
                    # i_old -> j_old
                    new_orientation.append(0)
                    new_in_nb[j_new] |= (1 << i_new)
                else:
                    # j_old -> i_old
                    new_orientation.append(1)
                    new_in_nb[i_new] |= (1 << j_new)
        
        return new_n, new_orientation, new_in_nb
    
    # Try primes in increasing order
    candidate_primes = [31, 43, 59, 67, 79, 83]
    
    for p in candidate_primes:
        if elapsed() > DEADLINE - 30:
            break
        
        _, orientation, in_nb = build_paley(p)
        
        # Check validity and find critical vertices
        valid, critical = is_valid_and_critical(p, in_nb)
        
        if not valid:
            continue
        
        # Try greedy reduction
        current_n = p
        current_orientation = orientation
        current_in_nb = in_nb
        
        # Iteratively remove non-critical vertices
        while elapsed() < DEADLINE - 10:
            valid, critical = is_valid_and_critical(current_n, current_in_nb)
            if not valid:
                break
            
            # Find a non-critical vertex
            removable = -1
            for w in range(current_n):
                if not critical[w]:
                    removable = w
                    break
            
            if removable == -1:
                break
            
            current_n, current_orientation, current_in_nb = remove_vertex(
                current_n, current_orientation, current_in_nb, removable
            )
        
        return {'n': current_n, 'orientation': current_orientation}
    
    # Fallback
    p = 67
    _, orientation, _ = build_paley(p)
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
