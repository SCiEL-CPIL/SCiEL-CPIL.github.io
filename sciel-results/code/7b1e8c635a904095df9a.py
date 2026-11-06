"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    mask = (1 << 201) - 1  # Bits 0 to 200
    
    # Greedy descending with bitset
    selected = []
    reachable = 1  # Bit 0 set
    
    for x in range(100, 0, -1):
        if (reachable >> (200 - x)) & 1:
            pass
        else:
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= mask
            
    S = set(selected)
    U = set(range(1, 101)) - S
    
    def compute_reachable(elements):
        """Compute reachable sums for a set of elements using bitset."""
        R = 1
        for y in elements:
            R |= (R << y)
            R &= mask
        return R
    
    def can_add_pair(R_base, u1, u2):
        """Check if adding u1 and u2 to R_base keeps 200 unreachable."""
        R1 = R_base | (R_base << u1)
        R1 &= mask
        if (R1 >> 200) & 1:
            return False
        R2 = R1 | (R1 << u2)
        R2 &= mask
        return not ((R2 >> 200) & 1)
    
    def can_add_triple(R_base, u1, u2, u3):
        """Check if adding u1, u2, u3 to R_base keeps 200 unreachable."""
        R1 = R_base | (R_base << u1)
        R1 &= mask
        if (R1 >> 200) & 1:
            return False
        R2 = R1 | (R1 << u2)
        R2 &= mask
        if (R2 >> 200) & 1:
            return False
        R3 = R2 | (R2 << u3)
        R3 &= mask
        return not ((R3 >> 200) & 1)
    
    improved = True
    while improved:
        improved = False
        
        # Try 1-for-2 swaps: remove 1 from S, add 2 from U
        for s in list(S):
            R_temp = compute_reachable(S - {s})
            
            for u1 in sorted(U, reverse=True):
                R1 = R_temp | (R_temp << u1)
                R1 &= mask
                if (R1 >> 200) & 1:
                    continue
                
                for u2 in sorted(U, reverse=True):
                    if u1 == u2:
                        continue
                    R2 = R1 | (R1 << u2)
                    R2 &= mask
                    if not ((R2 >> 200) & 1):
                        S.remove(s)
                        S.add(u1)
                        S.add(u2)
                        U.remove(u1)
                        U.remove(u2)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-for-3 swaps: remove 2 from S, add 3 from U
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                s1, s2 = S_list[i], S_list[j]
                R_temp = compute_reachable(S - {s1, s2})
                
                U_list = sorted(U, reverse=True)
                for u1 in U_list:
                    R1 = R_temp | (R_temp << u1)
                    R1 &= mask
                    if (R1 >> 200) & 1:
                        continue
                    
                    for u2 in U_list:
                        if u1 == u2:
                            continue
                        R2 = R1 | (R1 << u2)
                        R2 &= mask
                        if (R2 >> 200) & 1:
                            continue
                        
                        for u3 in U_list:
                            if u3 == u1 or u3 == u2:
                                continue
                            R3 = R2 | (R2 << u3)
                            R3 &= mask
                            if not ((R3 >> 200) & 1):
                                S.remove(s1)
                                S.remove(s2)
                                S.add(u1)
                                S.add(u2)
                                S.add(u3)
                                U.remove(u1)
                                U.remove(u2)
                                U.remove(u3)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return list(S)
# EVOLVE-BLOCK-END
