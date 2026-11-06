"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Greedy descending with bitset
    selected = []
    reachable = 1  # Bit 0 set
    mask = (1 << 201) - 1  # Bits 0 to 200
    
    for x in range(100, 0, -1):
        # Check if adding x makes 200 reachable
        # new_reachable = reachable | (reachable << x)
        # We only care if bit 200 is set in new_reachable
        # Bit 200 in new_reachable is set if:
        # 1. Bit 200 was already in reachable (should not happen if we maintain invariant)
        # 2. Bit (200-x) is in reachable
        if (reachable >> (200 - x)) & 1:
            # 200-x is reachable, so adding x makes 200 reachable
            pass
        else:
            # Safe to add
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= mask
            
    # Local Search to improve cardinality
    # S = selected, U = {1..100} \ S
    S = set(selected)
    U = set(range(1, 101)) - S
    
    # Helper to check if a set of additions is valid given a base set
    # We will modify S in place if improvement found
    
    improved = True
    while improved:
        improved = False
        # Try to remove 1 element from S and add 2 from U
        for s in list(S):
            # Calculate reachable sums for S \ {s}
            # This is expensive if done from scratch. 
            # But N=100, so we can recompute or use a trick.
            # Recomputing reachable for S\{s} takes O(|S| * 1) bit ops.
            # |S| ~ 40. So 40 ops.
            # We do this for each s in S (40 times) -> 1600 ops.
            # Then for each pair in U (1800 pairs) -> check.
            
            # Efficiently compute reachable for S \ {s}
            # We can just recompute from scratch for simplicity, it's fast enough.
            R_temp = 1
            for y in S:
                if y == s: continue
                R_temp |= (R_temp << y)
                R_temp &= mask
            
            # Now try to add pairs from U
            # We need to check if u1, u2 can be added to R_temp
            # Order of addition doesn't matter for the final set of sums, 
            # but for the "can we add" check, we just need to see if 
            # 200 is reachable in (S\{s}) U {u1, u2}
            
            # Optimization: 
            # If we add u1, new R1 = R_temp | (R_temp << u1)
            # Then add u2, new R2 = R1 | (R1 << u2)
            # Check if bit 200 in R2.
            
            # To speed up, we can precompute something?
            # Or just iterate.
            
            for u1 in U:
                R1 = R_temp | (R_temp << u1)
                R1 &= mask
                if (R1 >> 200) & 1:
                    continue # u1 alone makes 200 reachable (with S\{s})
                
                for u2 in U:
                    if u1 == u2: continue
                    R2 = R1 | (R1 << u2)
                    R2 &= mask
                    if not ((R2 >> 200) & 1):
                        # Valid swap
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
                
        # Try to remove 2 from S and add 3 from U?
        # This is more complex. Let's stick to 1-for-2 first.
        
    return list(S)
# EVOLVE-BLOCK-END
