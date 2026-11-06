"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    # Try to construct a 30-vertex coloring by extending the Paley graph of order 29
    n_base = 29
    # Paley graph of order 29
    squares = set()
    for x in range(1, n_base):
        squares.add((x * x) % n_base)
    
    # adj[i] is a bitmask of neighbors in color 0 for vertex i in 0..28
    # We only need adjacency within 0..28
    adj0 = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % n_base
            if diff in squares:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            # else color 1, no bit set
    
    # Precompute has_k22_0 and has_k22_1 for all 4-subsets
    # has_k22_0[combo] is True if the 4 vertices contain a K2,2 in color 0
    # has_k22_1[combo] is True if the 4 vertices contain a K2,2 in color 1
    # We use a dictionary keyed by frozenset or tuple of sorted vertices
    has_k22_0 = {}
    has_k22_1 = {}
    
    all_4_subsets = list(itertools.combinations(range(n_base), 4))
    for combo in all_4_subsets:
        a, b, c, d = combo
        # Check 3 partitions
        # Partition 1: (a,b) vs (c,d)
        # Partition 2: (a,c) vs (b,d)
        # Partition 3: (a,d) vs (b,c)
        
        def is_k22(p1, p2, color):
            # p1, p2 are tuples of 2 vertices
            # Check all 4 cross edges
            u1, v1 = p1
            u2, v2 = p2
            if color == 0:
                return ((adj0[u1] >> u2) & 1) and ((adj0[u1] >> v2) & 1) and \
                       ((adj0[v1] >> u2) & 1) and ((adj0[v1] >> v2) & 1)
            else:
                # Color 1 means NOT in adj0
                return not ((adj0[u1] >> u2) & 1) and not ((adj0[u1] >> v2) & 1) and \
                       not ((adj0[v1] >> u2) & 1) and not ((adj0[v1] >> v2) & 1)
        
        k22_0 = False
        k22_1 = False
        if is_k22((a,b), (c,d), 0) or is_k22((a,c), (b,d), 0) or is_k22((a,d), (b,c), 0):
            k22_0 = True
        if is_k22((a,b), (c,d), 1) or is_k22((a,c), (b,d), 1) or is_k22((a,d), (b,c), 1):
            k22_1 = True
            
        has_k22_0[combo] = k22_0
        has_k22_1[combo] = k22_1
        
    # Precompute edges_from_ik_to_rest for 5-subsets
    # For a 5-subset Y and a vertex yk in Y, 
    # edges_from_ik_to_rest_0[Y, yk] is True if all edges from yk to Y\{yk} are color 0
    # We store this in a dict keyed by (frozenset(Y), yk)
    # Or better, for each 5-subset, store a bitmask of which yk satisfy the condition for color 0 and 1
    # mask_0[Y] has bit yk set if edges from yk to rest are color 0
    # mask_1[Y] has bit yk set if edges from yk to rest are color 1
    
    all_5_subsets = list(itertools.combinations(range(n_base), 5))
    mask_0 = {}
    mask_1 = {}
    for Y in all_5_subsets:
        m0 = 0
        m1 = 0
        for yk in Y:
            rest = [v for v in Y if v != yk]
            # Check color 0
            ok0 = True
            for v in rest:
                if not ((adj0[yk] >> v) & 1):
                    ok0 = False
                    break
            if ok0:
                m0 |= (1 << yk)
                
            # Check color 1
            ok1 = True
            for v in rest:
                if (adj0[yk] >> v) & 1:
                    ok1 = False
                    break
            if ok1:
                m1 |= (1 << yk)
                
        mask_0[Y] = m0
        mask_1[Y] = m1

    # Local search on S0 (bitmask of 29 bits, bit i set if edge (29, i) is color 0)
    # S0 determines colors of edges (29, i) for i in 0..28
    # If bit i is 0, edge (29, i) is color 1.
    
    def is_valid(S0):
        # Check all 5-subsets Y of 0..28
        for Y in all_5_subsets:
            # Check Color 0 violation
            # Condition: There exists yk in Y such that:
            # 1. Y is subset of S0 (all edges 29-v are color 0)
            #    This means (S0 & mask_Y) == mask_Y, where mask_Y is bitmask of Y
            #    Wait, S0 is global. Y is a specific subset.
            #    We need Y \subset S0.
            #    Let mask_Y be the bitmask of vertices in Y.
            #    If (S0 & mask_Y) == mask_Y:
            #       Then all edges from 29 to Y are color 0.
            #       Now we need to check if there is a yk in Y such that:
            #       - yk is in S0 (implied by Y subset S0)
            #       - edges from yk to Y\{yk} are color 0 (mask_0[Y] has bit yk)
            #       - Y\{yk} contains a K2,2 in color 0 (has_k22_0[Y\{yk}] is True)
            
            mask_Y = 0
            for v in Y:
                mask_Y |= (1 << v)
                
            if (S0 & mask_Y) == mask_Y:
                # All edges from 29 to Y are color 0
                # Check if there is a "center" yk in Y
                # yk must be in mask_0[Y]
                # And Y\{yk} must be in has_k22_0
                m0 = mask_0[Y]
                # Iterate through bits set in m0
                temp_m = m0
                while temp_m:
                    lsb = temp_m & -temp_m
                    yk = lsb.bit_length() - 1
                    # Check Y\{yk}
                    Y_rest = tuple(v for v in Y if v != yk)
                    if has_k22_0[Y_rest]:
                        return False
                    temp_m ^= lsb
                    
            # Check Color 1 violation
            # S1 is complement of S0 in 0..28
            # mask_S1 = ((1 << 29) - 1) ^ S0
            mask_S1 = ((1 << n_base) - 1) ^ S0
            if (mask_S1 & mask_Y) == mask_Y:
                # All edges from 29 to Y are color 1
                m1 = mask_1[Y]
                temp_m = m1
                while temp_m:
                    lsb = temp_m & -temp_m
                    yk = lsb.bit_length() - 1
                    Y_rest = tuple(v for v in Y if v != yk)
                    if has_k22_1[Y_rest]:
                        return False
                    temp_m ^= lsb
                    
        return True

    # Initialize S0 randomly
    S0 = random.getrandbits(n_base)
    
    # Local search
    max_iters = 20000
    for _ in range(max_iters):
        if is_valid(S0):
            # Construct full coloring for n=30
            n = 30
            colors = []
            # Edges within 0..28
            for i in range(n_base):
                for j in range(i + 1, n_base):
                    if (adj0[i] >> j) & 1:
                        colors.append(0)
                    else:
                        colors.append(1)
            # Edges from 29 to 0..28
            for i in range(n_base):
                if (S0 >> i) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
            return {'n': n, 'colors': colors}
        
        # Flip a random bit in S0
        bit = random.randint(0, n_base - 1)
        S0 ^= (1 << bit)

    # Fallback to n=29
    n = 29
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj0[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
