import time
import numpy as np
import itertools

def _has_k222_fast(adj_bits, n):
    """Fast K2,2,2 check using bitmasks."""
    # For each triple of disjoint pairs, check all 12 cross-edges
    # Use the fact that for pairs (a,b), (c,d), (e,f):
    # Need: a~c, a~d, b~c, b~d, a~e, a~f, b~e, b~f, c~e, c~f, d~e, d~f
    
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            # Pair 1: (v0, v1)
            # Need v0~v1 to be irrelevant (they're in same pair)
            # For pair 2, pick v2, v3 from remaining
            for v2 in range(v0 + 1, n):
                if v2 == v1:
                    continue
                for v3 in range(v2 + 1, n):
                    if v3 == v1:
                        continue
                    # Pair 2: (v2, v3)
                    # Check edges between pair 1 and pair 2
                    if not ((adj_bits[v0] >> v2) & 1 and (adj_bits[v0] >> v3) & 1 and
                            (adj_bits[v1] >> v2) & 1 and (adj_bits[v1] >> v3) & 1):
                        continue
                    # For pair 3, pick v4, v5 from remaining
                    for v4 in range(v3 + 1, n):
                        if v4 == v1 or v4 == v2:
                            continue
                        for v5 in range(v4 + 1, n):
                            if v5 == v1 or v5 == v2 or v5 == v3:
                                continue
                            # Pair 3: (v4, v5)
                            # Check edges between pair 1 and pair 3
                            if not ((adj_bits[v0] >> v4) & 1 and (adj_bits[v0] >> v5) & 1 and
                                    (adj_bits[v1] >> v4) & 1 and (adj_bits[v1] >> v5) & 1):
                                continue
                            # Check edges between pair 2 and pair 3
                            if ((adj_bits[v2] >> v4) & 1 and (adj_bits[v2] >> v5) & 1 and
                                (adj_bits[v3] >> v4) & 1 and (adj_bits[v3] >> v5) & 1):
                                return True
    return False

def _check_validity_fast(colors, n):
    """Check validity using bitmask representation."""
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    return not _has_k222_fast(adj0, n) and not _has_k222_fast(adj1, n)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Paley graph on 29 vertices
    p = 29
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    base_colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            base_colors.append(0 if diff in squares else 1)
    
    # Verify base
    if not _check_validity_fast(base_colors, p):
        return {'n': p, 'colors': base_colors}
    
    # Try to extend to n=30
    # Use a smarter search: try all 2^29 is too many, but we can use
    # the constraint that the new vertex must not create K2,2,2 with any
    # existing structure.
    
    # Actually, let's try a different known construction.
    # Use the Paley graph on 31 vertices and remove one vertex.
    
    p31 = 31
    # For p=31, 31 mod 4 = 3, so -1 is not a QR.
    # Define S = {x^2 mod 31 : x = 1..15} (15 elements)
    S = set()
    for x in range(1, p31):
        S.add((x * x) % p31)
    # S has 15 elements (since 31 is prime, there are (31-1)/2 = 15 QRs)
    
    # Build coloring for K_31
    colors31 = []
    for i in range(p31):
        for j in range(i + 1, p31):
            diff = (j - i) % p31
            colors31.append(0 if diff in S else 1)
    
    # Check if K_31 coloring is valid (it shouldn't be, since R=31)
    # But let's check subsets of size 30
    
    best_n = p
    best_colors = base_colors[:]
    
    # Try removing each vertex from K_31
    for remove in range(p31):
        if time.time() > deadline:
            break
        
        # Build coloring for vertices 0..30 except 'remove'
        vertices = [v for v in range(p31) if v != remove]
        old_to_new = {old: new for new, old in enumerate(vertices)}
        
        colors30 = []
        for i_idx in range(len(vertices)):
            for j_idx in range(i_idx + 1, len(vertices)):
                old_i = vertices[i_idx]
                old_j = vertices[j_idx]
                # Find the color in colors31
                # Edge (old_i, old_j) in original
                if old_i < old_j:
                    pos = old_i * (p31 - 1) - old_i * (old_i + 1) // 2 + (old_j - old_i - 1)
                    # Actually, let's just recompute
                    diff = (old_j - old_i) % p31
                    c = 0 if diff in S else 1
                else:
                    diff = (old_i - old_j) % p31
                    c = 0 if diff in S else 1
                colors30.append(c)
        
        if _check_validity_fast(colors30, 30):
            return {'n': 30, 'colors': colors30}
    
    # If n=30 not found, return n=29
    return {'n': best_n, 'colors': best_colors}
