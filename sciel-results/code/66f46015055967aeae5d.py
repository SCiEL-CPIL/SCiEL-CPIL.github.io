import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # --- Helper: Build adjacency masks from colors ---
    def build_masks(colors, n, color_val):
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color_val:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        return adj

    # --- Helper: Check for K2,2,2 in a given color class using bitmasks ---
    def has_k222(adj, n):
        # We look for 3 disjoint pairs (u,v), (a,b), (c,d) such that:
        # 1. u-v, a-b, c-d are NON-edges (within parts)
        # 2. All 12 cross edges between the 3 parts ARE edges.
        # 
        # Optimization: Iterate over potential first pair (u,v) which must be a non-edge.
        # Let C be the set of common neighbors of u and v.
        # The other 4 vertices must be in C.
        # Among C, we need to find two disjoint non-edges (a,b) and (c,d)
        # such that a,b,c,d are all connected to each other (forming K4 between the two pairs? No, K2,2 between the pairs).
        # Actually, if (a,b) and (c,d) are in C, they are neighbors of u and v.
        # We need:
        # - a-b is non-edge
        # - c-d is non-edge
        # - a-c, a-d, b-c, b-d are edges.
        
        for u in range(n):
            for v in range(u + 1, n):
                # Check if u-v is a non-edge
                if adj[u] & (1 << v):
                    continue
                
                # Common neighbors of u and v
                C = adj[u] & adj[v]
                
                # We need to pick 4 vertices from C: a, b, c, d.
                # Partitioned into two pairs {a,b} and {c,d}.
                # Conditions:
                # 1. a-b is non-edge
                # 2. c-d is non-edge
                # 3. a~c, a~d, b~c, b~d are edges.
                
                # To speed this up, we can iterate over possible pairs (a,b) in C that are non-edges.
                # For each such pair, we check if there exists a pair (c,d) in C \ {a,b}
                # such that c-d is non-edge and c,d are connected to a,b.
                
                # Extract vertices in C
                verts_c = []
                mask = C
                while mask:
                    lsb = mask & -mask
                    v_idx = lsb.bit_length() - 1
                    verts_c.append(v_idx)
                    mask ^= lsb
                
                m = len(verts_c)
                if m < 4:
                    continue

                # Iterate over pairs (a,b) in verts_c
                for i in range(m):
                    a = verts_c[i]
                    for j in range(i + 1, m):
                        b = verts_c[j]
                        
                        # Check if a-b is non-edge
                        if adj[a] & (1 << b):
                            continue
                        
                        # Now we need to find (c,d) in verts_c \ {a,b}
                        # such that:
                        # 1. c-d is non-edge
                        # 2. c~a, c~b, d~a, d~b (c,d are neighbors of a and b)
                        
                        # Precompute mask of neighbors of a and b within C, excluding a and b
                        # Note: c and d must be in C, so they are already neighbors of u and v.
                        # We just need them to be neighbors of a and b.
                        
                        # Candidates for c,d must be in:
                        # neighbors_of_a_in_C = adj[a] & C
                        # neighbors_of_b_in_C = adj[b] & C
                        # common_ab_in_C = neighbors_of_a_in_C & neighbors_of_b_in_C
                        # Exclude a and b themselves (though they are non-adjacent, so they wouldn't be in each other's neighbor set anyway? 
                        # Wait, if a-b is non-edge, a is not in adj[b] and b is not in adj[a]. So a and b are naturally excluded from the intersection if we consider strict neighbors?
                        # adj[a] contains neighbors of a. Since a-b is non-edge, b is not in adj[a]. So b is not in adj[a] & C.
                        # Similarly a is not in adj[b] & C.
                        # So common_ab_in_C does not contain a or b.
                        
                        common_ab = (adj[a] & adj[b] & C)
                        
                        # Extract candidates
                        verts_ab = []
                        mask2 = common_ab
                        while mask2:
                            lsb = mask2 & -mask2
                            v_idx = lsb.bit_length() - 1
                            verts_ab.append(v_idx)
                            mask2 ^= lsb
                            
                        k = len(verts_ab)
                        if k < 2:
                            continue
                            
                        # Check pairs in verts_ab
                        for x in range(k):
                            c = verts_ab[x]
                            for y in range(x + 1, k):
                                d = verts_ab[y]
                                # Check if c-d is non-edge
                                if not (adj[c] & (1 << d)):
                                    return True
        return False

    def is_valid(colors, n):
        if len(colors) != n * (n - 1) // 2:
            return False
        adj0 = build_masks(colors, n, 0)
        if has_k222(adj0, n):
            return False
        adj1 = build_masks(colors, n, 1)
        if has_k222(adj1, n):
            return False
        return True

    # --- Paley Construction ---
    def paley_colors(n, p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                colors.append(0 if diff in squares else 1)
        return colors

    # --- Attempt n=29 (Known good) ---
    n_29 = 29
    p_29 = 29
    colors_29 = paley_colors(n_29, p_29)
    if is_valid(colors_29, n_29):
        return {'n': n_29, 'colors': colors_29}
    
    # Fallback: If Paley(29) somehow failed validation (shouldn't happen), try local search on it
    # But first, try to find n=30.

    # --- Attempt n=30 ---
    n_30 = 30
    # Strategy: Start with Paley(29) extended to 30.
    # Vertex 29 is new. We need to assign colors for (29, j) for j=0..28.
    # In Paley(29), vertices are 0..28.
    # If we use Paley(31) restricted to 30 vertices? No, 31 is prime.
    # Let's try Paley(31) restricted to first 30 vertices (0..29).
    p_31 = 31
    colors_31_restricted = paley_colors(30, p_31)
    
    if is_valid(colors_31_restricted, n_30):
        return {'n': n_30, 'colors': colors_31_restricted}

    # If Paley(31) restricted didn't work, try local search starting from it.
    # Also try random initialization for n=30.
    
    def local_search(colors, n, time_limit):
        start = time.time()
        if is_valid(colors, n):
            return True
        
        # Pre-compute some structures to speed up?
        # For simplicity, we do full checks.
        # To make it faster, we can try to flip edges that are in many violations.
        # But counting violations is as hard as checking validity.
        # So we just do random flips and check.
        
        # Optimization: Use a "soft" check or partial check?
        # Let's stick to full check but limit iterations.
        
        while time.time() - start < time_limit:
            # Pick a random edge to flip
            idx = random.randint(0, len(colors) - 1)
            colors[idx] = 1 - colors[idx]
            if is_valid(colors, n):
                return True
        return is_valid(colors, n)

    # Try local search on Paley(31) restricted
    if time.time() < deadline - 10:
        colors_try = colors_31_restricted[:]
        if local_search(colors_try, n_30, time_limit=min(10, deadline - time.time() - 5)):
            return {'n': n_30, 'colors': colors_try}

    # Try a few random restarts for n=30
    restarts = 5
    time_per_restart = (deadline - time.time() - 5) / restarts
    for r in range(restarts):
        if time.time() > deadline - 5:
            break
        colors_rand = [random.randint(0, 1) for _ in range(n_30 * (n_30 - 1) // 2)]
        # Quick validity check first
        if is_valid(colors_rand, n_30):
            return {'n': n_30, 'colors': colors_rand}
        
        # Local search
        if local_search(colors_rand, n_30, time_limit=max(1, time_per_restart)):
            return {'n': n_30, 'colors': colors_rand}

    # If n=30 failed, return n=29
    # Re-verify n=29 just in case
    if is_valid(colors_29, n_29):
        return {'n': n_29, 'colors': colors_29}
    
    # Fallback to smaller n if 29 fails (should not happen)
    for n in range(28, 10, -1):
        # Find next prime p >= n with p % 4 == 1
        p = n
        while True:
            is_prime = True
            for i in range(2, int(p**0.5)+1):
                if p % i == 0:
                    is_prime = False
                    break
            if is_prime and p % 4 == 1:
                break
            p += 1
        colors = paley_colors(n, p)
        if is_valid(colors, n):
            return {'n': n, 'colors': colors}
            
    # Last resort: n=1
    return {'n': 1, 'colors': []}

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}, len(colors)={len(res['colors'])}")
    # Validate
    n = res['n']
    colors = res['colors']
    valid = is_valid(colors, n)
    print(f"Valid: {valid}")
