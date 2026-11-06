"""Find 4-colorings of K_n with no C6 in any color class."""
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # The known optimal coloring for n=17 uses a distance-based scheme
    # Let's try multiple known constructions and verify them quickly
    
    # Construction 1: Based on Z_17 (cyclic)
    # For K_17, color edge (i,j) by (j-i) mod 17 mapped to 4 colors
    # The known R_4(C6) > 17 coloring uses specific distance classes
    
    # Let's use the known optimal coloring structure.
    # The 4-coloring of K_17 without C6 can be derived from the Paley graph structure
    # or from specific distance partitions of Z_17.
    
    # Distance-based coloring of Z_17:
    # distances 1..8 (since 17 is odd, max distance is 8)
    # We need to partition {1,2,3,4,5,6,7,8} into 4 sets, each not containing a C6
    
    # A known valid partition for n=17:
    # Color 0: distances {1, 4, 10, 14} -> but we use min(d, 17-d) so {1,4,6,7}
    # Actually let me think more carefully.
    
    # In Z_17, edge (i,j) has distance d = (j-i) mod 17, we can take d in {1,...,8}
    # by taking min(d, 17-d).
    
    # The standard optimal 4-coloring of K_17 uses:
    # Color 0: {1, 2, 4, 8}  (differences)
    # Color 1: {3, 5, 7, 11} -> min: {3,5,7,6}
    # Color 2: {6, 9, 12, 15} -> min: {6,8,5,2}
    # Hmm, let me just try all reasonable partitions.
    
    # Actually, the known result uses the following:
    # The 4-coloring comes from a (17, 4) difference family.
    # Let's try the construction where we assign colors based on quadratic residues.
    
    # In Z_17, quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # Non-residues: {3, 5, 6, 7, 10, 11, 12, 14}
    
    # One approach: use 4 subsets of Z_17 \ {0}
    
    # Let me just try the distance-based approach with different partitions
    # and verify quickly.
    
    # For n=17, the edges can be labeled by distance d in {1,...,8}
    # (where distance between i and j is min(|i-j|, 17-|i-j|))
    
    # We need to assign each distance to one of 4 colors such that no color
    # class contains a C6.
    
    # A C6 in a color class means there exist 6 vertices v0..v5 such that
    # consecutive pairs and (v5,v0) are all the same color.
    
    # For the cyclic group Z_17, a C6 in one color class would correspond to
    # 6 steps that sum to 0 mod 17, where each step is in the distance set for that color.
    
    # Let's try specific known valid colorings.
    
    # Known: The optimal 4-coloring of K_17 can be constructed as follows:
    # Assign to each unordered pair {i,j} the color based on (j-i) mod 17.
    
    # One valid partition of {1,...,16} into 4 sets (using representatives 1..8):
    # Let's try: 
    #   C0 = {1, 4, 10, 14}  -> as distances: {1, 4, 7, 3}
    #   C1 = {2, 5, 11, 15}  -> as distances: {2, 5, 6, 2} -> {2, 5, 6}
    #   C2 = {3, 6, 12, 16}  -> as distances: {3, 6, 5, 1}
    #   C3 = {7, 8, 9, 13}   -> as distances: {7, 8, 8, 4} -> {7, 8, 4}
    
    # This is getting complicated. Let me just use a verified coloring.
    
    # The simplest approach: use the known fact that n=17 is achievable.
    # I'll construct it using a specific scheme and verify.
    
    # Scheme: color(i,j) = f((j-i) mod 17) where f maps to {0,1,2,3}
    
    # Let me try: f(d) for d in 1..16
    # Using the partition:
    #   color 0: d in {1, 4, 10, 14}
    #   color 1: d in {2, 5, 11, 15}  
    #   color 2: d in {3, 6, 12, 16}
    #   color 3: d in {7, 8, 9, 13}
    
    n = 17
    color_map = {}
    for d in range(1, 17):
        if d in (1, 4, 10, 14):
            color_map[d] = 0
        elif d in (2, 5, 11, 15):
            color_map[d] = 1
        elif d in (3, 6, 12, 16):
            color_map[d] = 2
        else:  # d in (7, 8, 9, 13)
            color_map[d] = 3
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % 17
            colors.append(color_map[d])
    
    # Quick verification using a faster C6 check
    if time.time() < deadline - 5:
        if _verify_no_c6_fast(colors, n):
            return {'n': n, 'colors': colors}
    
    # If that didn't work, try another known construction
    # Alternative: use min-distance coloring
    colors2 = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(j - i)
            d = min(d, n - d)
            # Map distances 1..8 to 4 colors
            # Try: {1,5}->0, {2,6}->1, {3,7}->2, {4,8}->3
            if d in (1, 5):
                colors2.append(0)
            elif d in (2, 6):
                colors2.append(1)
            elif d in (3, 7):
                colors2.append(2)
            else:  # d in (4, 8)
                colors2.append(3)
    
    if time.time() < deadline - 5:
        if _verify_no_c6_fast(colors2, n):
            return {'n': n, 'colors': colors2}
    
    # Try yet another partition
    colors3 = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(j - i)
            d = min(d, n - d)
            if d in (1, 4):
                colors3.append(0)
            elif d in (2, 5):
                colors3.append(1)
            elif d in (3, 6):
                colors3.append(2)
            else:  # d in (7, 8)
                colors3.append(3)
    
    if time.time() < deadline - 5:
        if _verify_no_c6_fast(colors3, n):
            return {'n': n, 'colors': colors3}
    
    # Fallback: try smaller n with verified colorings
    for n_try in [16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline - 10:
            break
        # Use simple distance-based coloring
        colors_t = []
        for i in range(n_try):
            for j in range(i + 1, n_try):
                d = abs(j - i)
                d = min(d, n_try - d)
                colors_t.append((d - 1) % 4)
        
        if _verify_no_c6_fast(colors_t, n_try):
            return {'n': n_try, 'colors': colors_t}
    
    # Last resort: n=9
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(j - i)
            d = min(d, n - d)
            colors.append((d - 1) % 4)
    return {'n': n, 'colors': colors}


def _verify_no_c6_fast(colors, n):
    """Verify no color class contains a C6 using adjacency bitmasks."""
    # Build adjacency lists for each color
    adj = [0] * (4 * n)  # adj[c*n + v] is bitmask of neighbors of v in color c
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
    
    # For each color, check for C6
    for c in range(4):
        base = c * n
        # Check all 6-tuples (v0, v1, ..., v5) with v0 < v1 < ... < v5
        # and check if they form a cycle
        # This is still O(n^6) but with bitmask operations it's fast
        
        # More efficient: for each pair (v0, v1) adjacent in color c,
        # check if there's a path v1->v2->v3->v4->v5->v0 of length 4
        # where all edges are in color c.
        
        # Even faster: use the fact that we just need to detect ANY C6
        # For each vertex v, look at its neighbors in color c.
        # A C6 through v means there exist 5 other vertices forming a path of length 4
        # from some neighbor back to v.
        
        # Let's do: for each ordered pair (a, b) with a adjacent to b in color c,
        # check if there's a path of length 4 from b to a in color c
        # (not using edge (a,b) directly, but that's handled by the path structure)
        
        # Actually, let's just do the direct check with bitmasks for efficiency
        for v0 in range(n):
            mask0 = adj[base + v0]
            # For each neighbor v1 of v0
            v1 = 0
            m1 = mask0
            while m1:
                if m1 & 1:
                    # v1 = 0, check if there's a path 0->v2->v3->v4->v5->0
                    # i.e., a path of length 4 from v1 back to v0
                    pass
                m1 >>= 1
                v1 += 1
            
            # Simpler approach: enumerate all 6-combinations and check
            # But that's C(17,6) = 12376 per color, times 4 = ~50k checks
            # Each check is 6 adjacency lookups. Should be fast enough.
        
        # Let's just do the combination check - it's only 12376 * 4 = ~50k iterations
        from itertools import combinations
        for combo in combinations(range(n), 6):
            a, b, c2, d, e, f = combo
            # Check all 6 cyclic orderings (up to rotation and reversal, so 3 distinct)
            # Actually for a cycle a-b-c-d-e-f-a, the distinct cycles through these 6
            # vertices (considering the fixed set) are the 5! / 2 / 6 * ... 
            # Let's just check: a-b-c-d-e-f-a and a-f-e-d-c-b-a (same) 
            # and a-b-d-c-e-f-a etc.
            # There are (6-1)!/2 = 60 distinct cycles through 6 labeled vertices
            # But we can be smarter.
            
            # Check if a,b,c,d,e,f form a cycle in some order
            # The cycle is: v0-v1-v2-v3-v4-v5-v0
            # We need to check if there exists a permutation such that consecutive
            # pairs are adjacent in color c.
            
            # Quick necessary condition: each vertex must have degree >= 2 in the
            # induced subgraph on these 6 vertices (in color c)
            # But let's just do the direct check for the few likely patterns.
            
            # For speed, check the natural orderings:
            # (a,b,c,d,e,f), (a,b,c,d,f,e), (a,b,c,e,d,f), etc.
            # This is 60 permutations which is fine.
            
            # Actually, let me just check all 60 permutations of (b,c,d,e,f) with a fixed
            # and see if a-v1-v2-v3-v4-v5-a works.
            # But 60 * 12376 * 4 = ~3M checks, each with 6 adjacency tests.
            # That might be too slow in Python.
            
            # Better: precompute for each color, for each pair, whether they're adjacent.
            # Then for the 6-tuple, check the 60 cycle patterns.
            pass
        
        # Let me use a different approach: BFS/DFS to find cycles of length 6
        # For each color, build the graph and look for C6.
        # For n=17, this should be fast enough with bitmasks.
        
        # Use the following: for each pair (u,v) adjacent in color c,
        # find common neighbors of u and v (excluding each other), then
        # for each such w, find common neighbors of v and w (excluding u),
        # etc. This is getting complex.
        
        # Let me just do the brute force with early termination.
        # For n=17, C(17,6)=12376, and for each we check up to 60 permutations.
        # Total: ~740k checks per color, ~3M total. Each check is 6 bit operations.
        # In Python this might take a few seconds, which is acceptable.
        
        for combo in combinations(range(n), 6):
            # Check if these 6 vertices contain a C6 in color c
            # Try all cyclic permutations
            verts = list(combo)
            found = False
            # Fix v0 = verts[0], permute the rest
            from itertools import permutations
            rest = verts[1:]
            for perm in permutations(rest):
                # To avoid duplicates, require perm[0] < perm[4]
                if perm[0] > perm[4]:
                    continue
                v1, v2, v3, v4, v5 = perm
                # Check cycle: v0-v1-v2-v3-v4-v5-v0
                if (_adj(adj, base, verts[0], v1) and
                    _adj(adj, base, v1, v2) and
                    _adj(adj, base, v2, v3) and
                    _adj(adj, base, v3, v4) and
                    _adj(adj, base, v4, v5) and
                    _adj(adj, base, v5, verts[0])):
                    found = True
                    break
            if found:
                return False
    
    return True


def _adj(adj, base, u, v):
    """Check if u and v are adjacent in the given color class."""
    return (adj[base + u] >> v) & 1


if __name__ == "__main__":
    result = construct()
    print(f"Found coloring with n={result['n']}")
    print(f"Number of colors: {len(result['colors'])}")
