import time
import itertools

def _has_c6_in_adj(adj, n):
    """Check if graph with adjacency masks 'adj' contains a 6-cycle."""
    # Optimization: If a vertex has degree < 3, it can't be in a C6.
    # But C6 requires min degree 2.
    # We just run the search.
    for v0 in range(n):
        m0 = adj[v0]
        while m0:
            v1 = (m0 & -m0).bit_length() - 1
            m0 &= m0 - 1
            
            # Prune: v1 must have another neighbor besides v0
            if adj[v1] & ~(1 << v0) == 0:
                continue
                
            m1 = adj[v1] & ~(1 << v0)
            while m1:
                v2 = (m1 & -m1).bit_length() - 1
                m1 &= m1 - 1
                
                if adj[v2] & ~(1 << v0) & ~(1 << v1) == 0:
                    continue

                m2 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while m2:
                    v3 = (m2 & -m2).bit_length() - 1
                    m2 &= m2 - 1
                    
                    if adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) == 0:
                        continue

                    m3 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while m3:
                        v4 = (m3 & -m3).bit_length() - 1
                        m3 &= m3 - 1
                        
                        # Check if v4 connects to any neighbor of v0, excluding v0..v4
                        candidates = adj[v4] & adj[v0]
                        used = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        if candidates & ~used:
                            return True
    return False

def _check_cyclic_coloring(n, dist_colors):
    """
    n: number of vertices
    dist_colors: list of colors for distances 1..n//2
    Returns True if valid (no mono C6)
    """
    # Build adjacency masks for 4 colors
    adj = [[0] * n for _ in range(4)]
    
    # Map distance to color
    # Distance d is between i and j if (j-i)%n == d or (i-j)%n == d
    # For i < j, d = j-i. If d > n/2, d = n-d.
    
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            if d > n // 2:
                d = n - d
            c = dist_colors[d - 1]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
    for c in range(4):
        if _has_c6_in_adj(adj[c], n):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # Try to find valid cyclic coloring for n from 20 down to 10
    # We prioritize larger n.
    # Known: n=17 works.
    
    for n in range(20, 9, -1):
        if time.time() > deadline:
            break
            
        max_dist = n // 2
        # Generate all combinations of colors for distances 1..max_dist
        # Using itertools.product
        
        # Optimization: We can break early if we find one.
        # To maximize n, we stop at the first valid n.
        
        # For n=20, 4^10 is large. Let's see if we can prune.
        # Symmetry: Colors are interchangeable? No, we are checking specific classes.
        # But we can fix color of distance 1 to 0 to break symmetry?
        # Not necessarily, because the structure matters.
        # However, for a quick search, full product is okay for n<=19.
        # For n=20, it might be slow.
        
        # Let's try n=17, 18, 19 first as they are more likely to exist or be faster to find?
        # No, we want max n.
        
        # If n=20 is too slow, we might timeout.
        # Let's add a time check inside the loop.
        
        found = False
        # Use product
        for combo in itertools.product(range(4), repeat=max_dist):
            if time.time() > deadline:
                return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
                
            if _check_cyclic_coloring(n, list(combo)):
                # Found valid coloring
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = j - i
                        if d > n // 2:
                            d = n - d
                        colors.append(combo[d - 1])
                return {'n': n, 'colors': colors}
        
        # If loop finishes without finding, try smaller n
        # But we should check time.
        
    # Fallback to known good n=9 or try to find n=17 specifically if loop failed due to time?
    # If we reached here, we didn't find one in time for n>=10.
    # Return the seed result.
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
