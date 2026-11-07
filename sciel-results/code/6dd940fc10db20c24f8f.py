"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def color_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks
        # Palette is 0..4, so masks fit in 5 bits
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask
        
        # Check for empty domains initially
        for i in range(n):
            if domains[i] == 0:
                return None

        # AC-3 Arc Consistency
        # Queue of arcs (u, v) where u's domain might be reduced by v's domain
        # We store arcs as tuples (u, v)
        from collections import deque
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        def revise(u, v):
            # Check if domain of u is consistent with domain of v
            # For each color in u, is there a color in v that is different?
            # Since colors are distinct, if u has color c, v must have some color != c.
            # If v's domain is {c}, then u cannot have c.
            # Generally, u can keep c if domain_v has any bit set other than (1<<c).
            # domain_v_without_c = domain_v & ~(1 << c)
            # If domain_v_without_c == 0, then c must be removed from u.
            du = domains[u]
            dv = domains[v]
            
            # If dv is empty, we already handled it.
            # If dv has only one color c, and du has c, then c must be removed from du.
            # More generally:
            # For each bit c in du:
            #   if (dv & ~(1<<c)) == 0: remove c from du
            
            # Optimization: 
            # If dv is a singleton mask (1<<k), then du &= ~(1<<k)
            # If dv has >= 2 colors, then for any c in du, there is always a d in dv, d!=c?
            # Not necessarily. If dv = {0, 1} and du = {0, 1}. 
            # If u=0, v must be 1. OK.
            # If u=1, v must be 0. OK.
            # So if |dv| >= 2, does it always support any c in du?
            # Yes, because if c is in dv, there is another d in dv, d!=c. If c not in dv, any d in dv works.
            # So we only need to check if dv is a singleton.
            
            if dv == 0:
                return False # Should not happen if we check empty domains
            
            # Check if dv is a power of 2 (singleton)
            if dv & (dv - 1) == 0:
                # dv is singleton {k}
                # If du has k, remove k
                if du & dv:
                    domains[u] = du & ~dv
                    return domains[u] != du # True if changed
            else:
                # dv has >= 2 colors
                # Any color in du is supported by some color in dv?
                # Yes, as reasoned above.
                # Wait, what if du has a color c, and dv has {c, d}? 
                # If u takes c, v must take d. d is in dv. OK.
                # So no reduction needed.
                return False
            
            return domains[u] != du

        # Run AC-3
        while queue:
            u, v = queue.popleft()
            if revise(u, v):
                if domains[u] == 0:
                    return None
                for w in adj[u]:
                    if w != v:
                        queue.append((w, u))
        
        # Backtracking with MRV and Forward Checking
        # We use a static order initially? No, MRV is better.
        # To avoid deep recursion overhead and state copying, we'll use a stack or careful recursion.
        # Given n <= 408, recursion depth is manageable.
        
        assignment = [-1] * n
        # Copy domains for backtracking
        cur_domains = domains[:]
        
        # Helper to count bits
        def popcount(x):
            return bin(x).count("1")
            
        def solve():
            # Find unassigned vertex with MRV
            mrv_v = -1
            min_size = 10
            for v in range(n):
                if assignment[v] == -1:
                    size = popcount(cur_domains[v])
                    if size < min_size:
                        min_size = size
                        mrv_v = v
                        if size == 0:
                            return False
                        if size == 1:
                            # Can't get better than 1, but we might want to check if all are 1?
                            # Breaking early on 1 is fine, but we might miss a 0? 
                            # If size is 0, we returned. So min is 1.
                            break
            
            if mrv_v == -1:
                # All assigned
                return True
                
            if min_size == 0:
                return False
            
            # Save state of neighbors for restoration
            # We will modify cur_domains of neighbors
            v = mrv_v
            saved_neighbor_domains = []
            
            # Get colors in domain of v
            mask = cur_domains[v]
            colors = []
            for c in range(5):
                if mask & (1 << c):
                    colors.append(c)
                    
            # Try each color
            for c in colors:
                assignment[v] = c
                
                # Forward Checking: update neighbors
                conflict = False
                for u in adj[v]:
                    if assignment[u] == -1:
                        # Check if c is in cur_domains[u]
                        if cur_domains[u] & (1 << c):
                            saved_neighbor_domains.append((u, cur_domains[u]))
                            cur_domains[u] &= ~(1 << c)
                            if cur_domains[u] == 0:
                                conflict = True
                                break
                
                if not conflict:
                    if solve():
                        return True
                
                # Restore
                for u, old_dom in saved_neighbor_domains:
                    cur_domains[u] = old_dom
                assignment[v] = -1
                
            return False
            
        if solve():
            # Ensure all are assigned
            if -1 in assignment:
                return None
            return [int(c) for c in assignment]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = color_instance(n, edges, lists)
        if coloring is None:
            # Fallback: should not happen, but ensure valid structure
            coloring = [0] * n
        # Ensure types are Python int
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
