import random

def construct():
    d = 7
    n = 2 ** d
    # Initialize with the linear construction: a_ii=0, a_ij=1
    # L_i(u) = sum_{k != i} u_k
    # Select edge {u, u^e_i} if L_i(u) == 0
    
    edges = set()
    # We represent edges as (min, max)
    
    # Step 1: Add edges from the linear construction
    for i in range(d):
        for u in range(n):
            # Calculate L_i(u) = sum of bits of u except bit i
            # u ^ (1 << i) is the other end
            # L_i(u) is parity of u without bit i
            # This is equivalent to (u ^ (1 << i)) % 2? No.
            # Parity of u is p. Bit i is b.
            # Parity of u without i is p ^ b.
            # We want p ^ b == 0 => p == b.
            # So select if parity(u) == bit_i(u).
            
            bit_i = (u >> i) & 1
            parity = bin(u).count('1') % 2
            
            if parity == bit_i:
                v = u ^ (1 << i)
                if u < v:
                    edges.add((u, v))
                else:
                    edges.add((v, u))
                    
    # Step 2: Greedy add remaining edges if they don't form a C4
    # To speed up, we can just return the 224 edges if time is tight,
    # but let's try to add more.
    
    # Generate all edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))
                
    # Filter out already selected
    candidates = [e for e in all_edges if e not in edges]
    
    # Randomize order to avoid bias
    random.shuffle(candidates)
    
    for u, v in candidates:
        # Check if adding (u, v) creates a C4
        # A C4 is formed if there is a w such that (u, w) and (v, w) are in edges
        # and w is distance 2 from u? No.
        # In hypercube, C4 is u, v, w, z.
        # If we add (u, v), we check all common neighbors?
        # u and v differ in 1 bit.
        # Common neighbors of u and v? None.
        # A C4 containing (u, v) must be u-v-w-z-u.
        # v and z differ in 1 bit. w and z differ in 1 bit.
        # u and z differ in 2 bits.
        # So for each z such that dist(u, z) == 2 and dist(v, z) == 1?
        # No.
        # If (u, v) is added, it completes a C4 if there exists a path u-w-v of length 2 in the existing graph?
        # No, u and v are adjacent.
        # A C4 is a cycle of length 4.
        # Adding (u, v) creates a C4 if there is a path of length 3 from u to v in the existing graph.
        # Or simply, if there is a w such that (u, w) in E and (w, v) in E? No, that's a triangle.
        # In bipartite graph, no triangles.
        # C4: u-v-w-z-u.
        # If we add (u, v), we need to check if there is a w such that (v, w) in E and (w, z) in E and (z, u) in E?
        # Actually, it's easier to check:
        # For each neighbor w of v (in E), check if there is a common neighbor z of u and w in E.
        
        # Optimization:
        # u and v differ in bit i.
        # Any C4 containing (u, v) must involve another dimension j.
        # The C4 is u, u^e_i (v), u^e_i^e_j (w), u^e_j (z).
        # Edges: (u, v), (v, w), (w, z), (z, u).
        # We are adding (u, v).
        # We must ensure that NOT ((v, w) in E AND (w, z) in E AND (z, u) in E).
        
        ok = True
        for j in range(d):
            if j == i: continue
            w = v ^ (1 << j)
            z = u ^ (1 << j)
            
            # Check if edges (v, w), (w, z), (z, u) are in edges
            # Note: (z, u) is dim j edge. (v, w) is dim j edge. (w, z) is dim i edge.
            
            if (min(v, w), max(v, w)) in edges and \
               (min(w, z), max(w, z)) in edges and \
               (min(z, u), max(z, u)) in edges:
                ok = False
                break
                
        if ok:
            edges.add((u, v))
            
    return [list(e) for e in edges]
