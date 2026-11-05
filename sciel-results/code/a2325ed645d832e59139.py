import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Helper: Check if adding (u,v) creates a C4
    def creates_c4(adj, u, v):
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            x = lsb.bit_length() - 1
            mask_u ^= lsb
            if adj[x] & adj[v]:
                return True
        return False

    # Helper: Get min degree
    def min_degree(adj):
        return min(bin(a).count('1') for a in adj)

    # Helper: Get edges list
    def get_edges(adj):
        edges = []
        for u in range(len(adj)):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                mask ^= lsb
                if u < v:
                    edges.append([u, v])
        return edges

    # 1. Construct Projective Plane of Order 4 (PG(2,4))
    # n = 42, min degree = 5. Condition: 5 >= 42 - 39 = 3. Valid.
    q = 4
    # Points: (x, y, 1) and (x, 1, 0) etc?
    # Standard representation of PG(2, q)
    # Points: 
    # 1. (1, i, j) for i,j in GF(q) -> q^2 points
    # 2. (0, 1, i) for i in GF(q) -> q points
    # 3. (0, 0, 1) -> 1 point
    # Total q^2 + q + 1 points.
    
    # We need field arithmetic for GF(4).
    # GF(4) = {0, 1, alpha, alpha^2} where alpha^2 + alpha + 1 = 0.
    # 1, 2, 3 representing 0, 1, alpha, alpha+1?
    # Let's use integers 0..3 with custom add/mul.
    
    def gf_add(a, b):
        return a ^ b
        
    def gf_mul(a, b):
        # alpha^2 = alpha + 1
        # 0=0, 1=1, 2=alpha, 3=alpha+1
        # Table:
        # 0*x=0
        # 1*x=x
        # 2*2=3, 2*3=2, 3*3=1
        table = [
            [0,0,0,0],
            [0,1,2,3],
            [0,2,3,2],
            [0,3,2,1]
        ]
        return table[a][b]

    # Generate Points
    points = []
    # Type 1: (1, i, j)
    for i in range(q):
        for j in range(q):
            points.append((1, i, j))
    # Type 2: (0, 1, i)
    for i in range(q):
        points.append((0, 1, i))
    # Type 3: (0, 0, 1)
    points.append((0, 0, 1))
    
    # Generate Lines
    # Lines are sets of 3 points.
    # A line is defined by equation ax + by + cz = 0.
    # We can generate all lines by taking pairs of points and finding the 3rd.
    # Or iterate over all non-zero (a,b,c) up to scalar.
    
    lines = set()
    # Iterate over all triples (a,b,c) not all 0, unique up to scaling
    # Since q=4, we can just generate all lines by taking 2 points and finding the 3rd.
    
    def get_line(p1, p2):
        # p1, p2 are tuples (x,y,z)
        # Cross product in GF(4)
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        a = gf_mul(y1, z2) ^ gf_mul(y2, z1)
        b = gf_mul(z1, x2) ^ gf_mul(z2, x1)
        c = gf_mul(x1, y2) ^ gf_mul(x2, y1)
        return (a, b, c)

    # Find all unique lines
    # A line is a set of points.
    # We can represent a line by its sorted tuple of point indices.
    
    point_indices = {p: i for i, p in enumerate(points)}
    
    line_set = set()
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            l = get_line(points[i], points[j])
            # Normalize line equation? 
            # Just find all points on this line.
            pts_on_line = []
            for k in range(len(points)):
                x, y, z = points[k]
                val = gf_add(gf_add(gf_mul(l[0], x), gf_mul(l[1], y)), gf_mul(l[2], z))
                if val == 0:
                    pts_on_line.append(k)
            # A line in PG(2,q) has q+1 points.
            if len(pts_on_line) == q + 1:
                line_tuple = tuple(sorted(pts_on_line))
                line_set.add(line_tuple)
                
    # Construct Incidence Graph
    n_points = len(points)
    n_lines = len(line_set)
    n = n_points + n_lines
    
    adj = [0] * n
    
    # Connect points to lines
    for l_idx, line in enumerate(line_set):
        line_idx = n_points + l_idx
        for p_idx in line:
            adj[p_idx] |= (1 << line_idx)
            adj[line_idx] |= (1 << p_idx)
            
    # This graph is bipartite, regular degree q+1 = 5.
    # n = 42. Min degree 5.
    best_n = n
    best_edges = get_edges(adj)
    
    # 2. Try to improve with Local Search for n > 42
    # Targets: 47, 46, 45...
    
    for target_n in [47, 46, 45]:
        if time.time() > deadline:
            break
            
        target_min_deg = target_n - 39
        if target_min_deg < 1:
            target_min_deg = 1
            
        found = False
        
        # Multiple restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            adj = [0] * target_n
            
            # Greedy Construction
            edges_to_consider = [(u, v) for u in range(target_n) for v in range(u+1, target_n)]
            random.shuffle(edges_to_consider)
            
            for u, v in edges_to_consider:
                if time.time() > deadline:
                    break
                if not creates_c4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check
            if min_degree(adj) >= target_min_deg:
                found = True
                break
            
            # Local Search
            # Try to increase min degree
            for step in range(2000):
                if time.time() > deadline:
                    break
                    
                curr_min_deg = min_degree(adj)
                if curr_min_deg >= target_min_deg:
                    found = True
                    break
                    
                # Identify low degree vertices
                low_verts = [i for i in range(target_n) if bin(adj[i]).count('1') == curr_min_deg]
                u = random.choice(low_verts)
                
                # Try to add edge to u
                candidates = [v for v in range(target_n) if v != u and not ((adj[u] >> v) & 1)]
                if not candidates:
                    continue
                    
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    if not creates_c4(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        break
                        
                if not added:
                    # Try to remove an edge incident to u to unblock?
                    # Or just skip
                    pass

            if found:
                break
                
        if found:
            best_n = target_n
            best_edges = get_edges(adj)
            # If we found 47, we are done.
            if target_n == 47:
                break

    return {'n': best_n, 'edges': best_edges}
