import time
import random

def _is_c4_free(n, adj_bitsets):
    """Check if graph is C4-free using bitsets."""
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_bitsets[u] >> v) & 1:
                continue
            common = adj_bitsets[u] & adj_bitsets[v]
            # If common has >= 2 bits, C4 exists
            if common & (common - 1):
                return False
    return True

def _min_degree_from_bitsets(n, adj_bitsets):
    """Compute minimum degree from bitsets."""
    min_deg = n
    for i in range(n):
        deg = bin(adj_bitsets[i]).count('1')
        if deg < min_deg:
            min_deg = deg
    return min_deg

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Strategy 1: Structured Construction using Generalized Quadrangle Q+(3,7)
    # Q+(3,7) is a C4-free graph on 50 vertices with degree 28.
    # Taking an induced subgraph of 47 vertices yields min degree >= 25.
    # Required min degree for n=47 is 47-39=8. 25 >= 8.
    
    q = 7
    # Generate all projective points in F_q^4
    proj_points = []
    seen = set()
    for x0 in range(q):
        for x1 in range(q):
            for x2 in range(q):
                for x3 in range(q):
                    if x0 == 0 and x1 == 0 and x2 == 0 and x3 == 0:
                        continue
                    # Normalize: make first non-zero component 1
                    coords = (x0, x1, x2, x3)
                    idx = 0
                    while idx < 4 and coords[idx] == 0:
                        idx += 1
                    # coords[idx] is non-zero
                    inv = pow(coords[idx], q-2, q)
                    nx0 = (x0 * inv) % q
                    nx1 = (x1 * inv) % q
                    nx2 = (x2 * inv) % q
                    nx3 = (x3 * inv) % q
                    pt = (nx0, nx1, nx2, nx3)
                    if pt not in seen:
                        seen.add(pt)
                        proj_points.append(pt)
    
    # Try different diagonal quadratic forms to find Q+(3,7)
    # Q+(3,q) has q^2+1 points.
    # For q=7, we expect 50 points.
    candidates = [
        (1, 1, 1, 1),
        (1, 1, -1, -1),
        (1, 1, 1, -1),
        (1, -1, 1, -1),
        (1, 1, -1, 1),
        (1, -1, -1, 1),
        (-1, 1, 1, 1),
        (1, -1, 1, 1),
    ]
    
    for coeffs in candidates:
        if time.time() > deadline:
            break
        
        # Find points on the quadric Q(x) = sum c_i x_i^2 = 0
        quad_points = []
        for p in proj_points:
            val = 0
            for i in range(4):
                val = (val + coeffs[i] * p[i] * p[i]) % q
            if val == 0:
                quad_points.append(p)
        
        # Q+(3,q) should have q^2+1 points
        if len(quad_points) != q*q + 1:
            continue
        
        n_full = len(quad_points)
        adj_full = [0] * n_full
        
        # Build adjacency: u ~ v iff B(u,v) = 0
        for i in range(n_full):
            for j in range(i+1, n_full):
                b_val = 0
                for k in range(4):
                    b_val = (b_val + coeffs[k] * quad_points[i][k] * quad_points[j][k]) % q
                if b_val == 0:
                    adj_full[i] |= 1 << j
                    adj_full[j] |= 1 << i
        
        # Verify C4-free property
        if _is_c4_free(n_full, adj_full):
            # We have a valid C4-free graph on 50 vertices.
            # We want n=47.
            target_n = 47
            sub_adj = adj_full[:target_n]
            
            # Check min degree
            min_deg = _min_degree_from_bitsets(target_n, sub_adj)
            
            if min_deg >= target_n - 39:
                # Construct edges list
                edges = []
                for u in range(target_n):
                    a_mask = sub_adj[u]
                    while a_mask:
                        v_bit = a_mask & (-a_mask)
                        v = v_bit.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        a_mask ^= v_bit
                return {'n': target_n, 'edges': edges}

    # Fallback: Random Search (kept from previous version but optimized)
    best_n = 39
    best_edges = []
    
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg_needed = target_n - 39
        
        for seed in range(100):
            if time.time() > deadline:
                break
            random.seed(seed)
            adj = [0] * target_n
            deg = [0] * target_n
            
            all_edges = [(u, v) for u in range(target_n) for v in range(u+1, target_n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check C4
                creates_c4 = False
                A = adj[v] & ~(1 << u)
                B = adj[u] & ~(1 << v)
                a_mask = A
                while a_mask:
                    w_bit = a_mask & (-a_mask)
                    w = w_bit.bit_length() - 1
                    a_mask ^= w_bit
                    if adj[w] & B & ~(1 << w):
                        creates_c4 = True
                        break
                if not creates_c4:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
            
            if min(deg) >= min_deg_needed:
                if _is_c4_free(target_n, adj):
                    edges = []
                    for u in range(target_n):
                        a_mask = adj[u]
                        while a_mask:
                            v_bit = a_mask & (-a_mask)
                            v = v_bit.bit_length() - 1
                            if u < v:
                                edges.append([u, v])
                            a_mask ^= v_bit
                    best_n = target_n
                    best_edges = edges
                    break
        
        if best_n == target_n:
            break

    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
