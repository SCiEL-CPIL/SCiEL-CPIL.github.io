import time
import random
import numpy as np

def _build_adj_masks(colors, n):
    """Build adjacency bitmasks for both colors."""
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    full_mask = (1 << n) - 1
    adj0 = [0] * n
    for i in range(n):
        adj0[i] = full_mask ^ adj1[i] ^ (1 << i)
    
    return adj1, adj0

def _edge_index_table(n):
    """Precompute (i,j) -> index and index -> (i,j) mappings."""
    forward = {}
    reverse = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            forward[(i, j)] = idx
            reverse.append((i, j))
            idx += 1
    return forward, reverse

def _has_k222_mask(adj, n):
    """
    Check if graph with bitmask adjacency 'adj' on n vertices contains K2,2,2.
    K2,2,2 is complete tripartite with parts of size 2.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            
            C = adj[u] & adj[v]
            if C.bit_count() < 4:
                continue
            
            vertices_in_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                v_idx = lsb.bit_length() - 1
                vertices_in_c.append(v_idx)
                mask ^= lsb
                
            m = len(vertices_in_c)
            for i in range(m):
                a = vertices_in_c[i]
                for j in range(i + 1, m):
                    b = vertices_in_c[j]
                    if adj[a] & (1 << b):
                        continue
                    
                    neighbors_ab_in_c = (adj[a] & adj[b] & C) & ~(1 << a) & ~(1 << b)
                    if neighbors_ab_in_c.bit_count() < 2:
                        continue
                    
                    verts_ab = []
                    mask2 = neighbors_ab_in_c
                    while mask2:
                        lsb = mask2 & -mask2
                        v_idx = lsb.bit_length() - 1
                        verts_ab.append(v_idx)
                        mask2 ^= lsb
                        
                    k = len(verts_ab)
                    for x in range(k):
                        c = verts_ab[x]
                        for y in range(x + 1, k):
                            d = verts_ab[y]
                            if not (adj[c] & (1 << d)):
                                return True
    return False

def _count_k222_fast(adj, n):
    """Count K2,2,2 subgraphs using bitmask operations. Returns count."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            C = adj[u] & adj[v]
            if C.bit_count() < 4:
                continue
            # Enumerate non-edges (a,b) in C
            verts_c = []
            mask = C
            while mask:
                lsb = mask & -mask
                verts_c.append(lsb.bit_length() - 1)
                mask ^= lsb
            m = len(verts_c)
            for ii in range(m):
                a = verts_c[ii]
                for jj in range(ii + 1, m):
                    b = verts_c[jj]
                    if adj[a] & (1 << b):
                        continue
                    D = (adj[a] & adj[b] & C) & ~(1 << a) & ~(1 << b)
                    if D.bit_count() < 2:
                        continue
                    # Count non-edges in D
                    verts_d = []
                    mask2 = D
                    while mask2:
                        lsb = mask2 & -mask2
                        verts_d.append(lsb.bit_length() - 1)
                        mask2 ^= lsb
                    k = len(verts_d)
                    for xx in range(k):
                        c = verts_d[xx]
                        for yy in range(xx + 1, k):
                            d = verts_d[yy]
                            if not (adj[c] & (1 << d)):
                                count += 1
    return count

def _count_total_violations(adj1, adj0, n):
    """Count total K2,2,2 violations in both colors."""
    return _count_k222_fast(adj1, n) + _count_k222_fast(adj0, n)

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _is_prime(x):
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True

def _count_k222_numpy(adj_mat, n):
    """
    Count K2,2,2 subgraphs using numpy matrix operations.
    adj_mat is an n x n boolean numpy array (symmetric, zero diagonal).
    K2,2,2: vertices {u,v} (non-adjacent), {a,b} (non-adjacent, both in N(u)∩N(v)),
    {c,d} (non-adjacent, both in N(a)∩N(b)∩N(u)∩N(v))
    """
    # Non-adjacency matrix (excluding self-loops)
    non_adj = ~adj_mat
    np.fill_diagonal(non_adj, False)
    
    # For each pair (u,v) non-adjacent: C = N(u) ∩ N(v)
    # adj_mat[u] & adj_mat[v] gives common neighbors
    common_neighbors = adj_mat @ adj_mat  # (i,j) = |N(i) ∩ N(j)|... no, this is count
    
    # Actually, let's use a different approach:
    # For each non-edge (u,v), compute C = adj[u] & adj[v]
    # Then for each non-edge (a,b) within C, compute D = (adj[a] & adj[b] & C) \ {a,b}
    # Count non-edges in D
    
    count = 0
    # Precompute: for all pairs, whether they are non-adjacent
    # Use broadcasting for efficiency
    
    # Iterate over non-adjacent pairs (u,v)
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mat[u, v]:
                continue
            
            # C = common neighbors of u and v
            C = adj_mat[u] & adj_mat[v]
            c_count = C.sum()
            if c_count < 4:
                continue
            
            # Get indices of vertices in C
            verts_c = np.where(C)[0]
            m = len(verts_c)
            
            # For each non-adjacent pair (a,b) in C
            # adj_mat[a,b] should be False
            # D = (adj[a] & adj[b] & C) excluding a,b
            # Count non-edges in D
            
            # Extract submatrix for verts_c
            sub_adj = adj_mat[np.ix_(verts_c, verts_c)]
            
            # Non-adjacent pairs within C
            non_adj_sub = ~sub_adj
            np.fill_diagonal(non_adj_sub, False)
            
            # For each non-adjacent pair (i,j) in submatrix:
            # D = sub_adj[i] & sub_adj[j] & C_mask, excluding i,j
            # Then count non-edges in D
            
            # Vectorize: for all pairs (i,j) with non_adj_sub[i,j]:
            # common = sub_adj[i] & sub_adj[j]
            # But we need to exclude i and j from common
            # And then count non-edges among those common neighbors
            
            # This is getting complex. Let's do a hybrid approach.
            for i in range(m):
                a = verts_c[i]
                for j in range(i + 1, m):
                    b = verts_c[j]
                    if adj_mat[a, b]:
                        continue
                    
                    # D = (adj[a] & adj[b] & C) \ {a,b}
                    D = adj_mat[a] & adj_mat[b] & C
                    D[a] = False
                    D[b] = False
                    d_count = D.sum()
                    if d_count < 2:
                        continue
                    
                    # Count non-edges in D
                    verts_d = np.where(D)[0]
                    k = len(verts_d)
                    if k < 2:
                        continue
                    sub_d = adj_mat[np.ix_(verts_d, verts_d)]
                    non_edges_d = (~sub_d).sum() // 2  # symmetric
                    # Actually need to count pairs that are non-adjacent
                    # non_adj_d = ~sub_d, fill diagonal False, count True
                    non_adj_d = ~sub_d
                    np.fill_diagonal(non_adj_d, False)
                    count += non_adj_d.sum() // 2
    return count

def _local_search_vec(colors, n, max_iters=50000, time_limit=50, temp_start=1.0, temp_end=0.01):
    """
    Vectorized local search with simulated annealing using bitmask adjacency.
    """
    start_time = time.time()
    n_edges = n * (n - 1) // 2
    edge_fwd, edge_rev = _edge_index_table(n)
    
    # Build bitmask adjacency
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    full_mask = (1 << n) - 1
    adj0 = [full_mask ^ adj1[i] ^ (1 << i) for i in range(n)]
    
    def get_total():
        return _count_k222_fast(adj1, n) + _count_k222_fast(adj0, n)
    
    cur_violations = get_total()
    if cur_violations == 0:
        return True
    
    best_violations = cur_violations
    best_colors = colors[:]
    
    it = 0
    while it < max_iters:
        if time.time() - start_time > time_limit:
            break
        
        progress = min(it / max(1, max_iters - 1), 1.0)
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Pick edge to flip
        if random.random() < 0.7:
            # Guided: find a violation and flip an edge in it
            idx_flip = -1
            for _ in range(5):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                # Check color 1 violation
                if not (adj1[u] & (1 << v)):
                    C = adj1[u] & adj1[v]
                    if C.bit_count() >= 4:
                        verts_c = []
                        mask = C
                        while mask:
                            lsb = mask & -mask
                            verts_c.append(lsb.bit_length() - 1)
                            mask ^= lsb
                        m = len(verts_c)
                        for ii in range(m):
                            a = verts_c[ii]
                            for jj in range(ii + 1, m):
                                b = verts_c[jj]
                                if adj1[a] & (1 << b):
                                    continue
                                D = (adj1[a] & adj1[b] & C) & ~(1 << a) & ~(1 << b)
                                if D.bit_count() < 2:
                                    continue
                                verts_d = []
                                mask2 = D
                                while mask2:
                                    lsb = mask2 & -mask2
                                    verts_d.append(lsb.bit_length() - 1)
                                    mask2 ^= lsb
                                k = len(verts_d)
                                for xx in range(k):
                                    c = verts_d[xx]
                                    for yy in range(xx + 1, k):
                                        d = verts_d[yy]
                                        if not (adj1[c] & (1 << d)):
                                            # Found violation in color 1
                                            candidates = []
                                            for pair in [(a,c),(a,d),(b,c),(b,d)]:
                                                p1, p2 = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                                                if adj1[p1] & (1 << p2):
                                                    candidates.append(edge_fwd[(p1, p2)])
                                            if candidates:
                                                idx_flip = random.choice(candidates)
                                                break
                                    if idx_flip >= 0:
                                        break
                            if idx_flip >= 0:
                                break
                        if idx_flip >= 0:
                            break
                # Check color 0 violation
                if idx_flip < 0 and not (adj0[u] & (1 << v)):
                    C = adj0[u] & adj0[v]
                    if C.bit_count() >= 4:
                        verts_c = []
                        mask = C
                        while mask:
                            lsb = mask & -mask
                            verts_c.append(lsb.bit_length() - 1)
                            mask ^= lsb
                        m = len(verts_c)
                        for ii in range(m):
                            a = verts_c[ii]
                            for jj in range(ii + 1, m):
                                b = verts_c[jj]
                                if adj0[a] & (1 << b):
                                    continue
                                D = (adj0[a] & adj0[b] & C) & ~(1 << a) & ~(1 << b)
                                if D.bit_count() < 2:
                                    continue
                                verts_d = []
                                mask2 = D
                                while mask2:
                                    lsb = mask2 & -mask2
                                    verts_d.append(lsb.bit_length() - 1)
                                    mask2 ^= lsb
                                k = len(verts_d)
                                for xx in range(k):
                                    c = verts_d[xx]
                                    for yy in range(xx + 1, k):
                                        d = verts_d[yy]
                                        if not (adj0[c] & (1 << d)):
                                            candidates = []
                                            for pair in [(a,c),(a,d),(b,c),(b,d)]:
                                                p1, p2 = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                                                if not (adj1[p1] & (1 << p2)):
                                                    candidates.append(edge_fwd[(p1, p2)])
                                            if candidates:
                                                idx_flip = random.choice(candidates)
                                                break
                                    if idx_flip >= 0:
                                        break
                            if idx_flip >= 0:
                                break
                        if idx_flip >= 0:
                            break
                if idx_flip >= 0:
                    break
            if idx_flip < 0:
                idx_flip = random.randint(0, n_edges - 1)
        else:
            idx_flip = random.randint(0, n_edges - 1)
        
        # Flip
        i, j = edge_rev[idx_flip]
        old_val = colors[idx_flip]
        colors[idx_flip] = 1 - old_val
        
        # Update bitmasks
        bit_i = 1 << i
        bit_j = 1 << j
        if old_val == 1:
            adj1[i] &= ~bit_j
            adj1[j] &= ~bit_i
            adj0[i] |= bit_j
            adj0[j] |= bit_i
        else:
            adj1[i] |= bit_j
            adj1[j] |= bit_i
            adj0[i] &= ~bit_j
            adj0[j] &= ~bit_i
        
        new_total = get_total()
        
        if new_total < best_violations:
            best_violations = new_total
            best_colors = colors[:]
        
        delta = new_total - cur_violations
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            cur_violations = new_total
        else:
            # Undo
            colors[idx_flip] = old_val
            if old_val == 1:
                adj1[i] |= bit_j
                adj1[j] |= bit_i
                adj0[i] &= ~bit_j
                adj0[j] &= ~bit_i
            else:
                adj1[i] &= ~bit_j
                adj1[j] &= ~bit_i
                adj0[i] |= bit_j
                adj0[j] |= bit_i
        
        it += 1
    
    colors[:] = best_colors
    adj1_final = [0] * n
    idx = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            if colors[idx] == 1:
                adj1_final[ii] |= (1 << jj)
                adj1_final[jj] |= (1 << ii)
            idx += 1
    adj0_final = [full_mask ^ adj1_final[i] ^ (1 << i) for i in range(n)]
    return _count_k222_fast(adj1_final, n) + _count_k222_fast(adj0_final, n) == 0

def _local_search(colors, n, max_iters=5000, time_limit=50):
    """Local search to find valid coloring using violation-guided flips."""
    start = time.time()
    n_edges = n * (n - 1) // 2
    edge_fwd, edge_rev = _edge_index_table(n)
    
    adj1, adj0 = _build_adj_masks(colors, n)
    if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
        return True
    
    full_mask = (1 << n) - 1
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
            return True
        
        flipped = False
        for _ in range(20):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            # Try color 1
            if not (adj1[u] & (1 << v)):
                C = adj1[u] & adj1[v]
                if C.bit_count() >= 4:
                    verts_c = []
                    mask = C
                    while mask:
                        lsb = mask & -mask
                        verts_c.append(lsb.bit_length() - 1)
                        mask ^= lsb
                    m = len(verts_c)
                    for ii in range(m):
                        a = verts_c[ii]
                        for jj in range(ii + 1, m):
                            b = verts_c[jj]
                            if adj1[a] & (1 << b):
                                continue
                            D = (adj1[a] & adj1[b] & C) & ~(1 << a) & ~(1 << b)
                            if D.bit_count() < 2:
                                continue
                            verts_d = []
                            mask2 = D
                            while mask2:
                                lsb = mask2 & -mask2
                                verts_d.append(lsb.bit_length() - 1)
                                mask2 ^= lsb
                            k = len(verts_d)
                            for xx in range(k):
                                c = verts_d[xx]
                                for yy in range(xx + 1, k):
                                    d = verts_d[yy]
                                    if not (adj1[c] & (1 << d)):
                                        candidates = []
                                        for pair in [(a,c),(a,d),(b,c),(b,d)]:
                                            p1, p2 = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                                            if adj1[p1] & (1 << p2):
                                                candidates.append(edge_fwd[(p1, p2)])
                                        if candidates:
                                            idx_f = random.choice(candidates)
                                            colors[idx_f] = 1 - colors[idx_f]
                                            i2, j2 = edge_rev[idx_f]
                                            bit_i2, bit_j2 = 1 << i2, 1 << j2
                                            if colors[idx_f] == 0:
                                                adj1[i2] &= ~bit_j2
                                                adj1[j2] &= ~bit_i2
                                                adj0[i2] |= bit_j2
                                                adj0[j2] |= bit_i2
                                            else:
                                                adj1[i2] |= bit_j2
                                                adj1[j2] |= bit_i2
                                                adj0[i2] &= ~bit_j2
                                                adj0[j2] &= ~bit_i2
                                            flipped = True
                                            break
                                if flipped:
                                    break
                        if flipped:
                            break
                    if flipped:
                        break
            # Try color 0
            if not flipped and not (adj0[u] & (1 << v)):
                C = adj0[u] & adj0[v]
                if C.bit_count() >= 4:
                    verts_c = []
                    mask = C
                    while mask:
                        lsb = mask & -mask
                        verts_c.append(lsb.bit_length() - 1)
                        mask ^= lsb
                    m = len(verts_c)
                    for ii in range(m):
                        a = verts_c[ii]
                        for jj in range(ii + 1, m):
                            b = verts_c[jj]
                            if adj0[a] & (1 << b):
                                continue
                            D = (adj0[a] & adj0[b] & C) & ~(1 << a) & ~(1 << b)
                            if D.bit_count() < 2:
                                continue
                            verts_d = []
                            mask2 = D
                            while mask2:
                                lsb = mask2 & -mask2
                                verts_d.append(lsb.bit_length() - 1)
                                mask2 ^= lsb
                            k = len(verts_d)
                            for xx in range(k):
                                c = verts_d[xx]
                                for yy in range(xx + 1, k):
                                    d = verts_d[yy]
                                    if not (adj0[c] & (1 << d)):
                                        candidates = []
                                        for pair in [(a,c),(a,d),(b,c),(b,d)]:
                                            p1, p2 = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                                            if not (adj1[p1] & (1 << p2)):
                                                candidates.append(edge_fwd[(p1, p2)])
                                        if candidates:
                                            idx_f = random.choice(candidates)
                                            colors[idx_f] = 1 - colors[idx_f]
                                            i2, j2 = edge_rev[idx_f]
                                            bit_i2, bit_j2 = 1 << i2, 1 << j2
                                            if colors[idx_f] == 0:
                                                adj1[i2] &= ~bit_j2
                                                adj1[j2] &= ~bit_i2
                                                adj0[i2] |= bit_j2
                                                adj0[j2] |= bit_i2
                                            else:
                                                adj1[i2] |= bit_j2
                                                adj1[j2] |= bit_i2
                                                adj0[i2] &= ~bit_j2
                                                adj0[j2] &= ~bit_i2
                                            flipped = True
                                            break
                                if flipped:
                                    break
                        if flipped:
                            break
                    if flipped:
                        break
            if flipped:
                break
        
        if not flipped:
            idx_f = random.randint(0, n_edges - 1)
            colors[idx_f] = 1 - colors[idx_f]
            i2, j2 = edge_rev[idx_f]
            bit_i2, bit_j2 = 1 << i2, 1 << j2
            if colors[idx_f] == 0:
                adj1[i2] &= ~bit_j2
                adj1[j2] &= ~bit_i2
                adj0[i2] |= bit_j2
                adj0[j2] |= bit_i2
            else:
                adj1[i2] |= bit_j2
                adj1[j2] |= bit_i2
                adj0[i2] &= ~bit_j2
                adj0[j2] &= ~bit_i2
    
    return not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n)

def _fast_local_search(colors, n, time_limit=50):
    """
    Fast local search using bitmask adjacency with simulated annealing.
    """
    start_time = time.time()
    n_edges = n * (n - 1) // 2
    edge_fwd, edge_rev = _edge_index_table(n)
    
    # Build bitmask adjacency
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    full_mask = (1 << n) - 1
    adj0 = [full_mask ^ adj1[i] ^ (1 << i) for i in range(n)]
    
    def get_total():
        return _count_k222_fast(adj1, n) + _count_k222_fast(adj0, n)
    
    cur_total = get_total()
    if cur_total == 0:
        return True
    
    best_total = cur_total
    best_colors = colors[:]
    
    max_iters = 200000
    temp_start = 1.0
    temp_end = 0.001
    it = 0
    
    while it < max_iters:
        if time.time() - start_time > time_limit:
            break
        
        progress = min(it / max(1, max_iters - 1), 1.0)
        temp = temp_start * (temp_end / temp_start) ** progress
        
        # Choose edge to flip
        idx_flip = -1
        if random.random() < 0.65:
            # Guided
            for _ in range(3):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if not (adj1[u] & (1 << v)):
                    C = adj1[u] & adj1[v]
                    if C.bit_count() >= 4:
                        verts_c = []
                        mask = C
                        while mask:
                            lsb = mask & -mask
                            verts_c.append(lsb.bit_length() - 1)
                            mask ^= lsb
                        m = len(verts_c)
                        for ii in range(m):
                            a = verts_c[ii]
                            for jj in range(ii + 1, m):
                                b = verts_c[jj]
                                if adj1[a] & (1 << b):
                                    continue
                                D = (adj1[a] & adj1[b] & C) & ~(1 << a) & ~(1 << b)
                                if D.bit_count() < 2:
                                    continue
                                verts_d = []
                                mask2 = D
                                while mask2:
                                    lsb = mask2 & -mask2
                                    verts_d.append(lsb.bit_length() - 1)
                                    mask2 ^= lsb
                                k = len(verts_d)
                                for xx in range(k):
                                    c = verts_d[xx]
                                    for yy in range(xx + 1, k):
                                        d = verts_d[yy]
                                        if not (adj1[c] & (1 << d)):
                                            candidates = []
                                            for pair in [(a,c),(a,d),(b,c),(b,d)]:
                                                p1, p2 = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                                                if adj1[p1] & (1 << p2):
                                                    candidates.append(edge_fwd[(p1, p2)])
                                            if candidates:
                                                idx_flip = random.choice(candidates)
                                                break
                                    if idx_flip >= 0:
                                        break
                            if idx_flip >= 0:
                                break
                        if idx_flip >= 0:
                            break
                if idx_flip < 0 and not (adj0[u] & (1 << v)):
                    C = adj0[u] & adj0[v]
                    if C.bit_count() >= 4:
                        verts_c = []
                        mask = C
                        while mask:
                            lsb = mask & -mask
                            verts_c.append(lsb.bit_length() - 1)
                            mask ^= lsb
                        m = len(verts_c)
                        for ii in range(m):
                            a = verts_c[ii]
                            for jj in range(ii + 1, m):
                                b = verts_c[jj]
                                if adj0[a] & (1 << b):
                                    continue
                                D = (adj0[a] & adj0[b] & C) & ~(1 << a) & ~(1 << b)
                                if D.bit_count() < 2:
                                    continue
                                verts_d = []
                                mask2 = D
                                while mask2:
                                    lsb = mask2 & -mask2
                                    verts_d.append(lsb.bit_length() - 1)
                                    mask2 ^= lsb
                                k = len(verts_d)
                                for xx in range(k):
                                    c = verts_d[xx]
                                    for yy in range(xx + 1, k):
                                        d = verts_d[yy]
                                        if not (adj0[c] & (1 << d)):
                                            candidates = []
                                            for pair in [(a,c),(a,d),(b,c),(b,d)]:
                                                p1, p2 = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                                                if not (adj1[p1] & (1 << p2)):
                                                    candidates.append(edge_fwd[(p1, p2)])
                                            if candidates:
                                                idx_flip = random.choice(candidates)
                                                break
                                    if idx_flip >= 0:
                                        break
                            if idx_flip >= 0:
                                break
                        if idx_flip >= 0:
                            break
                if idx_flip >= 0:
                    break
        if idx_flip < 0:
            idx_flip = random.randint(0, n_edges - 1)
        
        # Flip
        i, j = edge_rev[idx_flip]
        old_val = colors[idx_flip]
        colors[idx_flip] = 1 - old_val
        bit_i = 1 << i
        bit_j = 1 << j
        if old_val == 1:
            adj1[i] &= ~bit_j
            adj1[j] &= ~bit_i
            adj0[i] |= bit_j
            adj0[j] |= bit_i
        else:
            adj1[i] |= bit_j
            adj1[j] |= bit_i
            adj0[i] &= ~bit_j
            adj0[j] &= ~bit_i
        
        new_total = get_total()
        
        if new_total < best_total:
            best_total = new_total
            best_colors = colors[:]
        
        delta = new_total - cur_total
        if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            cur_total = new_total
        else:
            colors[idx_flip] = old_val
            if old_val == 1:
                adj1[i] |= bit_j
                adj1[j] |= bit_i
                adj0[i] &= ~bit_j
                adj0[j] &= ~bit_i
            else:
                adj1[i] &= ~bit_j
                adj1[j] &= ~bit_i
                adj0[i] |= bit_j
                adj0[j] |= bit_i
        
        it += 1
    
    colors[:] = best_colors
    adj1_final = [0] * n
    idx = 0
    for ii in range(n):
        for jj in range(ii + 1, n):
            if colors[idx] == 1:
                adj1_final[ii] |= (1 << jj)
                adj1_final[jj] |= (1 << ii)
            idx += 1
    adj0_final = [full_mask ^ adj1_final[i] ^ (1 << i) for i in range(n)]
    return _count_k222_fast(adj1_final, n) + _count_k222_fast(adj0_final, n) == 0

def construct():
    deadline = time.time() + 135 # Safety margin (150s limit, leave 15s buffer)
    
    # Try Paley constructions
    # Paley(29) is a strong candidate for R(K2,2,2)-1
    
    # First, try to get n=29 using Paley(29)
    n_target = 29
    p_target = 29
    colors = _paley(n_target, p_target)
    if colors:
        adj1, adj0 = _build_adj_masks(colors, n_target)
        if not _has_k222_mask(adj1, n_target) and not _has_k222_mask(adj0, n_target):
            return {'n': n_target, 'colors': colors}
        else:
            # Try fast local search starting from Paley
            remaining = deadline - time.time()
            if remaining > 15:
                if _fast_local_search(colors, n_target, time_limit=int(remaining - 10)):
                    return {'n': n_target, 'colors': colors}

    # If Paley(29) failed, try other primes for n=29
    primes = [31, 37, 41, 43, 47, 53, 59]
    for p in primes:
        if time.time() > deadline - 15:
            break
        colors = _paley(29, p)
        if colors is None:
            continue
        adj1, adj0 = _build_adj_masks(colors, 29)
        if not _has_k222_mask(adj1, 29) and not _has_k222_mask(adj0, 29):
            return {'n': 29, 'colors': colors}
        remaining = deadline - time.time()
        if remaining > 20:
            if _fast_local_search(colors, 29, time_limit=int(remaining - 10)):
                return {'n': 29, 'colors': colors}

    # Try random initializations for n=29
    for _ in range(10):
        if time.time() > deadline - 15:
            break
        colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
        remaining = deadline - time.time()
        if remaining > 15:
            if _fast_local_search(colors, 29, time_limit=int(remaining - 10)):
                return {'n': 29, 'colors': colors}

    # Fallback: Try to find valid coloring for smaller n
    for n in range(28, 16, -1):
        if time.time() > deadline - 10:
            break
        # Try Paley with p=n if n is prime and 1 mod 4, else next prime
        p = n
        while p < 31:
            is_prime = True
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    is_prime = False
                    break
            if is_prime and p % 4 == 1:
                break
            p += 1
            
        colors = _paley(n, p)
        if colors:
            adj1, adj0 = _build_adj_masks(colors, n)
            if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
                return {'n': n, 'colors': colors}
            remaining = deadline - time.time()
            if remaining > 10:
                if _fast_local_search(colors, n, time_limit=int(remaining - 5)):
                    return {'n': n, 'colors': colors}
                
        # Random start
        colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        remaining = deadline - time.time()
        if remaining > 10:
            if _fast_local_search(colors, n, time_limit=int(remaining - 5)):
                return {'n': n, 'colors': colors}

    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}
