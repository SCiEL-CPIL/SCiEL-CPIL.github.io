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

def _count_k222_violations(adj1, adj0, n):
    """Count total K2,2,2 violations in both colors. Returns (count1, count0)."""
    count1 = _count_k222_in_adj(adj1, n)
    count0 = _count_k222_in_adj(adj0, n)
    return count1, count0

def _count_k222_in_adj(adj, n):
    """Count number of K2,2,2 subgraphs in a graph."""
    count = 0
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
                                count += 1
    return count

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

def _local_search(colors, n, max_iters=5000, time_limit=50):
    """Local search to find valid coloring using violation-guided flips."""
    start = time.time()
    
    # Check initial
    adj1, adj0 = _build_adj_masks(colors, n)
    if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
        return True
    
    # Precompute edge index mapping
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Rebuild adjacency
        adj1, adj0 = _build_adj_masks(colors, n)
        
        # Check if valid
        if not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n):
            return True
        
        # Find a violating edge and flip it
        # Strategy: try to find an edge involved in a K2,2,2 violation
        flipped = False
        attempts = 0
        max_attempts = min(50, max_iters - it)
        
        while not flipped and attempts < max_attempts:
            attempts += 1
            
            # Randomly pick a non-edge in color 1 (potential P1 pair)
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj1[u] & (1 << v)):
                continue
            
            C = adj1[u] & adj1[v]
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
            if m < 4:
                continue
            
            # Try to find a violation
            for i in range(m):
                a = vertices_in_c[i]
                for j in range(i + 1, m):
                    b = vertices_in_c[j]
                    if adj1[a] & (1 << b):
                        continue
                    
                    neighbors_ab_in_c = (adj1[a] & adj1[b] & C) & ~(1 << a) & ~(1 << b)
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
                            if not (adj1[c] & (1 << d)):
                                # Found a K2,2,2 violation in color 1!
                                # Flip one of the edges in this violation
                                # Options: flip edge (u,v) to break P1, or flip an edge within P2/P3
                                # Flip a random edge from the violation
                                candidates = []
                                # Edge (u,v) - but it's a non-edge in color 1, so flipping makes it color 1
                                # Actually we want to break the violation
                                # Flip edge (a,c) - if it exists in color 1
                                if adj1[a] & (1 << c):
                                    key = (a, c) if a < c else (c, a)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                if adj1[a] & (1 << d):
                                    key = (a, d) if a < d else (d, a)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                if adj1[b] & (1 << c):
                                    key = (b, c) if b < c else (c, b)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                if adj1[b] & (1 << d):
                                    key = (b, d) if b < d else (d, b)
                                    if key in edge_idx:
                                        candidates.append(edge_idx[key])
                                
                                if candidates:
                                    idx_to_flip = random.choice(candidates)
                                    colors[idx_to_flip] = 1 - colors[idx_to_flip]
                                    flipped = True
                                    break
                    if flipped:
                        break
                if flipped:
                    break
            if flipped:
                break
        
        # If no violation found in color 1, check color 0
        if not flipped:
            for u in range(n):
                for v in range(u + 1, n):
                    if adj0[u] & (1 << v):
                        continue
                    
                    C = adj0[u] & adj0[v]
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
                    if m < 4:
                        continue
                    
                    for i in range(m):
                        a = vertices_in_c[i]
                        for j in range(i + 1, m):
                            b = vertices_in_c[j]
                            if adj0[a] & (1 << b):
                                continue
                            
                            neighbors_ab_in_c = (adj0[a] & adj0[b] & C) & ~(1 << a) & ~(1 << b)
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
                                    if not (adj0[c] & (1 << d)):
                                        # Found violation in color 0
                                        # Flip an edge that's color 0 to break it
                                        candidates = []
                                        if not (adj1[a] & (1 << c)):
                                            key = (a, c) if a < c else (c, a)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        if not (adj1[a] & (1 << d)):
                                            key = (a, d) if a < d else (d, a)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        if not (adj1[b] & (1 << c)):
                                            key = (b, c) if b < c else (c, b)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        if not (adj1[b] & (1 << d)):
                                            key = (b, d) if b < d else (d, b)
                                            if key in edge_idx:
                                                candidates.append(edge_idx[key])
                                        
                                        if candidates:
                                            idx_to_flip = random.choice(candidates)
                                            colors[idx_to_flip] = 1 - colors[idx_to_flip]
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
        
        # If still no flip, do a random flip
        if not flipped:
            idx_to_flip = random.randint(0, len(colors) - 1)
            colors[idx_to_flip] = 1 - colors[idx_to_flip]
    
    # Final check
    adj1, adj0 = _build_adj_masks(colors, n)
    return not _has_k222_mask(adj1, n) and not _has_k222_mask(adj0, n)

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
            # Try local search starting from Paley
            remaining = deadline - time.time()
            if remaining > 15:
                if _local_search(colors, n_target, max_iters=8000, time_limit=int(remaining - 10)):
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
            if _local_search(colors, 29, max_iters=3000, time_limit=int(remaining - 10)):
                return {'n': 29, 'colors': colors}

    # Try random initializations for n=29
    for _ in range(15):
        if time.time() > deadline - 15:
            break
        colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
        remaining = deadline - time.time()
        if remaining > 15:
            if _local_search(colors, 29, max_iters=1500, time_limit=int(remaining - 10)):
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
                if _local_search(colors, n, max_iters=800, time_limit=int(remaining - 5)):
                    return {'n': n, 'colors': colors}
                
        # Random start
        colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        remaining = deadline - time.time()
        if remaining > 10:
            if _local_search(colors, n, max_iters=800, time_limit=int(remaining - 5)):
                return {'n': n, 'colors': colors}

    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}
