import time
import random
import math

def _creates_c4_fast(adj_int, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if C4 would form."""
    nu = adj_int[u]
    nv = adj_int[v]
    
    # C4 exists iff there exist x in N(u) \ {v}, y in N(v) \ {u} with x~y
    # and x != y, x != u, y != v
    # Equivalently: (common neighbors of u and v) is non-empty, OR
    # there exist x in N(u), y in N(v) with x~y, x != y, x != v, y != u
    
    # Fast check: if u and v have a common neighbor, adding (u,v) creates a C4
    # only if that common neighbor is NOT u or v (which it can't be since no self-loops)
    # Actually, if w is a common neighbor of u and v, then u-w-v-? no that's a path not C4
    # C4: u-x-v-y-u where x~y. So we need x in N(u)\N(v) and y in N(v)\N(u) with x~y,
    # OR x in N(u) and y in N(v) with x~y, x!=y, x!=v, y!=u
    
    # The condition is: there exist x, y such that x in N(u), y in N(v), x~y,
    # x != y, x != v, y != u
    # This is equivalent to: (N(u) ∩ N(y)) != empty for some y in N(v) \ {u},
    # where we also need x != y and x != v
    
    # Simpler: iterate over the smaller neighborhood
    deg_u = bin(nu).count('1')
    deg_v = bin(nv).count('1')
    
    if deg_u <= deg_v:
        # For each x in N(u), check if N(x) intersects N(v) \ {u, x}
        temp = nu
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            # Check if any neighbor of x is in N(v) and is not u and not x
            # N(x) & N(v) & ~(1<<u) & ~(1<<x)
            if adj_int[x] & nv & ~(1 << u) & ~(1 << x):
                return True
            temp ^= lsb
        return False
    else:
        temp = nv
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            if adj_int[y] & nu & ~(1 << v) & ~(1 << y):
                return True
            temp ^= lsb
        return False

def _build_polarity_graph(n, seed):
    """
    Build a polarity graph on n vertices.
    A polarity graph on a finite field or using quadratic forms is C4-free.
    We use the standard construction: vertices are 0..n-1, 
    edge (i,j) if (i-j)^2 is a quadratic non-residue mod p (for prime p).
    For non-prime n, we use a simpler approach with a random symmetric matrix
    that avoids C4 by construction using a known C4-free structure.
    
    Actually, let's use a simpler deterministic C4-free seed:
    Use a bipartite-like structure or a known C4-free graph.
    
    Better approach: Use the polarity graph over GF(q) where q is a prime power.
    For n vertices, find the smallest prime p >= n, build polarity graph on p vertices,
    then take the first n vertices.
    
    For simplicity and robustness, let's use a different approach:
    Start with a random graph and remove C4s, or use a known C4-free construction.
    
    Let me use the following: For n up to 47, I'll use a greedy construction
    but with a smarter initial seeding using a bipartite graph structure.
    """
    adj_int = [0] * n
    degrees = [0] * n
    
    # Strategy: Use a "blow-up" of a small C4-free graph, or
    # use a random graph and iteratively remove C4-creating edges.
    
    # Better: Use the fact that polarity graphs are C4-free.
    # For a prime p, the polarity graph has vertices 0..p-1,
    # and (i,j) is an edge if (i-j)^2 is a quadratic non-residue mod p,
    # or i=j and -1 is a non-residue (self-loop, which we ignore for simple graphs).
    
    # Find smallest prime >= n
    def is_prime(m):
        if m < 2:
            return False
        if m == 2:
            return True
        if m % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(m)) + 1, 2):
            if m % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Build quadratic residues mod p
    qr = set()
    for x in range(p):
        qr.add((x * x) % p)
    
    # Build polarity graph on p vertices
    # Edge (i,j) for i < j if (i-j)^2 mod p is NOT in qr (i.e., quadratic non-residue)
    # Note: (i-j)^2 = (j-i)^2, so this is symmetric.
    # Also, for i=j, (i-i)^2 = 0, which is in qr, so no self-loop.
    
    polarity_adj = [0] * p
    for i in range(p):
        for j in range(i+1, p):
            diff = (i - j) % p
            sq = (diff * diff) % p
            if sq not in qr:
                polarity_adj[i] |= (1 << j)
                polarity_adj[j] |= (1 << i)
    
    # Take first n vertices
    for i in range(n):
        adj_int[i] = polarity_adj[i] & ((1 << n) - 1)
        degrees[i] = bin(adj_int[i]).count('1')
    
    return adj_int, degrees

def _local_search(adj_int, n, min_deg_required, time_limit, seed):
    """
    Local search to maximize minimum degree while maintaining C4-free property.
    Uses aggressive edge additions with multiple attempts per vertex.
    """
    start_time = time.time()
    degrees = [bin(adj_int[i]).count('1') for i in range(n)]
    
    best_min_deg = min(degrees)
    best_adj = adj_int[:]
    
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Check if we've reached the target
        if min(degrees) >= min_deg_required:
            break
        
        # Try to add edges to improve minimum degree
        # Focus on ALL vertices below target, prioritizing lowest
        sorted_verts = sorted(range(n), key=lambda i: degrees[i])
        
        improved = False
        for u in sorted_verts:
            if time.time() - start_time > time_limit:
                break
            if degrees[u] >= min_deg_required:
                continue
            
            # Find non-neighbors of u
            non_neighbors = []
            mask = (1 << n) - 1
            non_neigh_mask = mask & ~(adj_int[u] | (1 << u))
            temp = non_neigh_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                non_neighbors.append(v)
                temp ^= lsb
            
            random.shuffle(non_neighbors)
            
            # Try ALL non-neighbors, add as many as possible
            for v in non_neighbors:
                if time.time() - start_time > time_limit:
                    break
                if degrees[u] >= min_deg_required:
                    break
                if _creates_c4_fast(adj_int, n, u, v):
                    continue
                # Add edge (u, v)
                adj_int[u] |= (1 << v)
                adj_int[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                improved = True
        
        if not improved:
            # Try 2-swap: remove (a,b), add (c,d) where this helps min degree
            # Find the bottleneck vertex
            low_vert = min(range(n), key=lambda i: degrees[i])
            if degrees[low_vert] >= min_deg_required:
                break
            
            # Try to find a swap that adds an edge to low_vert
            # Remove an edge (x,y) not incident to low_vert, then add (low_vert, z)
            # where z was a neighbor of x or y but not of low_vert
            
            # Get all edges
            edge_list = []
            for a in range(n):
                temp = adj_int[a] & ((1 << a) - 1)  # only b < a
                while temp:
                    lsb = temp & -temp
                    b = lsb.bit_length() - 1
                    edge_list.append((a, b))
                    temp ^= lsb
            
            random.shuffle(edge_list)
            
            swapped = False
            for (x, y) in edge_list[:200]:
                if time.time() - start_time > time_limit:
                    break
                # Remove edge (x, y)
                adj_int[x] ^= (1 << y)
                adj_int[y] ^= (1 << x)
                degrees[x] -= 1
                degrees[y] -= 1
                
                # Try to add (low_vert, z) for z in N(x) ∪ N(y) that's not already connected
                candidates = set()
                temp_x = adj_int[x]
                while temp_x:
                    lsb = temp_x & -temp_x
                    candidates.add(lsb.bit_length() - 1)
                    temp_x ^= lsb
                temp_y = adj_int[y]
                while temp_y:
                    lsb = temp_y & -temp_y
                    candidates.add(lsb.bit_length() - 1)
                    temp_y ^= lsb
                
                # Also try z = x and z = y themselves
                candidates.add(x)
                candidates.add(y)
                
                cand_list = [z for z in candidates if z != low_vert and not (adj_int[low_vert] & (1 << z))]
                random.shuffle(cand_list)
                
                found = False
                for z in cand_list[:10]:
                    if not _creates_c4_fast(adj_int, n, low_vert, z):
                        adj_int[low_vert] |= (1 << z)
                        adj_int[z] |= (1 << low_vert)
                        degrees[low_vert] += 1
                        degrees[z] += 1
                        found = True
                        break
                
                if not found:
                    # Restore edge (x, y)
                    adj_int[x] |= (1 << y)
                    adj_int[y] |= (1 << x)
                    degrees[x] += 1
                    degrees[y] += 1
                else:
                    swapped = True
                    break
            
            if not swapped:
                # No improvement possible, break
                break
        
        # Track best
        if min(degrees) > best_min_deg:
            best_min_deg = min(degrees)
            best_adj = adj_int[:]
    
    return best_adj, [bin(best_adj[i]).count('1') for i in range(n)]

def _construct_c4_free(n, min_deg_required, time_limit, seed):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_required."""
    random.seed(seed)
    start_time = time.time()
    
    # Try multiple polarity graph seeds with more attempts
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() - start_time > time_limit:
            break
        
        adj_int, degrees = _build_polarity_graph(n, seed + attempt)
        
        # Check if already good enough
        if min(degrees) >= min_deg_required:
            edges = []
            for u in range(n):
                temp = adj_int[u]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    temp ^= lsb
            return {'n': n, 'edges': edges}
        
        # Apply local search with more time
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            continue
        
        # Give more time to local search, especially for early attempts
        t_ls = min(remaining * 0.8, 8.0)
        best_adj, best_degrees = _local_search(adj_int, n, min_deg_required, t_ls, seed + attempt)
        
        if min(best_degrees) >= min_deg_required:
            edges = []
            for u in range(n):
                temp = best_adj[u]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    temp ^= lsb
            return {'n': n, 'edges': edges}
    
    return None

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Try n=47, 46, 45, ...
    # n=47 -> min_deg 8
    # n=46 -> min_deg 7
    # n=45 -> min_deg 6
    
    # Allocate more time to higher n values
    time_allocation = {
        47: 40.0,
        46: 35.0,
        45: 20.0,
        44: 10.0,
        43: 5.0,
    }
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate time based on n
        t_limit = min(time_allocation.get(n, 5.0), remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit, 42)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
