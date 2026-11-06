import time
import random

def _creates_c4(adj_int, n, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    adj_int: list of integers representing adjacency.
    u, v: vertices to connect. Assumes not already connected.
    """
    nu = adj_int[u]
    nv = adj_int[v]
    
    # Check common neighbors: if |N(u) & N(v)| >= 2, adding (u,v) creates C4
    common = nu & nv
    if common & (common - 1):  # More than 1 bit set
        return True
        
    # Check edges between N(u) and N(v)
    # For each x in N(u), check if N(x) intersects N(v) excluding x itself
    # We iterate over the smaller set to be efficient
    deg_u = bin(nu).count('1')
    deg_v = bin(nv).count('1')
    
    if deg_u <= deg_v:
        temp = nu
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            if adj_int[x] & (nv & ~(1 << x)):
                return True
            temp ^= lsb
    else:
        temp = nv
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            if adj_int[y] & (nu & ~(1 << y)):
                return True
            temp ^= lsb
            
    return False

def _find_and_remove_c4(adj_int, degrees, n):
    """
    Find a C4 in the graph and remove an edge to break it.
    Returns True if a C4 was found and an edge removed, False otherwise.
    """
    # A C4 exists if there is a pair (u,v) with |N(u) ∩ N(v)| >= 2
    # The C4 is u-w-x-v-u where w, x are common neighbors
    for u in range(n):
        for v in range(u+1, n):
            common = adj_int[u] & adj_int[v]
            if common & (common - 1): # >= 2 bits
                # Found C4. Pick two common neighbors w and x
                lsb = common & -common
                w = lsb.bit_length() - 1
                common_rest = common ^ lsb
                lsb2 = common_rest & -common_rest
                x = lsb2.bit_length() - 1
                
                # Vertices involved in C4: u, w, v, x
                # Edges in C4: (u,w), (w,v), (v,x), (x,u)
                # Remove the edge incident to the highest degree vertex to minimize damage
                candidates = [
                    (u, w, degrees[u]),
                    (w, v, degrees[w]),
                    (v, x, degrees[v]),
                    (x, u, degrees[x])
                ]
                candidates.sort(key=lambda t: t[2], reverse=True)
                a, b, _ = candidates[0]
                
                # Remove edge (a, b)
                adj_int[a] &= ~(1 << b)
                adj_int[b] &= ~(1 << a)
                degrees[a] -= 1
                degrees[b] -= 1
                return True
    return False

def _construct_c4_free(n, min_deg_req, time_limit):
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random graph
        p = random.uniform(0.5, 0.85)
        
        adj_int = [0] * n
        degrees = [0] * n
        
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj_int[u] |= (1 << v)
                    adj_int[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
        
        # Prune C4s
        changed = True
        while changed:
            if time.time() - start_time > time_limit:
                return None
            changed = _find_and_remove_c4(adj_int, degrees, n)
        
        # After pruning, check min degree
        min_deg = min(degrees)
        if min_deg >= min_deg_req:
            # Convert to edge list
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
            
        # If min_deg is close, try to add edges to low-degree vertices
        if min_deg >= min_deg_req - 2:
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                min_idx = min(range(n), key=lambda i: degrees[i])
                if degrees[min_idx] >= min_deg_req:
                    break
                for _ in range(20):
                    v = random.randint(0, n-1)
                    if v == min_idx or (adj_int[min_idx] >> v) & 1:
                        continue
                    if _creates_c4(adj_int, n, min_idx, v):
                        continue
                    adj_int[min_idx] |= (1 << v)
                    adj_int[v] |= (1 << min_idx)
                    degrees[min_idx] += 1
                    degrees[v] += 1
                    break
                    
            min_deg = min(degrees)
            if min_deg >= min_deg_req:
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
        
    return None

def construct():
    start_time = time.time()
    deadline = 115.0
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 1:
            break
            
        min_deg_req = max(1, n - 39)
        t_limit = min(30.0, remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
