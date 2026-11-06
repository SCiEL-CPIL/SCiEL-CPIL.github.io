import time
import random

def _creates_c4(adj_int, n, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    A C4 is created iff |N(u) ∩ N(v)| >= 2 (two common neighbors form the C4 with u,v).
    """
    common = adj_int[u] & adj_int[v]
    return common & (common - 1) != 0

def _find_and_remove_c4(adj_int, n, degrees):
    """
    Find a C4 in the graph and remove an edge to break it.
    Returns True if a C4 was found and an edge removed.
    """
    for u in range(n):
        nu = adj_int[u]
        for v in range(u + 1, n):
            common = nu & adj_int[v]
            if common & (common - 1):  # >= 2 common neighbors
                # Get two common neighbors
                lsb1 = common & -common
                w = lsb1.bit_length() - 1
                rest = common ^ lsb1
                lsb2 = rest & -rest
                x = lsb2.bit_length() - 1
                
                # C4: u-w-v-x-u (edges: u-w, w-v, v-x, x-u)
                # Remove edge incident to highest-degree vertex
                candidates = [
                    (u, w, degrees[u]),
                    (w, v, degrees[w]),
                    (v, x, degrees[v]),
                    (x, u, degrees[x])
                ]
                candidates.sort(key=lambda t: t[2], reverse=True)
                a, b, _ = candidates[0]
                
                adj_int[a] &= ~(1 << b)
                adj_int[b] &= ~(1 << a)
                degrees[a] -= 1
                degrees[b] -= 1
                return True
    return False

def _polarity_seed(n):
    """
    Create a polarity graph from projective plane of order q as seed.
    For n <= 47, we can use q=5 (projective plane has 31 points) or q=7 (57 points, too many).
    We'll use a simpler approach: create a graph based on quadratic residues.
    """
    adj_int = [0] * n
    degrees = [0] * n
    
    # Use a C4-free construction: for each vertex i, connect to i+1, i+2, ..., i+k
    # where k is chosen to maintain C4-freeness
    # Actually, let's use a known C4-free graph: the incidence graph of a projective plane
    # For now, use a simple construction: connect i to i+d mod n for d in a set with no repeated differences
    
    # Simpler: use a random seed with higher density
    p = random.uniform(0.6, 0.8)
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                adj_int[u] |= (1 << v)
                adj_int[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
    
    return adj_int, degrees

def _construct_c4_free(n, min_deg_req, time_limit):
    start_time = time.time()
    
    # Try multiple random initializations with varying densities
    densities = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    random.shuffle(densities)
    
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
            
        # Use varying densities
        p = densities[attempt % len(densities)]
        p = p + random.uniform(-0.05, 0.05)
        p = max(0.4, min(0.9, p))
        
        adj_int = [0] * n
        degrees = [0] * n
        
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj_int[u] |= (1 << v)
                    adj_int[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
        
        # Prune C4s
        changed = True
        prune_count = 0
        while changed:
            if time.time() - start_time > time_limit:
                return None
            if prune_count > 5000:  # Safety limit
                break
            changed = _find_and_remove_c4(adj_int, n, degrees)
            prune_count += 1
        
        # After pruning, check min degree
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
        
        # If min_deg is close, try to add edges to low-degree vertices
        if min_deg >= min_deg_req - 3:
            boost_rounds = 100
            for _ in range(boost_rounds):
                if time.time() - start_time > time_limit:
                    break
                min_idx = min(range(n), key=lambda i: degrees[i])
                if degrees[min_idx] >= min_deg_req:
                    break
                
                # Try to add an edge to min_idx
                added = False
                for _ in range(30):
                    v = random.randint(0, n - 1)
                    if v == min_idx or (adj_int[min_idx] >> v) & 1:
                        continue
                    if _creates_c4(adj_int, n, min_idx, v):
                        continue
                    adj_int[min_idx] |= (1 << v)
                    adj_int[v] |= (1 << min_idx)
                    degrees[min_idx] += 1
                    degrees[v] += 1
                    added = True
                    break
                
                if not added:
                    # Try all vertices systematically
                    for v in range(n):
                        if v == min_idx or (adj_int[min_idx] >> v) & 1:
                            continue
                        if not _creates_c4(adj_int, n, min_idx, v):
                            adj_int[min_idx] |= (1 << v)
                            adj_int[v] |= (1 << min_idx)
                            degrees[min_idx] += 1
                            degrees[v] += 1
                            added = True
                            break
                
                if not added:
                    # Can't boost this vertex, try another approach
                    # Remove an edge from a high-degree vertex and try to reconnect
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
        t_limit = min(35.0, remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
