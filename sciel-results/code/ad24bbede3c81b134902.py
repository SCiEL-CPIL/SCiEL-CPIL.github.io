import time
import random
import math

def _creates_c4(adj_int, n, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    adj_int: list of integers representing adjacency.
    u, v: vertices to connect. Assumes not already connected.
    """
    nu = adj_int[u]
    nv = adj_int[v]
    
    # Check common neighbors: if |N(u) ∩ N(v)| >= 2, adding (u,v) creates C4
    common = nu & nv
    if common & (common - 1):  # More than 1 bit set
        return True
        
    # Check edges between N(u) and N(v)
    # A C4 exists if there's x in N(u), y in N(v) with x != y and x~y
    # Equivalently: union of N(x) for x in N(u) intersects N(v), excluding u itself
    deg_u = bin(nu).count('1')
    deg_v = bin(nv).count('1')
    
    if deg_u <= deg_v:
        union = 0
        temp = nu
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            union |= adj_int[x]
            temp ^= lsb
        # Check if any neighbor of u is connected to a neighbor of v
        # We need x~y where x in N(u), y in N(v), x != y
        # union contains all vertices connected to any x in N(u)
        # We need intersection with N(v), but exclude u (which is in N(x) for all x in N(u))
        if union & nv:
            return True
    else:
        union = 0
        temp = nv
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            union |= adj_int[y]
            temp ^= lsb
        if union & nu:
            return True
            
    return False

def _find_and_remove_c4(adj_int, degrees, n):
    """
    Find a C4 in the graph and remove an edge to break it.
    Returns True if a C4 was found and an edge removed, False otherwise.
    """
    for u in range(n):
        for v in range(u+1, n):
            common = adj_int[u] & adj_int[v]
            if common & (common - 1):
                lsb = common & -common
                w = lsb.bit_length() - 1
                common_rest = common ^ lsb
                lsb2 = common_rest & -common_rest
                x = lsb2.bit_length() - 1
                
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

def _spectral_weights(adj_int, degrees, n, iterations=5):
    """
    Compute spectral weights using power iteration on adjacency matrix.
    Returns list of weights for each vertex.
    """
    # Initialize weights
    weights = [1.0] * n
    
    for _ in range(iterations):
        new_weights = [0.0] * n
        for u in range(n):
            temp = adj_int[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                new_weights[u] += weights[v]
                temp ^= lsb
        # Normalize
        max_w = max(new_weights) if max(new_weights) > 0 else 1.0
        for i in range(n):
            new_weights[i] /= max_w
        weights = new_weights
    
    return weights

def _construct_c4_free(n, min_deg_req, time_limit):
    start_time = time.time()
    
    # Strategy 1: Greedy construction with spectral guidance
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
            
        adj_int = [0] * n
        degrees = [0] * n
        
        # Randomize vertex order for diversity
        vertices = list(range(n))
        random.shuffle(vertices)
        
        # Greedy fill with spectral weighting
        max_steps = 2000
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return None
                
            # Check success
            if min(degrees) >= min_deg_req:
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
                
            # Compute spectral weights periodically
            if step % 50 == 0:
                weights = _spectral_weights(adj_int, degrees, n, iterations=3)
            else:
                # Use degree-based weights as fallback
                max_deg = max(degrees) if max(degrees) > 0 else 1
                weights = [1.0 / (degrees[i] + 1) for i in range(n)]
                # Scale by inverse degree to prioritize low-degree vertices
                weights = [w * (max_deg + 1) for w in weights]
            
            # Find the most "urgent" vertex (lowest degree, weighted by spectral importance)
            # Priority = 1/degree * spectral_weight
            priorities = []
            for i in range(n):
                if degrees[i] < min_deg_req:
                    pri = weights[i] / (degrees[i] + 1)
                    priorities.append((pri, i))
            
            if not priorities:
                break
                
            priorities.sort(reverse=True)
            min_idx = priorities[0][1]
            
            # Find valid neighbor to connect
            found = False
            candidates = [v for v in range(n) if v != min_idx and not (adj_int[min_idx] >> v) & 1]
            random.shuffle(candidates)
            
            # Sort candidates by spectral weight to prefer connections to well-connected vertices
            # but also consider degree balance
            candidate_scores = []
            for v in candidates:
                # Prefer vertices with moderate degree (not too high to avoid C4, not too low)
                score = weights[v] * (1.0 / (abs(degrees[v] - n//2) + 1))
                candidate_scores.append((score, v))
            
            candidate_scores.sort(reverse=True)
            
            for _, v in candidate_scores:
                if not _creates_c4(adj_int, n, min_idx, v):
                    adj_int[min_idx] |= (1 << v)
                    adj_int[v] |= (1 << min_idx)
                    degrees[min_idx] += 1
                    degrees[v] += 1
                    found = True
                    break
                    
            if not found:
                # Stuck - try random edge removal to unstick
                if any(degrees):
                    for _ in range(50):
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u < v and (adj_int[u] >> v) & 1:
                            adj_int[u] &= ~(1 << v)
                            adj_int[v] &= ~(1 << u)
                            degrees[u] -= 1
                            degrees[v] -= 1
                            break
    
    # Strategy 2: Random dense + prune (fallback)
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            p = random.uniform(0.6, 0.8)
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
            prune_count = 0
            while changed and prune_count < 500:
                if time.time() - start_time > time_limit:
                    break
                changed = _find_and_remove_c4(adj_int, degrees, n)
                prune_count += 1
            
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
            
            # Try to repair if close
            if min_deg >= min_deg_req - 1:
                for _ in range(30):
                    if time.time() - start_time > time_limit:
                        break
                    min_idx = min(range(n), key=lambda i: degrees[i])
                    if degrees[min_idx] >= min_deg_req:
                        break
                    for _ in range(15):
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
        t_limit = min(25.0, remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
