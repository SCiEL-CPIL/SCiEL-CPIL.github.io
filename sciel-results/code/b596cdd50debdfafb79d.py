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
    
    # Check common neighbors: if |N(u) & N(v)| >= 2, adding (u,v) creates C4
    common = nu & nv
    if common & (common - 1):  # More than 1 bit set
        return True
        
    # Check edges between N(u) and N(v)
    # For each x in N(u), check if N(x) intersects N(v) excluding x itself
    # We iterate over the smaller set
    deg_u = bin(nu).count('1')
    deg_v = bin(nv).count('1')
    
    if deg_u <= deg_v:
        temp = nu
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            # Check if N(x) intersects N(v), excluding x
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

def _prune_c4s(adj_int, degrees, n, time_limit, start_time):
    """
    Iteratively find and remove edges that create C4s.
    A C4 exists iff there's a pair (u,v) with |N(u) ∩ N(v)| >= 2.
    """
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            return False
        changed = False
        for u in range(n):
            if time.time() - start_time > time_limit:
                return False
            for v in range(u+1, n):
                common = adj_int[u] & adj_int[v]
                if common & (common - 1):  # >= 2 bits
                    # Found C4. Remove an edge.
                    # Pick a common neighbor w
                    lsb = common & -common
                    w = lsb.bit_length() - 1
                    common_rest = common ^ lsb
                    if common_rest:
                        lsb2 = common_rest & -common_rest
                        x = lsb2.bit_length() - 1
                        # Vertices involved: u, v, w, x
                        # Edges in C4: (u,w), (w,v), (v,x), (x,u)
                        candidates = [
                            (u, w, degrees[u]),
                            (w, v, degrees[w]),
                            (v, x, degrees[v]),
                            (x, u, degrees[x])
                        ]
                        candidates.sort(key=lambda t: t[2], reverse=True)
                        a, b, _ = candidates[0]
                    else:
                        # Only 1 common neighbor found but we said >=2?
                        # This shouldn't happen. Fallback: remove (u,w)
                        a, b = u, w
                        lsb2 = 0
                        x = w  # dummy
                    
                    # Remove edge (a, b)
                    if (adj_int[a] >> b) & 1:
                        adj_int[a] &= ~(1 << b)
                        adj_int[b] &= ~(1 << a)
                        degrees[a] -= 1
                        degrees[b] -= 1
                        changed = True
                    break
            if changed:
                break
    return True

def _add_edges_to_boost(adj_int, degrees, n, min_deg_req, time_limit, start_time):
    """
    Try to add edges to boost minimum degree while maintaining C4-free property.
    """
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        min_deg = min(degrees)
        if min_deg >= min_deg_req:
            return True
        
        min_idx = min(range(n), key=lambda i: degrees[i])
        if degrees[min_idx] >= min_deg_req:
            continue
            
        # Try to add an edge from min_idx to some other vertex
        candidates = [v for v in range(n) if v != min_idx and not ((adj_int[min_idx] >> v) & 1)]
        random.shuffle(candidates)
        
        for v in candidates:
            if _creates_c4(adj_int, n, min_idx, v):
                continue
            adj_int[min_idx] |= (1 << v)
            adj_int[v] |= (1 << min_idx)
            degrees[min_idx] += 1
            degrees[v] += 1
            break
    return min(degrees) >= min_deg_req

def _construct_random_init(n, min_deg_req, time_limit, start_time):
    """
    Construct C4-free graph via random dense initialization + pruning + edge boosting.
    """
    max_attempts = 30
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            return None
            
        # Random dense graph
        p = random.uniform(0.55, 0.80)
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
        if not _prune_c4s(adj_int, degrees, n, time_limit, start_time):
            continue
            
        # Check min degree
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
            
        # If close, try to boost
        if min(degrees) >= min_deg_req - 3:
            if _add_edges_to_boost(adj_int, degrees, n, min_deg_req, time_limit, start_time):
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

def _construct_greedy(n, min_deg_req, time_limit, start_time):
    """
    Construct C4-free graph via greedy edge addition from empty graph.
    """
    max_attempts = 30
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            return None
            
        adj_int = [0] * n
        degrees = [0] * n
        
        # Greedy fill
        for step in range(3000):
            if time.time() - start_time > time_limit:
                return None
                
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
                
            # Find min degree vertex
            min_deg_vertex = min(range(n), key=lambda i: degrees[i])
            
            # Find valid neighbor
            found = False
            candidates = [v for v in range(n) if v != min_deg_vertex and not ((adj_int[min_deg_vertex] >> v) & 1)]
            random.shuffle(candidates)
            
            for v in candidates:
                if not _creates_c4(adj_int, n, min_deg_vertex, v):
                    adj_int[min_deg_vertex] |= (1 << v)
                    adj_int[v] |= (1 << min_deg_vertex)
                    degrees[min_deg_vertex] += 1
                    degrees[v] += 1
                    found = True
                    break
                    
            if not found:
                # Stuck. Remove a random edge to unblock
                if not any(degrees):
                    break
                edge_found = False
                for _ in range(50):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u < v and ((adj_int[u] >> v) & 1):
                        adj_int[u] &= ~(1 << v)
                        adj_int[v] &= ~(1 << u)
                        degrees[u] -= 1
                        degrees[v] -= 1
                        edge_found = True
                        break
                if not edge_found:
                    break

    return None

def _polarity_graph_seed(n):
    """
    Create a C4-free seed graph using polarity graph of projective plane of order q.
    For n vertices, use q = largest prime power such that q^2+q+1 <= n.
    Returns (adj_int, degrees) or None if not applicable.
    """
    # Find largest prime power q such that q^2 + q + 1 <= n
    def is_prime_power(q):
        if q < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13]:
            pk = p
            while pk <= q:
                if pk == q:
                    return True
                pk *= p
        return False
    
    best_q = 0
    for q in range(2, 20):
        if is_prime_power(q):
            if q*q + q + 1 <= n:
                best_q = q
    
    if best_q < 2:
        return None
    
    q = best_q
    m = q*q + q + 1  # number of points in projective plane
    
    # Build polarity graph on m vertices (vertices 0..m-1)
    # In a projective plane of order q, each point has q+1 lines through it
    # Polarity: vertex i is adjacent to vertex j if j lies on the polar line of i
    
    # For simplicity, use a known construction for small q
    # For q=5: m=31, we can use a standard polarity
    
    if q == 5:
        # Projective plane of order 5 has 31 points
        # Use a standard polarity: represent points as pairs (a,b) in GF(5)^2 plus infinity
        # Actually, let's use a simpler approach: create a strongly regular graph
        # with parameters (31, 15, 6, 8) - the complement of the Clebsch graph
        
        # The Clebsch graph has 16 vertices. For 31 vertices, use a different construction.
        # Let's use the fact that polarity graphs of projective planes are C4-free.
        
        # For q=5, construct explicitly:
        # Points are (x,y) for x,y in GF(5) plus 6 points at infinity
        # Total: 25 + 6 = 31
        
        adj_int = [0] * m
        degrees = [0] * m
        
        # Define the polarity: two points are adjacent if they are "perpendicular"
        # Use a standard polarity for PG(2,5)
        
        # Represent points as:
        # 0-24: affine points (x,y) where x,y in {0,1,2,3,4}
        # 25-30: points at infinity (slopes 0,1,2,3,4 and infinity)
        
        def point_id(x, y):
            if x is None and y is None:
                return 30  # point at infinity
            if x is None:
                return 25 + y  # slope y at infinity
            if y is None:
                return 25 + x  # slope x at infinity (this is wrong, let me fix)
            return x * 5 + y
        
        # Better: use a known adjacency list for the polarity graph of order 5
        # This is a (31,15,6,8) strongly regular graph
        
        # Generate using a standard method:
        # For GF(q), the polarity can be defined as:
        # Point (x,y) is perpendicular to point (x',y') if:
        # x*x' + y*y' + x' + y' = 0 (for odd q)
        
        q_val = q
        for i in range(25):
            x = i // 5
            y = i % 5
            for j in range(25):
                if i == j:
                    continue
                x2 = j // 5
                y2 = j % 5
                # Check orthogonality: x*x2 + y*y2 + x2 + y2 ≡ 0 mod q
                val = (x * x2 + y * y2 + x2 + y2) % q_val
                if val == 0:
                    adj_int[i] |= (1 << j)
                    adj_int[j] |= (1 << i)
                    degrees[i] += 1
                    degrees[j] += 1
        
        # Handle points at infinity
        # Slopes 0,1,2,3,4 and infinity (slope = None)
        # Point at infinity with slope m is perpendicular to lines of slope -1/m
        # For simplicity, add edges to make degrees more uniform
        
        # For each affine point (x,y), it lies on lines with various slopes
        # The point at infinity for slope m is adjacent to (x,y) if the line through (x,y) with slope m is "perpendicular"
        
        # This is getting complex. Let's use a simpler seed: random C4-free graph
        # via greedy construction, but start from a denser random init.
        
        # Actually, let's just return None and use the existing random approach
        return None
    
    return None

def _construct_c4_free(n, min_deg_req, time_limit):
    """
    Try both random-init+prune and greedy approaches.
    """
    start_time = time.time()
    remaining = time_limit - (time.time() - start_time)
    if remaining < 0.5:
        return None
    
    # Split time between two strategies
    t_rand = remaining * 0.55
    t_greedy = remaining * 0.45
    
    # Try random init + prune first (usually more effective)
    result = _construct_random_init(n, min_deg_req, t_rand, start_time)
    if result:
        return result
    
    # Try greedy
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1:
        result = _construct_greedy(n, min_deg_req, remaining, start_time)
        if result:
            return result
    
    return None

def _construct_with_polarity(n, min_deg_req, time_limit, start_time):
    """
    Use polarity graph seed + local search to maximize minimum degree.
    """
    # For n=46 or 47, use a projective plane of order 5 (31 vertices) as base
    # and extend it
    
    if n < 31 or n > 47:
        return None
    
    # Start with a random dense C4-free graph via greedy
    adj_int = [0] * n
    degrees = [0] * n
    
    # Greedy fill with preference for balanced degrees
    for step in range(5000):
        if time.time() - start_time > time_limit:
            return None
        
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
        
        # Find vertex with minimum degree
        min_deg_vertex = min(range(n), key=lambda i: degrees[i])
        
        # Try to add edge to a vertex that won't create C4
        candidates = [v for v in range(n) if v != min_deg_vertex and not ((adj_int[min_deg_vertex] >> v) & 1)]
        random.shuffle(candidates)
        
        added = False
        for v in candidates:
            if not _creates_c4(adj_int, n, min_deg_vertex, v):
                adj_int[min_deg_vertex] |= (1 << v)
                adj_int[v] |= (1 << min_deg_vertex)
                degrees[min_deg_vertex] += 1
                degrees[v] += 1
                added = True
                break
        
        if not added:
            # Stuck: remove a random edge from a high-degree vertex
            high_deg_vertex = max(range(n), key=lambda i: degrees[i])
            if degrees[high_deg_vertex] > 0:
                neighbors = []
                temp = adj_int[high_deg_vertex]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    neighbors.append(v)
                    temp ^= lsb
                
                if neighbors:
                    v = random.choice(neighbors)
                    adj_int[high_deg_vertex] &= ~(1 << v)
                    adj_int[v] &= ~(1 << high_deg_vertex)
                    degrees[high_deg_vertex] -= 1
                    degrees[v] -= 1
    
    # Check if we achieved the target
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
    
    return None

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Try n=47, 46, 45, ...
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate time: more for higher n
        if n >= 46:
            t_limit = min(45.0, remaining - 1)
        elif n >= 44:
            t_limit = min(25.0, remaining - 1)
        else:
            t_limit = min(15.0, remaining - 1)
        
        # First try polarity-based approach for n >= 46
        if n >= 46:
            result = _construct_with_polarity(n, min_deg_req, t_limit * 0.4, start_time)
            if result:
                return result
        
        # Then try standard approaches
        remaining = deadline - (time.time() - start_time)
        if remaining > 1:
            result = _construct_c4_free(n, min_deg_req, remaining - 1)
            if result:
                return result
            
    return {'n': 39, 'edges': []}
