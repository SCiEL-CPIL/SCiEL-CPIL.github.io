# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin
    
    # Try from largest n down
    # We know 46 is the target. 47 is possible but harder.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() - start_time > deadline:
            break
        
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Try polarity-based construction for n >= 40
        # q=7 gives 57 vertices.
        if n <= 57:
            result = _try_polarity_subgraph(n, min_deg_req, time.time())
            if result:
                return result
        
        # Fallback to random greedy
        result = _try_random_greedy(n, min_deg_req, time.time())
        if result:
            return result

    return {'n': 39, 'edges': []}

def _try_polarity_subgraph(n_target, min_deg_req, start_time):
    """
    Construct C4-free graph on n_target vertices with min degree >= min_deg_req
    using a subgraph of the q=7 Polarity Graph (57 vertices, degree 8).
    """
    q = 7
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    N_total = len(points) # 57
    
    def is_orth(p1, p2):
        x1, x2, x3 = p1
        y1, y2, y3 = p2
        val = (x1*y2 + x1*y3 + x2*y1 + x2*y3 + x3*y1 + x3*y2) % q
        return val == 0

    # Build full adjacency
    adj_full = [set() for _ in range(N_total)]
    for i in range(N_total):
        for j in range(i+1, N_total):
            if is_orth(points[i], points[j]):
                adj_full[i].add(j)
                adj_full[j].add(i)
    
    # Try multiple random subsets of vertices to keep
    for attempt in range(20):
        if time.time() - start_time > 140:
            return None
            
        removed_count = N_total - n_target
        if removed_count < 0:
            return None
            
        kept_indices = list(range(N_total))
        random.shuffle(kept_indices)
        kept = set(kept_indices[:n_target])
        
        # Map indices
        sorted_kept = sorted(kept)
        old_to_new = {old: new for new, old in enumerate(sorted_kept)}
        new_to_old = {new: old for new, old in enumerate(sorted_kept)}
        
        adj = [set() for _ in range(n_target)]
        for u_new in range(n_target):
            u_old = new_to_old[u_new]
            for v_old in adj_full[u_old]:
                if v_old in kept:
                    v_new = old_to_new[v_old]
                    if u_new < v_new:
                        adj[u_new].add(v_new)
                        adj[v_new].add(u_new)
        
        # Greedy edge addition
        # We want to increase min degree to min_deg_req
        # Current max degree is 8.
        # If min_deg_req > 8, impossible with this base.
        if min_deg_req > 8:
            continue
            
        for _ in range(2000):
            if time.time() - start_time > 140:
                return None
                
            min_deg = min(len(adj[i]) for i in range(n_target))
            if min_deg >= min_deg_req:
                edges = []
                for u in range(n_target):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n_target, 'edges': edges}
                
            # Find vertex with min degree
            u = min(range(n_target), key=lambda i: len(adj[i]))
            
            candidates = []
            for v in range(n_target):
                if v == u or v in adj[u]:
                    continue
                common = adj[u] & adj[v]
                if len(common) <= 1:
                    candidates.append(v)
            
            if not candidates:
                break
            
            v = random.choice(candidates)
            adj[u].add(v)
            adj[v].add(u)
            
    return None

def _try_random_greedy(n, min_deg_req, start_time):
    for _ in range(100):
        if time.time() - start_time > 140:
            return None
        adj = [set() for _ in range(n)]
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            common = adj[u] & adj[v]
            if len(common) <= 1:
                adj[u].add(v)
                adj[v].add(u)
        
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= min_deg_req:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    return None
# EVOLVE-BLOCK-END
