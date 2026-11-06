import time
import random
import numpy as np

def _creates_c4_bitset(adj_int, n, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    adj_int: list of integers representing adjacency.
    """
    nu = adj_int[u]
    nv = adj_int[v]
    
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

def _construct_c4_free_bitset(n, min_deg_required, time_limit):
    start_time = time.time()
    
    max_attempts = 100
    best_result = None
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random edges, ensuring no C4
        # Start with empty graph
        adj_int = [0] * n
        degrees = [0] * n
        
        # Greedy fill
        # Randomize vertex order
        vertices = list(range(n))
        random.shuffle(vertices)
        
        for step in range(10000):
            if time.time() - start_time > time_limit:
                return best_result if best_result else None
                
            # Check success
            if min(degrees) >= min_deg_required:
                # Convert to edge list
                edges = []
                for u in range(n):
                    v_list = []
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
            candidates = [v for v in range(n) if v != min_deg_vertex and not (adj_int[min_deg_vertex] & (1 << v))]
            random.shuffle(candidates)
            
            for v in candidates:
                if not _creates_c4_bitset(adj_int, n, min_deg_vertex, v):
                    # Add edge
                    adj_int[min_deg_vertex] |= (1 << v)
                    adj_int[v] |= (1 << min_deg_vertex)
                    degrees[min_deg_vertex] += 1
                    degrees[v] += 1
                    found = True
                    break
                    
            if not found:
                # Stuck. Repair.
                # Remove a random edge
                if not any(degrees):
                    break
                # Pick a random existing edge
                edge_found = False
                for _ in range(200):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u < v and (adj_int[u] & (1 << v)):
                        # Remove
                        adj_int[u] ^= (1 << v)
                        adj_int[v] ^= (1 << u)
                        degrees[u] -= 1
                        degrees[v] -= 1
                        edge_found = True
                        break
                if not edge_found:
                    break

    return best_result

def _construct_c4_free_numpy(n, min_deg_required, time_limit):
    """
    Alternative construction using numpy for potentially better performance.
    """
    start_time = time.time()
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Use numpy boolean matrix for adjacency
        adj = np.zeros((n, n), dtype=bool)
        degrees = np.zeros(n, dtype=int)
        
        # Greedy fill
        for step in range(10000):
            if time.time() - start_time > time_limit:
                break
                
            # Check success
            if degrees.min() >= min_deg_required:
                # Convert to edge list
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u, v]:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
            # Find min degree vertex
            min_deg_vertex = int(np.argmin(degrees))
            
            # Find valid neighbors
            candidates = np.where(~adj[min_deg_vertex])[0]
            candidates = candidates[candidates != min_deg_vertex]
            
            if len(candidates) == 0:
                break
                
            # Shuffle candidates
            random.shuffle(candidates.tolist())
            candidates = np.array(candidates)
            
            found = False
            for v in candidates:
                v = int(v)
                # Check if adding edge creates C4
                # N(u) and N(v)
                nu = np.where(adj[min_deg_vertex])[0]
                nv = np.where(adj[v])[0]
                
                # Check if there's a C4
                # This is slow, use the bitset method instead
                # Convert to bitset for this check
                adj_int = [0] * n
                for i in range(n):
                    mask = 0
                    for j in np.where(adj[i])[0]:
                        mask |= (1 << j)
                    adj_int[i] = mask
                    
                if not _creates_c4_bitset(adj_int, n, min_deg_vertex, v):
                    adj[min_deg_vertex, v] = True
                    adj[v, min_deg_vertex] = True
                    degrees[min_deg_vertex] += 1
                    degrees[v] += 1
                    found = True
                    break
            
            if not found:
                break

    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Try n=47, 46, 45...
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate time based on n
        # Higher n gets less time per attempt since harder
        t_limit = min(15.0, remaining - 2)
        
        # First try bitset approach
        result = _construct_c4_free_bitset(n, min_deg_req, t_limit)
        if result:
            return result
            
        # If that failed, try numpy approach with remaining time
        if time.time() - start_time < deadline - 5:
            t_limit2 = deadline - (time.time() - start_time) - 3
            if t_limit2 > 5:
                result = _construct_c4_free_numpy(n, min_deg_req, t_limit2)
                if result:
                    return result

    # Fallback: return a valid small graph
    # n=39, min_deg 0 (empty graph)
    return {'n': 39, 'edges': []}
