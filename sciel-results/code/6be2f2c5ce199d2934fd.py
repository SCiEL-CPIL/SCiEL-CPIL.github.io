import time
import random
import math

def _build_adjacency(n, edges):
    adj = [[0] * n for _ in range(n)]
    deg = [0] * n
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _compute_scores_from_triangles(n, deg, triangles):
    """
    Computes max triangle degree sum and sum of squared triangle degree sums from a list of triangles.
    """
    max_sum = 0
    sum_sq = 0
    for u, v, w in triangles:
        s = deg[u] + deg[v] + deg[w]
        if s > max_sum:
            max_sum = s
        sum_sq += s * s
    return max_sum, sum_sq

def construct():
    n = 16
    m = 65
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initial construction: K_{8,8} + 1 edge
    # Vertices 0-7 and 8-15
    initial_edges = set()
    for u in range(8):
        for v in range(8, 16):
            initial_edges.add((u, v))
    # Add one edge within part A to make 65 edges and create triangles
    initial_edges.add((0, 1))
    
    # Convert to list for easy manipulation, keep set for O(1) lookup
    cur_edges = set(initial_edges)
    adj, deg = _build_adjacency(n, list(cur_edges))
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Initialize active triangles
    active_triangles = []
    # Only triangles involving (0,1) are possible in initial state
    # Since only (0,1) is an intra-part edge.
    for w in range(8, 16):
        if adj[0][w] and adj[1][w]:
            # Order u < v < w
            # 0 < 1 < w is always true for w in 8..15
            active_triangles.append((0, 1, w))
            
    cur_max, cur_sq = _compute_scores_from_triangles(n, deg, active_triangles)
    best_max = cur_max
    best_sq = cur_sq
    best_edges = set(cur_edges)
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Calculate temperature based on time
        elapsed = time.time() - start_time
        total_time = deadline - start_time
        progress = elapsed / total_time
        T = T_start * (T_end / T_start) ** progress
        
        # Pick edge to remove
        e_out = rng.choice(list(cur_edges))
        u1, v1 = e_out
        
        # Pick edge to add
        # Retry a few times if we pick an existing edge
        e_in = None
        for _ in range(10):
            candidate = rng.choice(all_edges)
            if candidate not in cur_edges:
                e_in = candidate
                break
        
        if e_in is None:
            continue
            
        u2, v2 = e_in
        
        # Perform swap
        cur_edges.discard(e_out)
        cur_edges.add(e_in)
        
        # Update adjacency and degrees
        # Remove e_out
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        deg[u1] -= 1
        deg[v1] -= 1
        
        # Add e_in
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        deg[u2] += 1
        deg[v2] += 1
        
        # Update active triangles incrementally
        # 1. Remove triangles involving e_out (u1, v1)
        for w in range(n):
            if w == u1 or w == v1:
                continue
            # Check if (u1, w) and (v1, w) are edges
            if adj[u1][w] and adj[v1][w]:
                # Triangle (u1, v1, w) was present and is now destroyed
                tri = (u1, v1, w)
                tri = tuple(sorted(tri))
                if tri in active_triangles:
                    active_triangles.remove(tri)
        
        # 2. Add triangles involving e_in (u2, v2)
        for w in range(n):
            if w == u2 or w == v2:
                continue
            # Check if (u2, w) and (v2, w) are edges
            if adj[u2][w] and adj[v2][w]:
                # Triangle (u2, v2, w) is formed
                tri = (u2, v2, w)
                tri = tuple(sorted(tri))
                active_triangles.append(tri)
        
        # Compute new scores
        new_max, new_sq = _compute_scores_from_triangles(n, deg, active_triangles)
        
        # Cost function
        old_cost = (cur_max ** 2) + 0.1 * cur_sq
        new_cost = (new_max ** 2) + 0.1 * new_sq
        
        delta = new_cost - old_cost
        
        # Acceptance criterion
        accepted = False
        if delta <= 0:
            accepted = True
        else:
            # Boltzmann distribution
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    accepted = True
        
        if accepted:
            cur_max = new_max
            cur_sq = new_sq
            # Update best
            if cur_max < best_max or (cur_max == best_max and cur_sq < best_sq):
                best_max = cur_max
                best_sq = cur_sq
                best_edges = set(cur_edges)
        else:
            # Revert
            cur_edges.discard(e_in)
            cur_edges.add(e_out)
            
            # Revert adjacency and degrees
            adj[u1][v1] = 1
            adj[v1][u1] = 1
            deg[u1] += 1
            deg[v1] += 1
            
            adj[u2][v2] = 0
            adj[v2][u2] = 0
            deg[u2] -= 1
            deg[v2] -= 1
            
            # Revert triangles
            # Remove triangles involving e_in (u2, v2) that were just added
            for w in range(n):
                if w == u2 or w == v2:
                    continue
                if adj[u2][w] and adj[v2][w]:
                    tri = (u2, v2, w)
                    tri = tuple(sorted(tri))
                    if tri in active_triangles:
                        active_triangles.remove(tri)
                        
            # Add back triangles involving e_out (u1, v1) that were just removed
            for w in range(n):
                if w == u1 or w == v1:
                    continue
                if adj[u1][w] and adj[v1][w]:
                    tri = (u1, v1, w)
                    tri = tuple(sorted(tri))
                    active_triangles.append(tri)
            
            # cur_max, cur_sq remain unchanged (pre-swap values)

    # Return best found
    edges_list = [[u, v] for u, v in best_edges if u < v]
    return {'n': 16, 'edges': edges_list}
