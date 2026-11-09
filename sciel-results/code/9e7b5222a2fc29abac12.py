"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _score_graph_fast(n, adj, deg):
    """Compute the maximum degree sum over all triangles using adjacency matrix."""
    max_sum = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
    return max_sum

def _build_adj_deg(n, edges):
    """Build adjacency matrix and degree list from edge list."""
    adj = [[0] * n for _ in range(n)]
    deg = [0] * n
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _simulated_annealing(n, initial_edges, deadline, seed=42):
    """Run simulated annealing on edge swaps to minimize max triangle degree sum."""
    rng = random.Random(seed)
    target_edges = 65
    
    adj, deg = _build_adj_deg(n, initial_edges)
    current_score = _score_graph_fast(n, adj, deg)
    
    best_edges = list(initial_edges)
    best_score = current_score
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    all_edges_set = set(all_edges)
    current_edge_set = set((min(u,v), max(u,v)) for u, v in initial_edges)
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    n_iter_total = 50000
    i = 0
    
    while time.time() < deadline:
        i += 1
        # Exponential cooling
        T = T_start * (T_end / T_start) ** (i / n_iter_total)
        
        # Pick a random edge to remove
        e_remove = rng.choice(initial_edges)
        u1, v1 = e_remove
        e_remove_key = (min(u1, v1), max(u1, v1))
        
        # Pick a random non-edge to add
        e_add = rng.choice(all_edges)
        e_add_key = (min(e_add[0], e_add[1]), max(e_add[0], e_add[1]))
        if e_add_key in current_edge_set:
            continue
        
        u2, v2 = e_add
        
        # Perform the swap
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        deg[u1] -= 1
        deg[v1] -= 1
        deg[u2] += 1
        deg[v2] += 1
        
        current_edge_set.remove(e_remove_key)
        current_edge_set.add(e_add_key)
        initial_edges.remove(e_remove)
        initial_edges.append(e_add)
        
        # Compute new score
        new_score = _score_graph_fast(n, adj, deg)
        
        delta = new_score - current_score
        
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = list(initial_edges)
        else:
            # Revert
            adj[u1][v1] = 1
            adj[v1][u1] = 1
            adj[u2][v2] = 0
            adj[v2][u2] = 0
            deg[u1] += 1
            deg[v1] += 1
            deg[u2] -= 1
            deg[v2] -= 1
            
            current_edge_set.remove(e_add_key)
            current_edge_set.add(e_remove_key)
            initial_edges.remove(e_add)
            initial_edges.append(e_remove)
    
    return best_edges, best_score

def _generate_initial_configs(n, target_edges, rng):
    """Generate diverse initial configurations."""
    configs = []
    
    # Config 1: K_{8,8} + one edge inside part A
    edges1 = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    configs.append(edges1)
    
    # Config 2: K_{8,8} + one edge inside part B
    edges2 = [[u, v] for u in range(8) for v in range(8, 16)] + [[8, 9]]
    configs.append(edges2)
    
    # Config 3: Random near-balanced bipartite + random edges
    for trial in range(5):
        edges = []
        # Start with a random bipartition
        left = list(range(8))
        right = list(range(8, 16))
        rng.shuffle(left)
        rng.shuffle(right)
        
        # Add most cross edges
        for u in left:
            for v in right:
                if rng.random() < 0.9:
                    edges.append([u, v])
        
        # Add some within-part edges to reach target
        while len(edges) < target_edges:
            if rng.random() < 0.5:
                u, v = rng.sample(left, 2)
            else:
                u, v = rng.sample(right, 2)
            if [u, v] not in edges and [v, u] not in edges:
                edges.append([u, v])
        
        # Trim to exact count
        if len(edges) > target_edges:
            edges = edges[:target_edges]
        
        if len(edges) == target_edges:
            configs.append(edges)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 16
    target_edges = 65
    
    rng = random.Random(42)
    
    # Generate multiple initial configurations
    initial_configs = _generate_initial_configs(n, target_edges, rng)
    
    best_edges = None
    best_score = float('inf')
    
    # Run simulated annealing from each initial config
    for idx, init_edges in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Allocate remaining time among remaining configs
        remaining_configs = len(initial_configs) - idx
        per_config_time = (deadline - time.time()) / max(remaining_configs, 1)
        config_deadline = time.time() + per_config_time
        
        edges, score = _simulated_annealing(n, list(init_edges), config_deadline, seed=42 + idx)
        
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Final polish: hill climbing from best solution
    if best_edges is not None and time.time() < deadline - 5:
        adj, deg = _build_adj_deg(n, best_edges)
        current_score = _score_graph_fast(n, adj, deg)
        
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        current_edge_set = set((min(u,v), max(u,v)) for u, v in best_edges)
        
        # Hill climbing: only accept improving moves
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for e_remove in best_edges:
                if time.time() > deadline - 2:
                    break
                u1, v1 = e_remove
                e_remove_key = (min(u1, v1), max(u1, v1))
                
                for e_add in all_edges:
                    e_add_key = (min(e_add[0], e_add[1]), max(e_add[0], e_add[1]))
                    if e_add_key in current_edge_set:
                        continue
                    
                    u2, v2 = e_add
                    
                    # Try the swap
                    adj[u1][v1] = 0
                    adj[v1][u1] = 0
                    adj[u2][v2] = 1
                    adj[v2][u2] = 1
                    deg[u1] -= 1
                    deg[v1] -= 1
                    deg[u2] += 1
                    deg[v2] += 1
                    
                    new_score = _score_graph_fast(n, adj, deg)
                    
                    if new_score < current_score:
                        # Accept
                        current_edge_set.remove(e_remove_key)
                        current_edge_set.add(e_add_key)
                        best_edges.remove(e_remove)
                        best_edges.append(e_add)
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                        improved = True
                    else:
                        # Revert
                        adj[u1][v1] = 1
                        adj[v1][u1] = 1
                        adj[u2][v2] = 0
                        adj[v2][u2] = 0
                        deg[u1] += 1
                        deg[v1] += 1
                        deg[u2] -= 1
                        deg[v2] -= 1
    
    if best_edges is None:
        # Fallback to initial construction
        best_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
