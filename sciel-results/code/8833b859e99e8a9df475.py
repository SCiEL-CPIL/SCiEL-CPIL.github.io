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

def _score_graph_sum_sq(n, adj, deg):
    """Compute sum of squared degree sums over all triangles (surrogate)."""
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        s = deg[u] + deg[v] + deg[w]
                        total += s * s
    return total

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

def _get_triangles(n, adj):
    """Get list of all triangles as tuples (u, v, w) with u < v < w."""
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        triangles.append((u, v, w))
    return triangles

def _simulated_annealing(n, initial_edges, deadline, seed=42, use_surrogate=True):
    """Run simulated annealing on edge swaps to minimize max triangle degree sum."""
    rng = random.Random(seed)
    
    adj, deg = _build_adj_deg(n, initial_edges)
    current_score = _score_graph_fast(n, adj, deg)
    current_surrogate = _score_graph_sum_sq(n, adj, deg) if use_surrogate else 0
    
    best_edges = [e[:] for e in initial_edges]
    best_score = current_score
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    current_edge_set = set((min(u,v), max(u,v)) for u, v in initial_edges)
    
    # Temperature schedule
    T_start = 8.0
    T_end = 0.05
    n_iter_total = 80000
    i = 0
    
    while time.time() < deadline:
        i += 1
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
        new_surrogate = _score_graph_sum_sq(n, adj, deg) if use_surrogate else 0
        
        # Use a combined objective: primarily max score, secondarily surrogate
        if use_surrogate:
            # Primary: minimize max score; secondary: minimize surrogate
            if new_score < current_score:
                accept = True
            elif new_score == current_score:
                if new_surrogate < current_surrogate:
                    accept = True
                else:
                    # Accept with some probability based on surrogate improvement
                    delta_surrogate = new_surrogate - current_surrogate
                    accept = rng.random() < math.exp(-delta_surrogate / max(T * 10, 1e-10))
            else:
                delta = new_score - current_score
                accept = rng.random() < math.exp(-delta / max(T, 1e-10))
        else:
            delta = new_score - current_score
            accept = delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10))
        
        if accept:
            current_score = new_score
            current_surrogate = new_surrogate
            if new_score < best_score:
                best_score = new_score
                best_edges = [e[:] for e in initial_edges]
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

def _hill_climb(n, initial_edges, deadline, seed=42):
    """Hill climbing: only accept strictly improving moves."""
    rng = random.Random(seed)
    
    adj, deg = _build_adj_deg(n, initial_edges)
    current_score = _score_graph_fast(n, adj, deg)
    current_surrogate = _score_graph_sum_sq(n, adj, deg)
    
    best_edges = [e[:] for e in initial_edges]
    best_score = current_score
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    current_edge_set = set((min(u,v), max(u,v)) for u, v in initial_edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle edge removal order for diversity
        edges_to_try = list(initial_edges)
        rng.shuffle(edges_to_try)
        
        for e_remove in edges_to_try:
            if time.time() > deadline:
                break
            u1, v1 = e_remove
            e_remove_key = (min(u1, v1), max(u1, v1))
            
            # Try all possible non-edges to add
            non_edges = [e for e in all_edges if (min(e[0], e[1]), max(e[0], e[1])) not in current_edge_set]
            rng.shuffle(non_edges)
            
            for e_add in non_edges:
                e_add_key = (min(e_add[0], e_add[1]), max(e_add[0], e_add[1]))
                
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
                new_surrogate = _score_graph_sum_sq(n, adj, deg)
                
                if new_score < current_score or (new_score == current_score and new_surrogate < current_surrogate):
                    # Accept
                    current_edge_set.remove(e_remove_key)
                    current_edge_set.add(e_add_key)
                    initial_edges.remove(e_remove)
                    initial_edges.append(e_add)
                    current_score = new_score
                    current_surrogate = new_surrogate
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = [e[:] for e in initial_edges]
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
    
    return best_edges, best_score

def _generate_initial_configs(n, target_edges, rng):
    """Generate diverse initial configurations."""
    configs = []
    
    # Config 1: K_{8,8} + one edge inside part A (vertices 0-7)
    edges1 = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    configs.append(edges1)
    
    # Config 2: K_{8,8} + one edge inside part B (vertices 8-15)
    edges2 = [[u, v] for u in range(8) for v in range(8, 16)] + [[8, 9]]
    configs.append(edges2)
    
    # Config 3: K_{8,8} + edge connecting high-degree vertices
    edges3 = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 7]]
    configs.append(edges3)
    
    # Config 4: K_{8,8} + edge connecting low-degree vertices  
    edges4 = [[u, v] for u in range(8) for v in range(8, 16)] + [[3, 4]]
    configs.append(edges4)
    
    # Config 5-8: Random perturbations of K_{8,8}
    for trial in range(4):
        edges = [[u, v] for u in range(8) for v in range(8, 16)]
        # Remove a random cross edge
        u = rng.randint(0, 7)
        v = rng.randint(8, 15)
        edges.remove([u, v])
        # Add a random within-part edge
        if rng.random() < 0.5:
            a, b = rng.sample(range(8), 2)
        else:
            a, b = rng.sample(range(8, 16), 2)
        edges.append([a, b])
        configs.append(edges)
    
    # Config 9-12: More diverse random configs
    for trial in range(4):
        edges = []
        # Start with a random near-bipartite structure
        left = list(range(8))
        right = list(range(8, 16))
        
        # Add most cross edges
        for u in left:
            for v in right:
                if rng.random() < 0.92:
                    edges.append([u, v])
        
        # Add within-part edges to reach target
        while len(edges) < target_edges:
            if rng.random() < 0.5:
                u, v = rng.sample(left, 2)
            else:
                u, v = rng.sample(right, 2)
            edge = [min(u, v), max(u, v)]
            if edge not in edges:
                edges.append(edge)
        
        # Trim to exact count if needed
        if len(edges) > target_edges:
            rng.shuffle(edges)
            edges = edges[:target_edges]
        
        if len(edges) == target_edges:
            configs.append(edges)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    rng = random.Random(42)
    
    # Generate multiple initial configurations
    initial_configs = _generate_initial_configs(n, target_edges, rng)
    
    best_edges = None
    best_score = float('inf')
    
    # Phase 1: Simulated annealing from each initial config
    num_configs = len(initial_configs)
    per_config_time = (deadline - time.time()) / (num_configs * 2)  # Half time for SA, half for hill climb
    
    for idx, init_edges in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        config_deadline = time.time() + per_config_time
        use_surrogate = (idx % 2 == 0)  # Alternate between surrogate and direct
        
        edges, score = _simulated_annealing(n, [e[:] for e in init_edges], config_deadline, seed=100 + idx, use_surrogate=use_surrogate)
        
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 2: Hill climbing from best solution
    if best_edges is not None and time.time() < deadline - 10:
        hc_deadline = deadline - 5
        edges, score = _hill_climb(n, [e[:] for e in best_edges], hc_deadline, seed=200)
        
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 3: Additional SA restarts from best if time remains
    if time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            restart_deadline = time.time() + (deadline - 5 - time.time()) / (3 - restart)
            edges, score = _simulated_annealing(n, [e[:] for e in best_edges], restart_deadline, seed=300 + restart, use_surrogate=True)
            
            if score < best_score:
                best_score = score
                best_edges = edges
    
    # Phase 4: Final hill climb
    if best_edges is not None and time.time() < deadline - 2:
        hc_deadline = deadline - 1
        edges, score = _hill_climb(n, [e[:] for e in best_edges], hc_deadline, seed=400)
        
        if score < best_score:
            best_score = score
            best_edges = edges
    
    if best_edges is None:
        # Fallback to initial construction
        best_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
