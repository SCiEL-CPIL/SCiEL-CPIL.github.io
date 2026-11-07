import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Edge indexing
    edge_map = {}
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[idx] = (u, v)
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            idx += 1
            
    def get_adjacency(edges):
        adj = [set() for _ in range(n)]
        for e in edges:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]

    def count_violations(edges):
        if not edges:
            return 0
        adj = get_adjacency(edges)
        deg = get_degrees(adj)
        
        violations = 0
        for e in edges:
            u, v = edge_map[e]
            # s in N(u)\{v}, t in N(v)\{u}
            # if s != t and deg[s] == deg[t] -> violation
            su = adj[u] - {v}
            tv = adj[v] - {u}
            
            # Group by degree for efficiency?
            # n is small, just nested loop
            for s in su:
                ds = deg[s]
                for t in tv:
                    if s != t:
                        if ds == deg[t]:
                            violations += 1
        return violations

    # Initial state: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    current_edges = set()
    for u in A:
        for v in B:
            current_edges.add(edge_index[(u, v)])
            
    best_edges = current_edges.copy()
    best_violations = count_violations(best_edges) # Should be 0
    best_count = len(best_edges)
    
    T = 10.0
    min_T = 0.01
    
    # Precompute list of all edges
    all_edge_ids = list(range(total_edges))
    
    while time.time() < deadline:
        # Generate move
        r = random.random()
        new_edges = current_edges.copy()
        
        if r < 0.5:
            # Add edge
            missing = [e for e in all_edge_ids if e not in current_edges]
            if missing:
                e_add = random.choice(missing)
                new_edges.add(e_add)
        elif r < 0.7:
            # Remove edge
            if current_edges:
                e_rem = random.choice(list(current_edges))
                new_edges.remove(e_rem)
        else:
            # Swap
            if current_edges:
                missing = [e for e in all_edge_ids if e not in current_edges]
                if missing:
                    e_rem = random.choice(list(current_edges))
                    e_add = random.choice(missing)
                    new_edges.remove(e_rem)
                    new_edges.add(e_add)

        v_new = count_violations(new_edges)
        v_curr = count_violations(current_edges)
        
        d_edges = len(new_edges) - len(current_edges)
        d_viol = v_new - v_curr
        
        # We want to maximize edges, minimize violations
        # Score = Edges - 100 * Violations
        d_score = d_edges - 100 * d_viol
        
        accept = False
        if d_score >= 0:
            accept = True
        else:
            # Simulated Annealing
            if T > min_T:
                prob = math.exp(d_score / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            current_edges = new_edges
            if v_new == 0:
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = current_edges.copy()
            
        T *= 0.995
        
        # Periodic restart if stuck?
        # If T is very low and no improvement for a while, maybe random restart from best?
        
    # Final check
    if count_violations(best_edges) != 0:
        # Fallback to K_{7,8} if best is invalid (should not happen)
        best_edges = set()
        for u in A:
            for v in B:
                best_edges.add(edge_index[(u, v)])
                
    result = []
    for e in best_edges:
        u, v = edge_map[e]
        result.append([u, v])
        
    return result
