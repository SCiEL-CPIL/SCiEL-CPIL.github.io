import time
import random
import math

def construct():
    n = 16
    m = 65
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Use bitmasks for fast triangle detection
    # adj[u] is a bitmask of neighbors of u
    # deg[u] is the degree of u
    
    def calc_max_score(adj, deg):
        """Calculate max triangle degree sum using bitmasks."""
        max_sum = 0
        for u in range(n):
            du = deg[u]
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    w = v + 1
                    while w < n:
                        if (common >> w) & 1:
                            s = du + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
                        w += 1
        return max_sum

    def calc_sum_sq_score(adj, deg):
        """Calculate sum of squared triangle degree sums (QUBO-like objective)."""
        total = 0
        for u in range(n):
            du = deg[u]
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    dv = deg[v]
                    common = adj[u] & adj[v]
                    w = v + 1
                    while w < n:
                        if (common >> w) & 1:
                            s = du + dv + deg[w]
                            total += s * s
                        w += 1
        return total

    # Initialize with K_{8,8} + edge (0,1)
    adj = [0] * n
    deg = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1

    # Build K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            add_edge(u, v)
    # Add one edge inside part 0
    add_edge(0, 1)
    
    current_max_score = calc_max_score(adj, deg)
    current_sq_score = calc_sum_sq_score(adj, deg)
    best_max_score = current_max_score
    best_adj = adj[:]
    best_deg = deg[:]
    
    # Simulated Annealing with QUBO-like objective
    T = 200.0
    cool_rate = 0.9997
    min_T = 5.0
    
    # Current edge list for quick reference
    current_edges = set()
    non_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                current_edges.add((u, v))
            else:
                non_edges.add((u, v))
    
    while time.time() < deadline:
        # Pick a random edge to remove
        u, v = random.choice(list(current_edges))
        
        # Pick a random non-edge to add
        x, y = random.choice(list(non_edges))
        
        # Perform move
        remove_edge(u, v)
        add_edge(x, y)
        current_edges.remove((u, v))
        non_edges.remove((x, y))
        current_edges.add((x, y))
        non_edges.add((u, v))
        
        new_sq_score = calc_sum_sq_score(adj, deg)
        new_max_score = calc_max_score(adj, deg)
        
        # Acceptance: primarily based on sum-of-squares, with hard constraint on max
        delta_sq = new_sq_score - current_sq_score
        delta_max = new_max_score - current_max_score
        
        # Use combined acceptance: prefer lower sum-of-squares, but also consider max
        if delta_max < 0:
            # Strictly better on max score
            accept = True
        elif delta_max == 0:
            # Same max, use sum-of-squares
            accept = delta_sq < 0 or random.random() < math.exp(-delta_sq / T)
        else:
            # Worse max score - only accept with small probability if sum-of-squares much better
            # This allows exploration but discourages worsening the primary objective
            if delta_sq < -1000:
                accept = random.random() < math.exp(-delta_sq / (T * 2))
            else:
                accept = False
        
        if accept:
            current_max_score = new_max_score
            current_sq_score = new_sq_score
            if new_max_score < best_max_score:
                best_max_score = new_max_score
                best_adj = adj[:]
                best_deg = deg[:]
        else:
            # Revert
            remove_edge(x, y)
            add_edge(u, v)
            current_edges.remove((x, y))
            non_edges.remove((u, v))
            current_edges.add((u, v))
            non_edges.add((x, y))
            
        T = max(T * cool_rate, min_T)

    # Final polish: local search on best solution targeting max score
    adj[:] = best_adj
    deg[:] = best_deg
    current_max_score = best_max_score
    current_edges = set()
    non_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                current_edges.add((u, v))
            else:
                non_edges.add((u, v))
    
    polish_deadline = start_time + 115
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        for u, v in list(current_edges):
            for x, y in list(non_edges):
                if x == u or x == v or y == u or y == v:
                    continue
                # Try swap
                remove_edge(u, v)
                add_edge(x, y)
                current_edges.remove((u, v))
                non_edges.remove((x, y))
                current_edges.add((x, y))
                non_edges.add((u, v))
                
                new_max_score = calc_max_score(adj, deg)
                
                if new_max_score < current_max_score:
                    current_max_score = new_max_score
                    if current_max_score < best_max_score:
                        best_max_score = current_max_score
                        best_adj = adj[:]
                        best_deg = deg[:]
                    improved = True
                else:
                    # Revert
                    remove_edge(x, y)
                    add_edge(u, v)
                    current_edges.remove((x, y))
                    non_edges.remove((u, v))
                    current_edges.add((u, v))
                    non_edges.add((x, y))
                
                if time.time() > polish_deadline:
                    break
            if time.time() > polish_deadline:
                break

    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
