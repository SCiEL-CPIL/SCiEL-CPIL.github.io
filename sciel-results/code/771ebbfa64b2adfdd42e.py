import numpy as np
import time
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n = 20
    max_edges = n * (n - 1) // 2
    
    # Precompute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
    
    # State: adjacency matrix
    def random_dense_graph():
        # Start with a moderately dense random graph
        p = 0.5
        adj = np.zeros((n, n), dtype=np.int8)
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u, v] = 1
                    adj[v, u] = 1
        return adj
    
    def count_violations(adj):
        """Count number of K2,3 violations (pairs with >=3 common neighbors)."""
        violations = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = np.sum(adj[u] & adj[v])
                if common >= 3:
                    violations += 1
        return violations
    
    def edge_count(adj):
        return np.sum(adj) // 2
    
    def energy(adj):
        """Energy = -edges + penalty * violations"""
        e = edge_count(adj)
        v = count_violations(adj)
        return -e + 100 * v
    
    # Simulated annealing
    best_adj = random_dense_graph()
    best_energy = energy(best_adj)
    best_edges = edge_count(best_adj)
    
    current_adj = best_adj.copy()
    current_energy = best_energy
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline:
        # Try a random edge flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        old_val = current_adj[u, v]
        current_adj[u, v] = 1 - old_val
        current_adj[v, u] = 1 - old_val
        
        new_energy = energy(current_adj)
        delta_e = new_energy - current_energy
        
        if delta_e <= 0 or random.random() < np.exp(-delta_e / max(T, T_min)):
            current_energy = new_energy
            new_edges = edge_count(current_adj)
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = current_adj.copy()
        else:
            # Revert
            current_adj[u, v] = old_val
            current_adj[v, u] = old_val
        
        T *= cooling
    
    # Return edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    return edges
