"""Stronger algorithm for dense K3,3-free graphs on 20 vertices."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 20

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges_total = len(all_edges)  # 190

    # Precompute for each edge, which other edges share vertices with it
    # This helps in fast validity checking
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx

    # For each vertex, list of edges incident to it
    vertex_edges = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(all_edges):
        vertex_edges[u].append(idx)
        vertex_edges[v].append(idx)

    # Precompute for each pair of vertices, the list of edges that connect to both
    # i.e., edges (a,b) where a is in N(u) and b is in N(v) - this is complex
    # Instead, we'll use a different approach for validity checking

    def count_edges(adj_bits):
        return sum(b.bit_count() for b in adj_bits) // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def is_k33_free(adj_bits):
        """Check if graph is K3,3-free using bitsets."""
        for i in range(n):
            for j in range(i+1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    def get_missing(adj_bits):
        return [(i, j) for i in range(n) for j in range(i+1, n)
                if not (adj_bits[i] >> j) & 1]

    def get_existing(adj_bits):
        return [(i, j) for i in range(n) for j in range(i+1, n)
                if (adj_bits[i] >> j) & 1]

    def hill_climb(adj_bits, deadline):
        """Fast hill climbing to reach a local optimum."""
        current = adj_bits[:]
        current_edges = count_edges(current)
        
        missing = get_missing(current)
        random.shuffle(missing)
        existing = get_existing(current)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(missing)
            
            new_missing = []
            for u, v in missing:
                if time.time() > deadline - 5:
                    new_missing.append((u, v))
                    continue
                add_edge(current, u, v)
                if is_k33_free(current):
                    current_edges += 1
                    improved = True
                    existing.append((u, v))
                else:
                    remove_edge(current, u, v)
                    new_missing.append((u, v))
            
            if improved:
                missing = new_missing
                continue
            
            # Try swaps if stuck
            if existing and missing and time.time() < deadline - 5:
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    e1 = random.choice(existing)
                    e2 = random.choice(missing)
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    remove_edge(current, u1, v1)
                    add_edge(current, u2, v2)
                    if is_k33_free(current):
                        existing.remove(e1)
                        existing.append(e2)
                        missing.remove(e2)
                        missing.append(e1)
                        improved = True
                    else:
                        remove_edge(current, u2, v2)
                        add_edge(current, u1, v1)
        
        return current, current_edges

    def simulated_annealing(adj_bits, deadline):
        """Simulated annealing to escape local optima."""
        current = adj_bits[:]
        current_edges = count_edges(current)
        best = current[:]
        best_edges = current_edges
        
        # Initial temperature
        T = 5.0
        cooling_rate = 0.9995
        min_T = 0.1
        
        # Precompute for faster validity checks
        # For each edge (u,v), precompute the set of common neighbors of u and v
        # and count how many have >= 3 common neighbors with any third vertex
        
        existing = get_existing(current)
        missing = get_missing(current)
        
        steps = 0
        while time.time() < deadline - 5 and T > min_T:
            steps += 1
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random missing edge
                if not missing:
                    T *= cooling_rate
                    continue
                u, v = random.choice(missing)
                add_edge(current, u, v)
                new_edges = current_edges + 1
                
                if is_k33_free(current):
                    # Accept
                    current_edges = new_edges
                    existing.append((u, v))
                    missing.remove((u, v))
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best = current[:]
                else:
                    # Reject
                    remove_edge(current, u, v)
                
            elif move_type < 0.6:
                # Remove a random existing edge
                if not existing:
                    T *= cooling_rate
                    continue
                u, v = random.choice(existing)
                remove_edge(current, u, v)
                new_edges = current_edges - 1
                
                # Always accept removals (they maintain validity)
                current_edges = new_edges
                existing.remove((u, v))
                missing.append((u, v))
                
            else:
                # Swap: remove one existing, add one missing
                if not existing or not missing:
                    T *= cooling_rate
                    continue
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                
                remove_edge(current, u1, v1)
                add_edge(current, u2, v2)
                
                if is_k33_free(current):
                    # Accept
                    existing.remove((u1, v1))
                    existing.append((u2, v2))
                    missing.remove((u2, v2))
                    missing.append((u1, v1))
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best = current[:]
                else:
                    # Reject
                    remove_edge(current, u2, v2)
                    add_edge(current, u1, v1)
            
            T *= cooling_rate
        
        # Final hill climb on the best found
        if time.time() < deadline - 3:
            best, best_edges = hill_climb(best, deadline - 3)
        
        return best, best_edges

    def init_random_dense(deadline):
        """Initialize with a random dense graph, then prune."""
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    add_edge(adj, i, j)
        # Prune to make K3,3-free
        all_edges_local = [(i, j) for i in range(n) for j in range(i+1, n)
                           if (adj[i] >> j) & 1]
        random.shuffle(all_edges_local)
        adj_clean = [0] * n
        for u, v in all_edges_local:
            add_edge(adj_clean, u, v)
            if not is_k33_free(adj_clean):
                remove_edge(adj_clean, u, v)
        return adj_clean

    def init_random_greedy(deadline):
        """Build a random maximal K3,3-free graph via greedy edge addition."""
        adj = [0] * n
        all_edges_local = all_edges[:]
        random.shuffle(all_edges_local)
        
        for u, v in all_edges_local:
            if time.time() > deadline - 10:
                break
            add_edge(adj, u, v)
            if not is_k33_free(adj):
                remove_edge(adj, u, v)
        return adj

    best_adj = None
    best_edges = -1

    # Phase 1: Quick hill climb from random greedy initialization
    if time.time() < deadline - 30:
        adj1 = init_random_greedy(deadline)
        adj1, e1 = hill_climb(adj1, deadline - 25)
        if e1 > best_edges:
            best_edges = e1
            best_adj = adj1[:]

    # Phase 2: Simulated annealing from dense random start
    if time.time() < deadline - 20:
        adj2 = init_random_dense(deadline)
        # First hill climb
        adj2, e2 = hill_climb(adj2, deadline - 15)
        # Then simulated annealing
        adj2, e2 = simulated_annealing(adj2, deadline - 10)
        if e2 > best_edges:
            best_edges = e2
            best_adj = adj2[:]

    # Phase 3: Simulated annealing from the best found so far
    if best_adj is not None and time.time() < deadline - 10:
        adj3 = best_adj[:]
        adj3, e3 = simulated_annealing(adj3, deadline - 5)
        if e3 > best_edges:
            best_edges = e3
            best_adj = adj3[:]

    # Phase 4: Another random dense start + SA
    if time.time() < deadline - 15:
        adj4 = init_random_dense(deadline)
        adj4, e4 = hill_climb(adj4, deadline - 10)
        adj4, e4 = simulated_annealing(adj4, deadline - 5)
        if e4 > best_edges:
            best_edges = e4
            best_adj = adj4[:]

    # Phase 5: Final SA polish on the best
    if best_adj is not None and time.time() < deadline - 5:
        adj5 = best_adj[:]
        adj5, e5 = simulated_annealing(adj5, deadline - 2)
        if e5 > best_edges:
            best_edges = e5
            best_adj = adj5[:]

    # Final hill climb polish
    if best_adj is not None and time.time() < deadline - 2:
        best_adj, best_edges = hill_climb(best_adj, deadline - 2)

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
