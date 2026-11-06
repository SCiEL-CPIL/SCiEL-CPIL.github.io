"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 16 vertices with exactly 65 edges that minimizes
    the maximum degree sum over triangles.
    
    Strategy: Start with K_{8,8} + one edge (score ~25-26) and use local
    search to find better configurations. The key insight is that we want
    to distribute edges more evenly while keeping triangles' degree sums low.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    # Helper functions
    def build_adj(edges):
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj
    
    def compute_score(adj):
        """Compute the maximum degree sum over all triangles."""
        deg = [sum(adj[i]) for i in range(n)]
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges
    
    def random_edge_swap(adj, edges):
        """Try a random edge swap: remove edge (a,b), add edge (c,d)."""
        if len(edges) < 2:
            return None
        # Pick two random edges to swap
        i, j = random.sample(range(len(edges)), 2)
        a, b = edges[i]
        c, d = edges[j]
        
        # We want to remove (a,b) and add (c,d) where (c,d) doesn't exist
        # and (a,b) exists
        if adj[c][d] == 1:  # (c,d) already exists, can't add
            return None
        if adj[a][b] == 0:  # (a,b) doesn't exist, can't remove
            return None
        
        # Try the swap
        adj[a][b] = 0
        adj[b][a] = 0
        adj[c][d] = 1
        adj[d][c] = 1
        
        new_edges = edges_from_adj(adj)
        new_score = compute_score(adj)
        
        # Check if valid (still 65 edges)
        if len(new_edges) != target_edges:
            # Revert
            adj[a][b] = 1
            adj[b][a] = 1
            adj[c][d] = 0
            adj[d][c] = 0
            return None
        
        return new_edges, new_score
    
    # Initial construction: K_{8,8} + one edge
    # Part A: vertices 0-7, Part B: vertices 8-15
    adj = [[0]*n for _ in range(n)]
    edges = []
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
            edges.append([u, v])
    # Add one edge within part A
    adj[0][1] = 1
    adj[1][0] = 1
    edges.append([0, 1])
    
    current_score = compute_score(adj)
    best_score = current_score
    best_edges = edges[:]
    
    # Local search with simulated annealing
    temperature = 5.0
    cooling_rate = 0.995
    max_iterations = 50000
    
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Random edge swap
        if len(edges) < 2:
            break
        
        i, j = random.sample(range(len(edges)), 2)
        a, b = edges[i]
        c, d = edges[j]
        
        # Check if swap is valid
        if adj[c][d] == 1 or adj[a][b] == 0:
            continue
        
        # Apply swap
        adj[a][b] = 0
        adj[b][a] = 0
        adj[c][d] = 1
        adj[d][c] = 1
        
        new_edges = edges_from_adj(adj)
        new_score = compute_score(adj)
        
        if len(new_edges) != target_edges:
            # Revert
            adj[a][b] = 1
            adj[b][a] = 1
            adj[c][d] = 0
            adj[d][c] = 0
            continue
        
        # Accept if better or with probability exp(-(delta)/T)
        if new_score < current_score:
            current_score = new_score
            edges = new_edges
            if new_score < best_score:
                best_score = new_score
                best_edges = edges[:]
        else:
            delta = new_score - current_score
            if delta == 0:
                # Accept equal scores
                current_score = new_score
                edges = new_edges
            else:
                prob = np.exp(-delta / temperature)
                if random.random() < prob:
                    current_score = new_score
                    edges = new_edges
                else:
                    # Revert
                    adj[a][b] = 1
                    adj[b][a] = 1
                    adj[c][d] = 0
                    adj[d][c] = 0
                    edges = edges_from_adj(adj)
        
        temperature *= cooling_rate
    
    # Return the best found
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
