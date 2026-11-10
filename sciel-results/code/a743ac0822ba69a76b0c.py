# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    Uses simulated annealing with vectorized NumPy operations.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 20
    num_pairs = n * (n - 1) // 2
    
    # Precompute pair indices for vectorization
    pair_u = np.array([i for i in range(n) for j in range(i+1, n)])
    pair_v = np.array([j for i in range(n) for j in range(i+1, n)])
    
    best_edges = None
    best_score = -1
    
    def compute_energy(adj):
        """Compute energy: -edges + penalty * violations."""
        # adj is n x n binary matrix
        # Common neighbors: (adj @ adj) gives common neighbor counts
        common = adj @ adj  # common[u,v] = number of common neighbors of u,v
        # Count pairs with >= 3 common neighbors (K2,3 violations)
        violations = np.sum(common[np.triu_indices(n, k=1)] >= 3)
        edge_count = np.sum(adj[np.triu_indices(n, k=1)])
        penalty = 100.0
        return -(edge_count - penalty * violations)
    
    def adj_to_edges(adj):
        """Convert adjacency matrix to edge list."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges
    
    def is_k23_free(adj):
        """Check if adjacency matrix has no K2,3."""
        common = adj @ adj
        return np.all(common[np.triu_indices(n, k=1)] < 3)
    
    def repair(adj):
        """Remove edges to fix K2,3 violations."""
        adj = adj.copy()
        for _ in range(100):  # max iterations
            common = adj @ adj
            # Find violating pairs
            violating = False
            for i in range(n):
                for j in range(i+1, n):
                    if common[i, j] >= 3:
                        violating = True
                        # Remove an edge incident to a common neighbor
                        common_neighbors = np.where((adj[i] == 1) & (adj[j] == 1))[0]
                        if len(common_neighbors) > 0:
                            # Remove edge from a common neighbor to one of the pair
                            # Try removing (common_neighbor, i) or (common_neighbor, j)
                            cn = common_neighbors[0]
                            adj[cn, i] = 0
                            adj[i, cn] = 0
                            break
                if violating:
                    break
            if not violating:
                break
        return adj
    
    def simulated_annealing(init_adj, t_start, t_end, steps):
        """Run simulated annealing from initial adjacency."""
        adj = init_adj.copy()
        current_energy = compute_energy(adj)
        best_local = adj.copy()
        best_local_energy = current_energy
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge to flip
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i == j:
                continue
            
            # Compute energy delta
            old_val = adj[i, j]
            adj[i, j] = 1 - old_val
            adj[j, i] = 1 - old_val
            
            new_energy = compute_energy(adj)
            delta_e = new_energy - current_energy
            
            # Temperature schedule
            t = t_start * (t_end / t_start) ** (step / steps)
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / max(t, 1e-10)):
                current_energy = new_energy
                if new_energy < best_local_energy:
                    best_local = adj.copy()
                    best_local_energy = new_energy
            else:
                # Reject: revert
                adj[i, j] = old_val
                adj[j, i] = old_val
        
        return best_local, best_local_energy
    
    # Multiple restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with varying density
        density = 0.3 + 0.5 * random.random()
        init_adj = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    init_adj[i, j] = 1
                    init_adj[j, i] = 1
        
        # Repair initial graph
        init_adj = repair(init_adj)
        
        # Run simulated annealing
        t_start = 1.0
        t_end = 0.01
        steps = 5000
        result_adj, result_energy = simulated_annealing(init_adj, t_start, t_end, steps)
        
        # Final repair to ensure validity
        result_adj = repair(result_adj)
        
        # Check validity
        if is_k23_free(result_adj):
            edges = adj_to_edges(result_adj)
            edge_count = len(edges)
            if edge_count > best_score:
                best_score = edge_count
                best_edges = edges
    
    # If no valid graph found, return a simple valid one
    if best_edges is None:
        # Return five disjoint K4's (30 edges)
        edges = []
        for k in range(5):
            for a in range(4):
                for b in range(a+1, 4):
                    edges.append([4*k + a, 4*k + b])
        return edges
    
    return best_edges
# EVOLVE-BLOCK-END
