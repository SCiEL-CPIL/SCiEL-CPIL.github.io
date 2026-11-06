import time
import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def compute_books(adj):
        """Compute book size for every edge using matrix multiplication."""
        # adj is n x n boolean adjacency matrix
        # For each edge (i,j), book size = number of common neighbors = sum_k adj[i,k] & adj[j,k]
        # This is (adj @ adj.T)[i,j] for i != j, but we only care about edges
        # Actually, common neighbors of i and j = sum over k of adj[i,k] * adj[j,k]
        # That's exactly (adj @ adj.T)[i,j]
        common = adj @ adj.T
        # For each edge (i,j) where i<j, the book size is common[i,j]
        # We want the max over all edges
        # Get upper triangle (excluding diagonal)
        iu = np.triu_indices(n, k=1)
        edge_books = common[iu]
        # Only consider actual edges
        edge_books = edge_books[adj[iu] > 0]
        if len(edge_books) == 0:
            return 0
        return int(np.max(edge_books))
    
    def edges_to_adj(edge_set):
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edge_set:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def adj_to_edges(adj):
        edge_set = set()
        iu = np.triu_indices(n, k=1)
        for i, j in zip(*iu):
            if adj[i, j]:
                edge_set.add((i, j))
        return edge_set
    
    # Phase 1: Greedy construction
    # Start with no edges, add edges one by one minimizing max book size
    # Use a priority: prefer edges that don't create large books
    
    best_edge_set = None
    best_score = float('inf')
    
    # Try multiple random initializations
    for trial in range(5):
        if time.time() > deadline - 30:
            break
            
        adj = np.zeros((n, n), dtype=np.int8)
        edge_set = set()
        
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Greedy: repeatedly add the edge that minimizes the resulting max book
        # To speed up, we can be smarter: start with a random subset, then optimize
        
        # Strategy: Start with a random 401-edge graph, then do local search
        random.shuffle(all_edges)
        for e in all_edges[:target_edges]:
            edge_set.add(e)
        
        adj = edges_to_adj(edge_set)
        score = compute_books(adj)
        
        if score < best_score:
            best_score = score
            best_edge_set = set(edge_set)
        
        # Phase 2: Local search with edge swaps
        T = 3.0
        cooling = 0.9995
        current_score = score
        current_adj = adj.copy()
        current_edge_set = set(edge_set)
        
        # Precompute non-edges for fast sampling
        all_edges_set = set(all_edges)
        
        while time.time() < deadline - 10:
            # Pick a random edge to remove
            u, v = random.choice(list(current_edge_set))
            current_edge_set.remove((u, v))
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            
            # Pick a random non-edge to add
            # Sample from all non-edges
            non_edges = [e for e in all_edges if e not in current_edge_set]
            if not non_edges:
                break
            a, b = random.choice(non_edges)
            current_edge_set.add((a, b))
            current_adj[a, b] = 1
            current_adj[b, a] = 1
            
            new_score = compute_books(current_adj)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edge_set = set(current_edge_set)
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        # Reject: revert
                        current_edge_set.remove((a, b))
                        current_adj[a, b] = 0
                        current_adj[b, a] = 0
                        current_edge_set.add((u, v))
                        current_adj[u, v] = 1
                        current_adj[v, u] = 1
                else:
                    current_score = new_score
            
            T *= cooling
            if T < 0.5:
                T = 0.5
        
        if time.time() > deadline - 10:
            break
    
    if best_edge_set is None:
        # Fallback: K20,20 + 1
        best_edge_set = set()
        for u in range(20):
            for v in range(20, 40):
                best_edge_set.add((u, v))
        u, v = random.sample(range(20), 2)
        best_edge_set.add((u, v))
    
    return [[u, v] for u, v in best_edge_set]
