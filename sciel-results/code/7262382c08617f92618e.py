"""Construct a graph with no induced regular subgraph on 6+ vertices, maximizing n."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 148.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - start_time)

    MAX_N = 22

    # Precompute C(n, k) for quick subset counts
    # For n up to 22, we can check subsets of size 6, 7, 8, ...

    # Fast regularity check for a subset using numpy on adjacency matrix
    def is_regular_subset(adj_mat, subset_indices):
        """Check if induced subgraph on subset_indices is regular."""
        k = len(subset_indices)
        if k < 6:
            return False
        # Extract submatrix
        sub = adj_mat[np.ix_(subset_indices, subset_indices)]
        degs = sub.sum(axis=1)
        return np.all(degs == degs[0])

    # Full verification of a graph
    def full_verify(adj_mat, n):
        """Verify no induced regular subgraph of size >= 6."""
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                if is_regular_subset(adj_mat, subset):
                    return False
        return True

    # Try to build a valid graph using local search
    # Start with a random graph and flip edges to eliminate regular subsets
    
    def random_graph(n, p=0.5):
        """Generate a random graph with edge probability p."""
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    edges.add((i, j))
        return edges

    def edges_to_adj_mat(edges, n):
        """Convert edge set to numpy adjacency matrix."""
        adj = np.zeros((n, n), dtype=np.int8)
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1
        return adj

    def adj_mat_to_edges(adj_mat, n):
        """Convert adjacency matrix to edge list."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mat[i, j]:
                    edges.append([i, j])
        return edges

    def find_all_bad_subsets(adj_mat, n, min_size=6):
        """Find all induced regular subgraphs of size >= min_size."""
        bad = []
        for size in range(min_size, n + 1):
            for subset in combinations(range(n), size):
                if is_regular_subset(adj_mat, subset):
                    bad.append(subset)
        return bad

    def flip_edge(adj_mat, i, j):
        """Flip edge (i,j) in adjacency matrix. Returns new matrix."""
        adj_mat = adj_mat.copy()
        adj_mat[i, j] = 1 - adj_mat[i, j]
        adj_mat[j, i] = adj_mat[i, j]
        return adj_mat

    # Local search: start with random graph, flip edges to reduce bad subsets
    def local_search(n, max_iters=5000, timeout=30.0):
        """
        Local search to find a graph with no regular induced subgraphs of size >= 6.
        Cost = number of bad subsets. We want to minimize this to 0.
        """
        best_adj = None
        best_cost = float('inf')
        
        for attempt in range(20):  # multiple restarts
            if time_left() < 5:
                break
            
            # Start with random graph
            p = random.uniform(0.3, 0.7)
            edges = random_graph(n, p)
            adj = edges_to_adj_mat(edges, n)
            
            # Compute initial cost
            bad = find_all_bad_subsets(adj, n)
            cost = len(bad)
            
            if cost == 0:
                return adj
            
            if cost < best_cost:
                best_cost = cost
                best_adj = adj.copy()
            
            # Local search
            for it in range(max_iters):
                if time_left() < 5:
                    break
                if cost == 0:
                    return adj
                
                # Pick a random bad subset and try to fix it by flipping an edge within it
                if bad:
                    # Pick a random bad subset
                    subset = random.choice(bad)
                    subset_set = set(subset)
                    
                    # Try flipping each edge within the subset
                    best_flip = None
                    best_new_cost = cost
                    
                    for idx_i in range(len(subset)):
                        for idx_j in range(idx_i + 1, len(subset)):
                            i, j = subset[idx_i], subset[idx_j]
                            
                            # Flip this edge
                            new_adj = flip_edge(adj, i, j)
                            
                            # Recompute bad subsets (expensive, but necessary)
                            new_bad = find_all_bad_subsets(new_adj, n)
                            new_cost = len(new_bad)
                            
                            if new_cost < best_new_cost:
                                best_new_cost = new_cost
                                best_flip = (i, j, new_adj, new_bad)
                    
                    if best_flip is None:
                        # Try random edge flips
                        for _ in range(10):
                            i = random.randint(0, n - 1)
                            j = random.randint(0, n - 1)
                            if i == j:
                                continue
                            if i > j:
                                i, j = j, i
                            
                            new_adj = flip_edge(adj, i, j)
                            new_bad = find_all_bad_subsets(new_adj, n)
                            new_cost = len(new_bad)
                            
                            if new_cost < best_new_cost:
                                best_new_cost = new_cost
                                best_flip = (i, j, new_adj, new_bad)
                    
                    if best_flip is not None:
                        adj = best_flip[2]
                        bad = best_flip[3]
                        cost = best_new_cost
                        if cost < best_cost:
                            best_cost = cost
                            best_adj = adj.copy()
                    else:
                        # Stuck, restart
                        break
                else:
                    break
        
        return best_adj

    # Try to find the largest valid graph
    best_n = 0
    best_edges = []
    
    # Try n from 22 down to 1
    for n in range(MAX_N, 0, -1):
        if time_left() < 10:
            break
        
        # Try multiple local search runs
        found = False
        for attempt in range(50):
            if time_left() < 10:
                break
            
            adj = local_search(n, max_iters=2000, timeout=20.0)
            
            if adj is not None:
                # Verify
                if full_verify(adj, n):
                    best_n = n
                    best_edges = adj_mat_to_edges(adj, n)
                    found = True
                    break
        
        if found:
            break
    
    # If no graph found, return empty
    if best_n == 0:
        return {'n': 1, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
