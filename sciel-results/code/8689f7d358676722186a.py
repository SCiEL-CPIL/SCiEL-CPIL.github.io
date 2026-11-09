import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    t_start = time.time()
    deadline = t_start + 115.0  # Leave 5s margin for 120s limit

    def get_max_book(adj):
        # adj is 40x40 numpy array of 0s and 1s
        # Vectorized calculation of common neighbors for all edges
        # adj @ adj.T gives number of common neighbors for all pairs
        # But we only care about edges where adj[i,j] == 1
        common = np.dot(adj, adj.T)
        # Mask with adj to only consider edges
        # Note: adj is symmetric
        max_book = 0
        # Iterate over edges to find max
        # Using numpy where
        # We can just take the max of common where adj is 1
        # But diagonal is degree, not book size (though usually not an edge)
        # We need to ignore diagonal and non-edges
        # Since adj is 0/1, common[i,j] is count of common neighbors.
        # We want max common[i,j] for adj[i,j]==1.
        # To avoid diagonal, we can set diagonal to 0 before max
        # Or just iterate, n=40 is small
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    if common[i, j] > max_book:
                        max_book = common[i, j]
        return max_book

    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                edges.add((u, v))
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj, list(edges)

    def bipartite_plus_k(k):
        # K_20,20 has 400 edges. Add k edges within partition.
        # This creates large books, but is a structured start.
        adj = np.zeros((n, n), dtype=np.uint8)
        # Partition 0..19 and 20..39
        for i in range(20):
            for j in range(20, 40):
                adj[i, j] = 1
                adj[j, i] = 1
        edges = [(i, j) for i in range(20) for j in range(20, 40)]
        
        # Add k edges within first partition
        intra_edges = set()
        while len(intra_edges) < k:
            u = random.randint(0, 19)
            v = random.randint(0, 19)
            if u != v:
                if u > v: u, v = v, u
                if (u, v) not in intra_edges:
                    intra_edges.add((u, v))
        for u, v in intra_edges:
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append((u, v))
        return adj, edges

    best_score = 100
    best_adj = None
    best_edges = None

    # Run multiple SA instances
    num_runs = 5
    time_per_run = 10.0

    for run in range(num_runs):
        # Choose start
        if run == 0:
            current_adj, current_edges = bipartite_plus_k(1)
        else:
            current_adj, current_edges = random_graph()
            
        current_score = get_max_book(current_adj)
        
        if current_score < best_score:
            best_score = current_score
            best_adj = current_adj.copy()
            best_edges = current_edges[:]
            if best_score == 7:
                # Can't do better than 7
                return [[u, v] for u, v in best_edges]

        # Simulated Annealing
        T = 5.0
        cooling_rate = 0.99
        run_deadline = time.time() + time_per_run
        
        while time.time() < run_deadline and time.time() < deadline:
            # Pick random edge to remove
            out_u, out_v = random.choice(current_edges)
            
            # Pick random non-edge to add
            while True:
                in_u = random.randint(0, n-1)
                in_v = random.randint(0, n-1)
                if in_u == in_v: continue
                if in_u > in_v: in_u, in_v = in_v, in_u
                if (in_u, in_v) == (out_u, out_v): continue
                if current_adj[in_u, in_v] == 0:
                    break

            # Apply swap
            current_adj[out_u, out_v] = 0
            current_adj[out_v, out_u] = 0
            current_edges.remove((out_u, out_v))
            
            current_adj[in_u, in_v] = 1
            current_adj[in_v, in_u] = 1
            current_edges.append((in_u, in_v))

            new_score = get_max_book(current_adj)
            delta = new_score - current_score
            
            if delta < 0:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj.copy()
                    best_edges = current_edges[:]
                    if best_score == 7:
                        return [[u, v] for u, v in best_edges]
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_adj[in_u, in_v] = 0
                    current_adj[in_v, in_u] = 0
                    current_edges.remove((in_u, in_v))
                    
                    current_adj[out_u, out_v] = 1
                    current_adj[out_v, out_u] = 1
                    current_edges.append((out_u, out_v))
            
            T *= cooling_rate
            if T < 0.1:
                T = 0.1

    return [[u, v] for u, v in best_edges]
