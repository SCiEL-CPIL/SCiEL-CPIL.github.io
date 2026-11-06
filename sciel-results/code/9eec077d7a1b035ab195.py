# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n = 31
    m = 91  # number of edges required
    q = 5
    
    # Build PG(2,5) points as 1D subspaces of GF(5)^3
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], q-2, q)
                vec = [(x * inv) % q for x in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    if len(points) != 31:
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        return [[i, j] for i, j in all_possible[:m]]
    
    def perp(p, qpt):
        return (p[0]*qpt[0] + p[1]*qpt[1] + p[2]*qpt[2]) % q == 0
    
    # Build adjacency for the full polarity graph
    adj = [[False]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if perp(points[i], points[j]):
                adj[i][j] = adj[j][i] = True
                edges.append((i, j))
    
    polarity_edges = len(edges)
    
    if polarity_edges < m:
        random.seed(42)
        existing = set(edges)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for e in all_possible:
            if e not in existing:
                edges.append(e)
                if len(edges) == m:
                    break
        return [[u, v] for u, v in edges]
    
    # Helper to count C4s in the current graph
    def count_c4(adj_mat):
        c4_count = 0
        for u in range(n):
            for v in range(u+1, n):
                common = []
                for w in range(n):
                    if w != u and w != v and adj_mat[u][w] and adj_mat[v][w]:
                        common.append(w)
                for i in range(len(common)):
                    for j in range(i+1, len(common)):
                        w = common[i]
                        x = common[j]
                        if adj_mat[w][x]:
                            c4_count += 1
        return c4_count
    
    # Helper to compute the number of C4s an edge (u,v) belongs to
    def count_c4s_for_edge(adj_mat, u, v):
        count = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj_mat[u][w]:
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if adj_mat[v][x] and adj_mat[w][x]:
                        count += 1
        return count

    # Helper to compute the number of C4s created by adding edge (u,v)
    def count_c4s_created_by_adding(adj_mat, u, v):
        count = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj_mat[v][w]:
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if adj_mat[u][x] and adj_mat[w][x]:
                        count += 1
        return count

    # Initialize graph: Remove 2 edges from polarity graph to get 91 edges
    current_adj = [row[:] for row in adj]
    current_edges = set(edges)
    
    # Remove 2 edges
    to_remove = polarity_edges - m
    for _ in range(to_remove):
        best_edge = None
        best_count = -1
        for e in current_edges:
            u, v = e
            count = count_c4s_for_edge(current_adj, u, v)
            if count > best_count:
                best_count = count
                best_edge = e
        if best_edge:
            current_edges.remove(best_edge)
            u, v = best_edge
            current_adj[u][v] = current_adj[v][u] = False
    
    current_c4 = count_c4(current_adj)
    best_adj = [row[:] for row in current_adj]
    best_edges = set(current_edges)
    best_c4 = current_c4
    
    # List of all possible non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Simulated annealing
    max_iterations = 5000
    iter_count = 0
    T = 5.0  # initial temperature
    cooling = 0.99
    
    while time.time() < deadline and iter_count < max_iterations:
        iter_count += 1
        
        # Calculate C4 counts for all current edges
        edge_c4_counts = {}
        for e in current_edges:
            u, v = e
            edge_c4_counts[e] = count_c4s_for_edge(current_adj, u, v)
        
        # Sort edges by C4 count (descending) to pick bad ones
        sorted_edges = sorted(edge_c4_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate C4 creation counts for all non-edges
        non_edges = [e for e in all_pairs if e not in current_edges]
        non_edge_c4_counts = {}
        for e in non_edges:
            u, v = e
            non_edge_c4_counts[e] = count_c4s_created_by_adding(current_adj, u, v)
        
        # Sort non-edges by C4 creation count (ascending) to pick good ones
        sorted_non_edges = sorted(non_edge_c4_counts.items(), key=lambda x: x[1])
        
        # Try a few swaps
        K = 10
        best_swap = None
        best_c4_after = current_c4
        
        for e_out, c_out in sorted_edges[:K]:
            for e_in, c_in in sorted_non_edges[:K]:
                u_out, v_out = e_out
                u_in, v_in = e_in
                
                current_adj[u_out][v_out] = False
                current_adj[v_out][u_out] = False
                current_adj[u_in][v_in] = True
                current_adj[v_in][u_in] = True
                
                new_c4 = count_c4(current_adj)
                
                # Revert
                current_adj[u_out][v_out] = True
                current_adj[v_out][u_out] = True
                current_adj[u_in][v_in] = False
                current_adj[v_in][u_in] = False
                
                if new_c4 < best_c4_after:
                    best_c4_after = new_c4
                    best_swap = (e_out, e_in)
        
        if best_swap:
            e_out, e_in = best_swap
            u_out, v_out = e_out
            u_in, v_in = e_in
            
            # Apply swap permanently
            current_edges.remove(e_out)
            current_edges.add(e_in)
            current_adj[u_out][v_out] = False
            current_adj[v_out][u_out] = False
            current_adj[u_in][v_in] = True
            current_adj[v_in][u_in] = True
            
            current_c4 = best_c4_after
            
            # Update best
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = [row[:] for row in current_adj]
                best_edges = set(current_edges)
        else:
            # Try a random swap to escape local minimum
            if random.random() < 0.5:
                random_edge = random.choice(list(current_edges))
                random_non_edge = random.choice(non_edges)
                u_out, v_out = random_edge
                u_in, v_in = random_non_edge
                
                current_adj[u_out][v_out] = False
                current_adj[v_out][u_out] = False
                current_adj[u_in][v_in] = True
                current_adj[v_in][u_in] = True
                
                new_c4 = count_c4(current_adj)
                delta = new_c4 - current_c4
                
                # Accept if better or with probability exp(-delta/T)
                if delta < 0 or random.random() < np.exp(-delta/T):
                    current_edges.remove(random_edge)
                    current_edges.add(random_non_edge)
                    current_c4 = new_c4
                    
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_adj = [row[:] for row in current_adj]
                        best_edges = set(current_edges)
                else:
                    # Revert
                    current_adj[u_out][v_out] = True
                    current_adj[v_out][u_out] = True
                    current_adj[u_in][v_in] = False
                    current_adj[v_in][u_in] = False
            
            T *= cooling
            if T < 0.1:
                break
    
    final_edges = list(best_edges)
    
    # Verify
    if len(final_edges) != m:
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        final_edges = all_possible[:m]
    
    result = [[u, v] for u, v in final_edges]
    return result

# EVOLVE-BLOCK-END
