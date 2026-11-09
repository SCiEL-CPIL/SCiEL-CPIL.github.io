import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 148
    
    n = 16
    adj = [[False]*n for _ in range(n)]
    edge_set = set()
    
    def add_edge(u, v):
        u, v = min(u,v), max(u,v)
        if u != v and (u,v) not in edge_set:
            edge_set.add((u,v))
            adj[u][v] = adj[v][u] = True
    
    def remove_edge(u, v):
        u, v = min(u,v), max(u,v)
        if (u,v) in edge_set:
            edge_set.remove((u,v))
            adj[u][v] = adj[v][u] = False
    
    def compute_score():
        deg = [sum(adj[i]) for i in range(n)]
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def compute_penalty_score():
        """Penalty-based objective: sum of squares of triangle degree sums, plus a large penalty for max > threshold"""
        deg = [sum(adj[i]) for i in range(n)]
        tri_sums = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            tri_sums.append(deg[i] + deg[j] + deg[k])
        if not tri_sums:
            return 0.0
        max_sum = max(tri_sums)
        # Penalty: square of max + sum of squares of all triangle sums (to flatten distribution)
        sum_sq = sum(s*s for s in tri_sums)
        # Heavy penalty for high max values
        penalty = max_sum * max_sum * 10 + sum_sq
        return penalty
    
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    best_score = float('inf')
    best_edge_set = None
    best_penalty = float('inf')
    
    def try_local_search(init_edge_set, max_iterations=50000, use_penalty=True):
        nonlocal best_score, best_edge_set, best_penalty
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
        for e in init_edge_set:
            add_edge(*e)
        
        current_score = compute_score()
        current_penalty = compute_penalty_score() if use_penalty else current_score
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
        
        if use_penalty and current_penalty < best_penalty:
            best_penalty = current_penalty
        
        no_improve = 0
        for it in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try all possible swaps
            for e_remove in list(edge_set):
                if time.time() > deadline - 3:
                    break
                for e_add in all_edges:
                    if e_add in edge_set or e_add == e_remove:
                        continue
                    
                    remove_edge(*e_remove)
                    add_edge(*e_add)
                    
                    if use_penalty:
                        new_penalty = compute_penalty_score()
                        if new_penalty < current_penalty:
                            current_penalty = new_penalty
                            improved = True
                            new_score = compute_score()
                            if new_score < best_score:
                                best_score = new_score
                                best_edge_set = set(edge_set)
                            if new_penalty < best_penalty:
                                best_penalty = new_penalty
                            break
                        else:
                            remove_edge(*e_add)
                            add_edge(*e_remove)
                    else:
                        new_score = compute_score()
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_edge_set = set(edge_set)
                            break
                        else:
                            remove_edge(*e_add)
                            add_edge(*e_remove)
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > 50:
                    break
    
    # Spectral heuristic for initial graph construction
    def spectral_init():
        """Use Laplacian eigenvector to balance degrees, then select top 65 edges by weight"""
        # Create a random perturbation to break symmetry
        np.random.seed(42)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                A[i,j] = A[j,i] = 1 + np.random.randn() * 0.1
        
        # Compute Laplacian
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Get eigenvectors
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Use the second eigenvector (Fiedler vector) to partition
        # Or use a combination to create edge weights
        # Create edge weights based on a smooth function of vertex potentials
        # Use the top eigenvectors to create a potential
        potentials = np.abs(eigvecs[:, 1])  # Fiedler vector
        
        # Assign edge weights based on potentials: prefer edges between similar potential vertices
        # to balance degrees
        edge_weights = {}
        for i in range(n):
            for j in range(i+1, n):
                # Weight based on potential difference - prefer edges that balance
                w = 1.0 - abs(potentials[i] - potentials[j]) / (max(potentials) - min(potentials) + 1e-10)
                edge_weights[(i,j)] = w
        
        # Sort edges by weight and take top 65
        sorted_edges = sorted(edge_weights.items(), key=lambda x: -x[1])
        return set(e for e, w in sorted_edges[:65])
    
    # Initial construction: K_{8,8} + one edge (baseline)
    base_edges = set()
    for u in range(8):
        for v in range(8, 16):
            base_edges.add((u, v))
    
    # Try different initial intra-part edges
    for part in [0, 1]:
        for u in range(8):
            for v in range(u+1, 8):
                if part == 0:
                    edge = (u, v)
                else:
                    edge = (u+8, v+8)
                
                init_edges = base_edges | {edge}
                try_local_search(init_edges, max_iterations=20000, use_penalty=True)
                
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Also try spectral initialization
    if time.time() < deadline - 20:
        spec_init = spectral_init()
        try_local_search(spec_init, max_iterations=30000, use_penalty=True)
    
    # Final polish pass: pure score-based local search from best found
    if best_edge_set is not None and time.time() < deadline - 15:
        try_local_search(best_edge_set, max_iterations=20000, use_penalty=False)
    
    if best_edge_set is None:
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        add_edge(0, 1)
        best_edge_set = set(edge_set)
    
    edge_list = [[u, v] for (u, v) in best_edge_set]
    return {'n': n, 'edges': edge_list}
