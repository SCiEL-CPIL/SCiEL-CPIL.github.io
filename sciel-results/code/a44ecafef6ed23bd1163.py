import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    n = 20
    
    # Precompute all triples of vertices for K3,3 check
    triples = list(combinations(range(n), 3))
    triple_indices = np.array(triples)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_idx_to_uv = {i: e for i, e in enumerate(all_edges)}
    uv_to_edge_idx = {e: i for i, e in enumerate(all_edges)}
    
    def count_violations(adj):
        """
        Vectorized check for K3,3 subgraphs.
        A K3,3 exists if there are 3 vertices with 3 common neighbors.
        For each triple (a,b,c), count common neighbors. If count >= 3, violation.
        We sum the number of triples with >= 3 common neighbors.
        """
        # adj is n x n boolean matrix
        # For each triple, we want the bitwise AND of their adjacency rows (excluding self, but diag is 0)
        # Then sum the bits.
        
        # Extract rows for the triples
        # triple_indices shape: (T, 3)
        # We want for each triple t, the vector v_t = adj[a_t] & adj[b_t] & adj[c_t]
        # This is hard to vectorize directly without loops in pure numpy without advanced indexing tricks.
        # However, n=20 is small. T = 1140.
        # Let's use a loop over triples but vectorize the inner count?
        # Or better:
        # For each vertex m, let S_m be the set of vertices adjacent to m.
        # A K3,3 centered at m? No, K3,3 is bipartite.
        
        # Alternative vectorized approach:
        # Let A be adjacency matrix.
        # A^3 diagonal entries? No.
        
        # Let's stick to the triple loop but make it fast.
        # Actually, we can precompute for each pair (u,v) the common neighbors as a bitmask or list.
        # But adj changes.
        
        # Fastest NumPy way for n=20:
        # For each triple (i, j, k), common = sum(adj[i] & adj[j] & adj[k])
        # We can do this by iterating triples. 1140 iterations is fast in Python if the inner op is fast.
        # adj[i] & adj[j] & adj[k] is 3 vector ops of size 20.
        
        violations = 0
        for i, j, k in triples:
            # Count common neighbors
            # Exclude i, j, k themselves? 
            # adj[i][i] is 0, so it's fine.
            common = np.sum(adj[i] & adj[j] & adj[k])
            if common >= 3:
                violations += 1
        return violations
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # Initialize with a random dense graph
    # p = 0.7 is a good start for dense graphs
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.7:
                adj[i][j] = True
                adj[j][i] = True
    
    current_violations = count_violations(adj)
    current_edges = np.sum(adj) // 2
    
    best_adj = adj.copy()
    best_edges = current_edges
    best_violations = current_violations
    
    # Simulated Annealing
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999  # Cooling factor
    T = T_start
    
    steps = 0
    while time.time() < deadline - 5 and T > T_end:
        # Pick a random edge to flip
        u, v = random.choice(all_edges)
        
        # Flip
        adj[u][v] = not adj[u][v]
        adj[v][u] = not adj[v][u]
        
        new_violations = count_violations(adj)
        new_edges = np.sum(adj) // 2
        
        # Energy function: We want to maximize edges, subject to violations == 0.
        # Let's define energy E = -edges + 100 * violations
        # We want to minimize E.
        E_old = -current_edges + 100 * current_violations
        E_new = -new_edges + 100 * new_violations
        
        delta_E = E_new - E_old
        
        # Accept if better, or with probability exp(-delta_E / T)
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            # Accept
            current_violations = new_violations
            current_edges = new_edges
            
            # Update best if valid
            if current_violations == 0 and current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
                best_violations = 0
            elif current_violations < best_violations:
                # Keep track of best invalid too? No, we only care about valid.
                # But maybe track best invalid to guide search? 
                # For now, only update best if valid.
                pass
        else:
            # Reject
            adj[u][v] = not adj[u][v]
            adj[v][u] = not adj[v][u]
            
        T *= alpha
        steps += 1
        
        # If we have a valid graph and T is low, we can tighten constraints?
        # Or just let it run.
        
    # Final local search on the best valid graph found
    if best_violations == 0:
        # Try to add more edges to the best graph
        # Greedy add
        for u, v in all_edges:
            if not best_adj[u][v]:
                best_adj[u][v] = True
                best_adj[v][u] = True
                if count_violations(best_adj) == 0:
                    pass # Keep
                else:
                    best_adj[u][v] = False
                    best_adj[v][u] = False
                    
        # Greedy remove (if we ended up invalid, but we should be valid)
        # Also try swaps
        # For simplicity, just return the best valid found during SA or improved by greedy
        
    # Ensure we return a valid graph
    # If best_violations != 0, we need to find a valid one.
    # Fallback: use a known valid graph or random restarts
    if best_violations != 0:
        # Try a few random restarts with SA
        for restart in range(5):
            if time.time() > deadline - 10: break
            
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.6:
                        adj[i][j] = True
                        adj[j][i] = True
            
            current_violations = count_violations(adj)
            current_edges = np.sum(adj) // 2
            
            T = T_start
            while time.time() < deadline - 5 and T > T_end:
                u, v = random.choice(all_edges)
                adj[u][v] = not adj[u][v]
                adj[v][u] = not adj[v][u]
                
                new_violations = count_violations(adj)
                new_edges = np.sum(adj) // 2
                
                E_old = -current_edges + 100 * current_violations
                E_new = -new_edges + 100 * new_violations
                delta_E = E_new - E_old
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    current_violations = new_violations
                    current_edges = new_edges
                    if current_violations == 0 and current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj.copy()
                        best_violations = 0
                else:
                    adj[u][v] = not adj[u][v]
                    adj[v][u] = not adj[v][u]
                    
                T *= alpha
            
            if best_violations == 0:
                break
    
    # If still invalid, return empty or a known valid graph
    if best_violations != 0:
        # Fallback to K2,18
        best_adj = np.zeros((n, n), dtype=bool)
        for a in range(2):
            for b in range(2, 20):
                best_adj[a][b] = True
                best_adj[b][a] = True
        best_edges = np.sum(best_adj) // 2

    return get_edges(best_adj)

# EVOLVE-BLOCK-END
