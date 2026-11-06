"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def build_adj(edges):
        """Build adjacency bitmasks and degrees from edge list."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score_from_adj(adj, deg):
        """Compute max triangle degree sum using bitmasks."""
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                # Common neighbors of i and j, with index > j
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                    common &= common - 1
        return max_sum
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        adj, deg = build_adj(edges)
        return compute_score_from_adj(adj, deg)
    
    def is_valid(edges):
        """Check if edges form a simple graph with exactly target_edges edges."""
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True
    
    # Strategy: K_{6,10} + 5 edges in the 10-vertex part
    # Vertices 0-5 are the "small" part, 6-15 are the "large" part
    # Base degrees: small=10, large=6
    # Adding 5 internal edges: each endpoint gets +1 degree
    # Triangle score = 10 + deg(i) + deg(j) for internal edge (i,j)
    # To get score <= 23, need deg(i) + deg(j) <= 13 for all internal edges
    # i.e., (6+a_i) + (6+a_j) <= 13 => a_i + a_j <= 1
    # where a_v = number of internal edges incident to v
    
    # Start with K_{6,10} + matching (score 24)
    base_edges_k610 = [[u, v] for u in range(6) for v in range(6, 16)]
    initial_internal = [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    best_edges = base_edges_k610 + initial_internal
    best_score = compute_score(best_edges)
    
    # Also try K_{8,8} + 1 edge
    k88_edges = [[u, v] for u in range(8) for v in range(8, 16)]
    k88_plus1 = k88_edges + [[0, 1]]
    k88_score = compute_score(k88_plus1)
    if k88_score < best_score:
        best_score = k88_score
        best_edges = k88_plus1[:]
    
    # Simulated annealing on edge swaps
    def simulated_annealing(edges, max_iters=50000, T_initial=5.0, T_final=0.1, cooling=0.9999):
        """Simulated annealing to escape local minima."""
        current = edges[:]
        adj, deg = build_adj(current)
        current_score = compute_score_from_adj(adj, deg)
        
        edge_set = set()
        for u, v in current:
            edge_set.add((min(u, v), max(u, v)))
        
        all_edges = set()
        for i in range(n):
            for j in range(i+1, n):
                all_edges.add((i, j))
        
        best_sa = current_score
        best_sa_edges = current[:]
        T = T_initial
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick random edge to remove
            u1, v1 = random.choice(list(edge_set))
            
            # Pick random non-edge to add
            non_edges = list(all_edges - edge_set)
            if not non_edges:
                break
            u2, v2 = random.choice(non_edges)
            
            # Perform swap
            new_adj = adj[:]
            new_deg = deg[:]
            
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            new_deg[u2] += 1
            new_deg[v2] += 1
            
            new_score = compute_score_from_adj(new_adj, new_deg)
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                adj = new_adj
                deg = new_deg
                current_score = new_score
                edge_set.remove((u1, v1))
                edge_set.add((u2, v2))
                
                if current_score < best_sa:
                    best_sa = current_score
                    best_sa_edges = [[i, j] for i, j in edge_set]
            
            T *= cooling
        
        return best_sa_edges, best_sa
    
    # Run simulated annealing from best starting point
    sa_edges, sa_score = simulated_annealing(best_edges, max_iters=30000)
    if sa_score < best_score:
        best_score = sa_score
        best_edges = sa_edges[:]
    
    # Also run SA from K_{6,10}+matching specifically
    sa_edges2, sa_score2 = simulated_annealing(base_edges_k610 + initial_internal, max_iters=20000)
    if sa_score2 < best_score:
        best_score = sa_score2
        best_edges = sa_edges2[:]
    
    # Final deterministic local search polish
    def local_search_polish(edges, max_iters=10000):
        """Final greedy local search."""
        current = edges[:]
        adj, deg = build_adj(current)
        current_score = compute_score_from_adj(adj, deg)
        
        edge_set = set()
        for u, v in current:
            edge_set.add((min(u, v), max(u, v)))
        
        all_edges = set()
        for i in range(n):
            for j in range(i+1, n):
                all_edges.add((i, j))
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            for attempt in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                u1, v1 = random.choice(list(edge_set))
                non_edges = list(all_edges - edge_set)
                if not non_edges:
                    break
                u2, v2 = random.choice(non_edges)
                
                new_adj = adj[:]
                new_deg = deg[:]
                
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                new_deg[u1] -= 1
                new_deg[v1] -= 1
                
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                new_deg[u2] += 1
                new_deg[v2] += 1
                
                new_score = compute_score_from_adj(new_adj, new_deg)
                
                if new_score < current_score:
                    adj = new_adj
                    deg = new_deg
                    current_score = new_score
                    edge_set.remove((u1, v1))
                    edge_set.add((u2, v2))
                    current = [[i, j] for i, j in edge_set]
                    improved = True
                    break
        
        return current, current_score
    
    polish_edges, polish_score = local_search_polish(best_edges, max_iters=8000)
    if polish_score < best_score:
        best_score = polish_score
        best_edges = polish_edges[:]
    
    # Final verification
    if not is_valid(best_edges):
        best_edges = base_edges_k610 + [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
