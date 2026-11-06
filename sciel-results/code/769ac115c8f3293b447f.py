import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n = 16
    target_edges = 65
    
    def score_graph(edges):
        """Compute max degree sum over triangles."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(adj[i]) for i in range(n)]
        
        max_sum = 0
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    for k in adj[i] & adj[j]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    # Start with K_{8,8} + edge (0,1)
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    best_edges = list(edge_set)
    best_score = score_graph(best_edges)
    
    # Local search: try edge swaps
    def get_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_triangles(edge_set):
        """Return list of (triangle, degree_sum) for all triangles."""
        adj = get_adj(edge_set)
        deg = [len(adj[i]) for i in range(n)]
        triangles = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    common = adj[i] & adj[j]
                    for k in common:
                        s = deg[i] + deg[j] + deg[k]
                        triangles.append((frozenset([i,j,k]), s))
        return triangles
    
    # Local search with swaps
    all_possible_edges = set((i, j) for i in range(n) for j in range(i+1, n))
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Try random swaps
        current_edge = random.choice(best_edges)
        possible_removals = [e for e in best_edges]
        possible_additions = [e for e in all_possible_edges if e not in edge_set]
        
        if not possible_additions:
            break
            
        new_edge = random.choice(possible_additions)
        
        # Try the swap
        new_edge_set = set(best_edges)
        new_edge_set.discard(current_edge)
        new_edge_set.add(new_edge)
        
        if len(new_edge_set) != target_edges:
            continue
        
        new_score = score_graph(list(new_edge_set))
        
        if new_score < best_score:
            best_score = new_score
            best_edges = list(new_edge_set)
            edge_set = new_edge_set
        elif new_score == best_score and random.random() < 0.3:
            # Accept equal score to escape local optima
            best_edges = list(new_edge_set)
            edge_set = new_edge_set
    
    # Final polish: try all single-edge swaps from current best
    if time.time() < deadline - 10:
        improved = True
        while improved:
            improved = False
            for current_edge in best_edges:
                if time.time() > deadline:
                    break
                for new_edge in all_possible_edges:
                    if new_edge in edge_set:
                        continue
                    new_edge_set = set(best_edges)
                    new_edge_set.discard(current_edge)
                    new_edge_set.add(new_edge)
                    if len(new_edge_set) != target_edges:
                        continue
                    new_score = score_graph(list(new_edge_set))
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = list(new_edge_set)
                        edge_set = new_edge_set
                        improved = True
    
    # Format edges as [[u,v], ...]
    result_edges = [[min(u,v), max(u,v)] for u, v in best_edges]
    return {'n': n, 'edges': result_edges}
