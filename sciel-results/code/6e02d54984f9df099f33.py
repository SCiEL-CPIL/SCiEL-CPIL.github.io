"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations

P = {'n': 15}

def construct():
    n = P['n']
    
    def check_valid(adj, deg):
        """Check if graph satisfies: for all 3-edge paths s-u-v-t, deg(s) != deg(t)."""
        for u in range(n):
            for v in adj[u]:
                if u >= v:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def backtrack(adj, deg, edge_idx, current_edges, best):
        """Backtrack over edges, trying to add each edge if valid."""
        if edge_idx == len(all_edges):
            if len(current_edges) > best[0]:
                best[0] = len(current_edges)
                best[1] = list(current_edges)
            return
        
        u, v = all_edges[edge_idx]
        
        # Option 1: Don't add edge (u, v)
        backtrack(adj, deg, edge_idx + 1, current_edges, best)
        
        # Option 2: Add edge (u, v) if it doesn't violate constraint
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        
        if check_valid(adj, deg):
            current_edges.append((u, v))
            backtrack(adj, deg, edge_idx + 1, current_edges, best)
            current_edges.pop()
        
        # Undo
        adj[u].remove(v)
        adj[v].remove(u)
        deg[u] -= 1
        deg[v] -= 1
    
    # Generate all possible edges
    all_edges = list(combinations(range(n), 2))
    
    # Use pruning: sort edges to try promising ones first
    # Heuristic: try edges between low-degree vertices first to allow more edges later
    # Actually, let's try a different order
    
    adj = [set() for _ in range(n)]
    deg = [0] * n
    best = [0, []]
    
    # Full backtracking is too slow for 105 edges. 
    # Instead, use a greedy approach with multiple restarts and local improvement.
    
    import random
    
    def greedy_build(seed_val):
        rnd = random.Random(seed_val)
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        all_e = list(combinations(range(n), 2))
        rnd.shuffle(all_e)
        
        for u, v in all_e:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            
            if check_valid(adj, deg):
                edges.append((u, v))
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
        
        # Local improvement: try adding any missing edge
        edge_set = set(edges)
        improved = True
        while improved:
            improved = False
            for u, v in combinations(range(n), 2):
                if (u, v) in edge_set:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                if check_valid(adj, deg):
                    edges.append((u, v))
                    edge_set.add((u, v))
                    improved = True
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    deg[u] -= 1
                    deg[v] -= 1
        
        return edges, adj, deg
    
    best_edges = []
    best_count = 0
    
    for seed_val in range(500):
        edges, adj, deg = greedy_build(seed_val)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            if best_count >= 65:
                break
    
    # Try more seeds if needed
    if best_count < 65:
        for seed_val in range(500, 1500):
            edges, adj, deg = greedy_build(seed_val)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
                if best_count >= 65:
                    break
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-END
