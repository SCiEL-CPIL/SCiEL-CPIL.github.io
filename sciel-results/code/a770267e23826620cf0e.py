# EVOLVE-BLOCK-START
import sys
from itertools import combinations
import random

P = {'n': 15}

def construct():
    n = P['n']
    all_edges = list(combinations(range(n), 2))
    
    def get_violations(adj, deg):
        """Return list of violating paths (s, u, v, t) and count."""
        violations = []
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
                            violations.append((s, u, v, t))
        return violations

    def local_search(seed_val):
        rnd = random.Random(seed_val)
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = set()
        
        # Start with random dense graph
        # Probability of edge 0.8
        for u, v in all_edges:
            if rnd.random() < 0.8:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.add((u, v))
        
        best_edges = set(edges)
        best_count = len(edges)
        
        for _ in range(1000): # Iterations
            violations = get_violations(adj, deg)
            if not violations:
                # Try to add edges
                improved = True
                while improved:
                    improved = False
                    missing = [e for e in all_edges if e not in edges]
                    rnd.shuffle(missing)
                    for u, v in missing:
                        adj[u].add(v)
                        adj[v].add(u)
                        deg[u] += 1
                        deg[v] += 1
                        if not get_violations(adj, deg):
                            edges.add((u, v))
                            improved = True
                        else:
                            adj[u].remove(v)
                            adj[v].remove(u)
                            deg[u] -= 1
                            deg[v] -= 1
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = set(edges)
                # Break if max edges?
                if best_count == len(all_edges):
                    break
            
            # Remove an edge from a random violation
            if violations:
                s, u, v, t = rnd.choice(violations)
                # Remove one of the 3 edges: (s,u), (u,v), (v,t)
                # Randomly pick one
                edge_to_remove = rnd.choice([(s, u), (u, v), (v, t)])
                a, b = edge_to_remove
                if a > b: a, b = b, a
                if (a, b) in edges:
                    adj[a].remove(b)
                    adj[b].remove(a)
                    deg[a] -= 1
                    deg[b] -= 1
                    edges.remove((a, b))
        
        return best_edges

    best_edges = set()
    best_count = 0
    
    for seed_val in range(100):
        edges = local_search(seed_val)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            if best_count >= 100:
                break

    result = [[int(u), int(v)] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-END
