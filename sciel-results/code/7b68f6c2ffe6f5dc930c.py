import time
import random
import math

def _check_validity(edges, n=15):
    """Check if the graph satisfies the constraint."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    
    # For each edge (u,v), check all pairs (s,t) where s in N(u)\{v}, t in N(v)\{u}, s!=t
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def _count_edges(edges):
    return len(edges)

def construct():
    n = 15
    deadline = time.time() + 85  # safety margin before 90s
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Start with a greedy construction
    # Try to build a graph with high edge count
    
    best_edges = []
    
    # Greedy: try to add edges in order of "least constraint violation"
    # Start with a near-complete graph and remove violating edges
    
    # Strategy: Start with complete graph, remove edges to fix violations
    edges = list(all_edges)
    
    # Iteratively remove edges that cause violations
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find a violation
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(adj[i]) for i in range(n)]
        
        violated_edge = None
        for idx, (u, v) in enumerate(edges):
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        violated_edge = idx
                        break
                if violated_edge is not None:
                    break
            if violated_edge is not None:
                break
        
        if violated_edge is not None:
            edges.pop(violated_edge)
            changed = True
    
    best_edges = list(edges)
    best_count = len(best_edges)
    
    # Try local search: swap edges to improve
    # Also try adding back removed edges
    
    # Local search with simulated annealing-like approach
    current = list(best_edges)
    current_count = best_count
    
    # Get the set of edges for fast lookup
    edge_set = set((min(u,v), max(u,v)) for u, v in current)
    
    removed_edges = [e for e in all_edges if (min(e[0],e[1]), max(e[0],e[1])) not in edge_set]
    
    # Try adding back edges
    for e in removed_edges:
        if time.time() > deadline:
            break
        test = current + [e]
        if _check_validity(test, n):
            current = test
            edge_set.add((min(e[0],e[1]), max(e[0],e[1])))
            current_count += 1
    
    if current_count > best_count:
        best_edges = list(current)
        best_count = current_count
    
    # Try swapping: remove one edge, add another
    for i in range(len(current)):
        if time.time() > deadline:
            break
        for j in range(len(removed_edges)):
            if time.time() > deadline:
                break
            test = current[:i] + current[i+1:] + [removed_edges[j]]
            if _check_validity(test, n):
                current = test
                removed_edges[j] = current[i]
                edge_set.discard((min(current[i][0],current[i][1]), max(current[i][0],current[i][1])))
                edge_set.add((min(removed_edges[j][0],removed_edges[j][1]), max(removed_edges[j][0],removed_edges[j][1])))
                # Rebuild removed_edges
                removed_edges = [e for e in all_edges if (min(e[0],e[1]), max(e[0],e[1])) not in edge_set]
                break
    
    if len(current) > best_count:
        best_edges = list(current)
        best_count = len(current)
    
    # Return as list of [u,v] with int types
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
