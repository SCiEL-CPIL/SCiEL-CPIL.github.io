import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_valid(edges):
    """Check if no 3-edge path has equal-degree endpoints."""
    n = 15
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u > v:
                continue
            # For edge (u,v), check all s in adj[u]\{v} and t in adj[v]\{u}
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u:
                        continue
                    if s == t:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def construct():
    """
    Start with K_{7,8} (56 edges) and try to add intra-part edges
    while maintaining validity. Uses a deadline guard.
    """
    deadline = time.time() + 75.0  # safety margin
    
    n = 15
    A = list(range(0, 7))
    B = list(range(7, 15))
    
    # Base: complete bipartite K_{7,8}
    base_edges = []
    for u in A:
        for v in B:
            base_edges.append([u, v])
    
    # Try adding intra-A edges
    intra_a_pairs = list(combinations(A, 2))
    # Try adding intra-B edges
    intra_b_pairs = list(combinations(B, 2))
    
    best_edges = base_edges
    best_count = len(base_edges)
    
    # Greedy: try to add intra-part edges one at a time
    # First try all intra-A, then intra-B, then combinations
    candidates_a = list(intra_a_pairs)
    candidates_b = list(intra_b_pairs)
    
    # Try adding one intra-A edge at a time
    for pair in candidates_a:
        if time.time() > deadline:
            break
        test_edges = base_edges + [[pair[0], pair[1]]]
        if _is_valid(test_edges):
            if len(test_edges) > best_count:
                best_edges = test_edges
                best_count = len(test_edges)
    
    # Now try adding intra-B edges to the current best
    current = list(best_edges)
    for pair in candidates_b:
        if time.time() > deadline:
            break
        # Check if pair already in current
        u, v = pair
        if u in [e[0] for e in current if e[1] == v] or v in [e[0] for e in current if e[1] == u]:
            continue
        test_edges = current + [[u, v]]
        if _is_valid(test_edges):
            if len(test_edges) > best_count:
                current = test_edges
                best_edges = test_edges
                best_count = len(test_edges)
    
    # Try adding more intra-A edges to current best
    for pair in candidates_a:
        if time.time() > deadline:
            break
        u, v = pair
        if u in [e[0] for e in current if e[1] == v] or v in [e[0] for e in current if e[1] == u]:
            continue
        test_edges = current + [[u, v]]
        if _is_valid(test_edges):
            if len(test_edges) > best_count:
                current = test_edges
                best_edges = test_edges
                best_count = len(test_edges)
    
    # Try adding more intra-B edges
    for pair in candidates_b:
        if time.time() > deadline:
            break
        u, v = pair
        if u in [e[0] for e in current if e[1] == v] or v in [e[0] for e in current if e[1] == u]:
            continue
        test_edges = current + [[u, v]]
        if _is_valid(test_edges):
            if len(test_edges) > best_count:
                current = test_edges
                best_edges = test_edges
                best_count = len(test_edges)
    
    # Final pass: try all remaining intra-part edges
    for pair in candidates_a + candidates_b:
        if time.time() > deadline:
            break
        u, v = pair
        if u in [e[0] for e in current if e[1] == v] or v in [e[0] for e in current if e[1] == u]:
            continue
        test_edges = current + [[u, v]]
        if _is_valid(test_edges):
            if len(test_edges) > best_count:
                current = test_edges
                best_edges = test_edges
                best_count = len(test_edges)
    
    return best_edges
# EVOLVE-BLOCK-END
