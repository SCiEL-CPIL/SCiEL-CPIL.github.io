import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    n = 15
    # Start with K_{7,8}: partition A = {0..6}, B = {7..14}
    # deg(A) = 8, deg(B) = 7
    # All 3-edge paths have endpoints in opposite partitions → degrees 8 vs 7 → OK
    
    best_edges = []
    best_count = 0
    
    # Build K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
    
    best_edges = [e[:] for e in edges]
    best_count = len(edges)  # 56
    
    # Try to add intra-partition edges and check validity
    # For a candidate graph, check: no simple 3-edge path has endpoints of equal degree
    
    def is_valid(edge_list):
        """Check if the graph satisfies the constraint."""
        n = 15
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        deg = [len(adj[i]) for i in range(n)]
        
        # For every simple 3-edge path s-u-v-t, check deg(s) != deg(t)
        # s-u-v-t means: s~u, u~v, v~t, all distinct
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue  # avoid double counting edges
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    # Try adding edges within A and within B
    # Intra-A edges: between vertices in A
    # Intra-B edges: between vertices in B
    
    intra_A = list(combinations(A, 2))
    intra_B = list(combinations(B, 2))
    
    # Try adding subsets of intra-partition edges
    # Greedy: try each edge, add if valid
    current = [e[:] for e in edges]
    
    # Try adding intra-B edges first (B has 8 vertices, more room)
    random.seed(42)
    random.shuffle(intra_B)
    random.shuffle(intra_A)
    
    for u, v in intra_B + intra_A:
        candidate = current + [[u, v]]
        if is_valid(candidate):
            current = candidate
    
    if len(current) > best_count:
        best_edges = [e[:] for e in current]
        best_count = len(current)
    
    # Try local search: remove some edges, add others
    # Use time budget
    while time.time() < deadline - 5:
        # Try random restarts with different strategies
        current = [e[:] for e in edges]  # reset to K_{7,8}
        
        # Random order of intra edges
        all_intra = intra_B + intra_A
        random.shuffle(all_intra)
        
        for u, v in all_intra:
            if time.time() > deadline - 5:
                break
            candidate = current + [[u, v]]
            if is_valid(candidate):
                current = candidate
        
        if len(current) > best_count:
            best_count = len(current)
            best_edges = [e[:] for e in current]
    
    # Ensure proper types
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
