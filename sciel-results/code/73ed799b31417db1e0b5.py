"""Optimized construction for dense odd-order graphs avoiding equal-degree path endpoints."""
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _check_valid(edges, n):
    """Check if no simple 3-edge path has endpoints of equal degree."""
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    for u in range(n):
        for v in adj[u]:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == s or t == u:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _build_greedy(n):
    """
    Greedy construction:
    - Assign each vertex a target degree
    - Vertices with same degree must form independent set
    - Build edges respecting constraints
    """
    # Strategy: use a "blow-up" of a small base structure
    # Key constraint: if deg(u) == deg(v), then u and v cannot be adjacent
    # More generally: for any edge (u,v), no neighbor s of u and neighbor t of v can have deg(s)==deg(t)
    
    # Let's try: assign degrees so that vertices of same degree are non-adjacent
    # and for each edge (u,v), the neighborhoods of u and v have disjoint degree sets
    
    # Simple approach: use a regular-ish graph where degrees are all distinct
    # or carefully structured
    
    # Try: make vertex 0 have degree 0 (isolated), rest form a structure
    # Actually, let's try to maximize edges
    
    # Approach: start with all edges, remove violations
    # Better: constructive approach
    
    # Let's assign degrees: vertex i gets degree d_i
    # Constraint 1: if d_u == d_v, edge (u,v) forbidden
    # Constraint 2: for edge (u,v), for all s in N(u)\{v}, t in N(v)\{u}: d_s != d_t
    
    # This is complex. Let's use a greedy edge-addition approach.
    
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edges = []
    
    # Try adding edges in a smart order
    # Process vertices, try to connect them
    
    # Greedy: try all pairs, add if valid
    # To be efficient, use a heuristic order
    
    # Order: try to build high-degree vertices first
    # Start with a star-like structure then add chords
    
    # Actually, let's just do greedy: try each potential edge, add if it doesn't create a violation
    # Check validity incrementally
    
    def _is_valid_with_edge(adj, deg, u, v):
        """Check if adding edge (u,v) keeps the graph valid."""
        # After adding (u,v):
        # deg[u] += 1, deg[v] += 1
        # New paths s-u-v-t where s in N(u)\{v}, t in N(v)\{u}
        # Need deg_new[s] != deg_new[t] for all such s,t
        # Note: deg_new differs from deg only for u and v
        
        for s in adj[u]:
            if s == v:
                continue
            deg_s = deg[s] + (1 if s == u else 0)  # s is neighbor of u, s != v
            # Actually s is in N(u), s != v, so s != u (no loops)
            deg_s = deg[s]
            for t in adj[v]:
                if t == u:
                    continue
                deg_t = deg[t]
                if deg_s == deg_t:
                    return False
        return True
    
    # Greedy: iterate through all possible edges, add if valid
    # Use a priority: try edges that create fewer conflicts
    
    # For speed, do multiple passes with different orderings
    best_edges = []
    best_count = 0
    
    deadline = time.time() + 45  # safety margin
    
    # Try several heuristics
    for trial in range(50):
        if time.time() > deadline:
            break
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        # Random-ish order of edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        if trial == 0:
            # Natural order
            pass
        else:
            import random
            rng = random.Random(trial * 12345 + 67890)
            rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if _is_valid_with_edge(adj, deg, u, v):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
    
    return best_edges

def construct():
    n = 15
    edges = _build_greedy(n)
    # Convert to list of lists with Python ints
    result = [[int(u), int(v)] for u, v in edges]
    return result

# EVOLVE-BLOCK-END
