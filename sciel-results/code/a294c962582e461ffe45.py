"""Rewritten program: find max triangle-free 16-vertex graph with chromatic number >= 4.

Strategy:
1. Enumerate all maximal triangle-free graphs via a backtracking edge-addition process
   with strong pruning (triangle check + 3-colorability check to keep only non-3-colorable).
   Actually, better: use the known fact that the maximum edges for a triangle-free graph
   on 16 vertices is floor(16^2/4) = 64 (Mantel's theorem). But we need chi >= 4.
   
   The complete bipartite graph K_8,8 has 64 edges, is triangle-free, and is 2-colorable.
   We need to remove edges to make it non-3-colorable while keeping as many edges as possible.
   
   Actually, a cleaner approach: search over all subsets of the 64 edges of K_8,8 that
   form a triangle-free graph (which they all do since K_8,8 is bipartite) and check
   3-colorability. But 2^64 is too large.

   Better approach: Use the Mycielski construction or known extremal examples.
   
   Known: The maximum number of edges in a triangle-free graph on n vertices with chi >= 4
   is studied. For n=16, let's think:
   
   The odd cycle C_5 has 5 edges, chi=3. Mycielski gives Grötzsch on 11 vertices, 20 edges, chi=4.
   To get to 16 vertices, we can add 5 more vertices.
   
   Alternative: Use a computer search with good heuristics. Since n=16 is small, we can
   use a branch-and-bound on edge additions.
   
   Let me implement a direct search:
   - Start with empty graph on 16 vertices.
   - Consider all 120 possible edges.
   - Use backtracking: add edges one by one, maintaining triangle-free property.
   - At each node, check if the graph is 3-colorable; if not, we have a valid candidate.
   - Use bounds: max additional edges possible given current adjacency.
   - Order edges to find high-edge-count solutions early.

   For efficiency with 120 s timeout, I'll use a randomized/local search approach
   combined with a small exact search on a restricted edge set.
   
   Actually, let me use a different known construction:
   Take C_5, apply Mycielski twice to get 47 vertices - too many.
   
   For 16 vertices with chi >= 4 and triangle-free:
   One known construction: take two copies of C_5, and connect them cleverly.
   
   Let me just do a focused search:
   1. Use the Grötzsch graph (11 vertices, 20 edges, chi=4, triangle-free).
   2. Add 5 more vertices (11-15).
   3. For each new vertex, determine which of the 11 old vertices to connect to,
      subject to triangle-freeness.
   4. Also consider edges among the 5 new vertices.
   5. Use local search / simulated annealing to maximize edges while keeping chi >= 4.

   Given time constraints, I'll implement a greedy + local search:
   - Start with Grötzsch graph.
   - Add vertices 11-15 one at a time, choosing connections that maximize edges
     while preserving triangle-free and non-3-colorability.
   - Then try to add more edges among the new vertices.
   - Local search: randomly add/remove edges, accept if improves edge count
     while maintaining constraints.

   For 3-colorability check, use backtracking with vertex ordering.
"""

import random
import itertools
from collections import defaultdict

def is_triangle_free(adj, n):
    """Check if graph with adjacency set is triangle-free."""
    for u in range(n):
        for v in adj[u]:
            for w in adj[u]:
                if v < w and v in adj[w]:
                    return False
    return True

def is_triangle_free_edge(adj, n, u, v):
    """Check if adding edge (u,v) keeps graph triangle-free."""
    if v in adj[u]:
        return False
    # Check no common neighbor
    for w in adj[u]:
        if w in adj[v]:
            return False
    return True

def is_3_colorable(adj, n):
    """Backtracking 3-colorability check."""
    color = [-1] * n
    
    # Order vertices by degree (highest first) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = {v: i for i, v in enumerate(order)}
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        used = set()
        for w in adj[v]:
            if color[w] != -1:
                used.add(color[w])
        for c in range(3):
            if c not in used:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def make_grotzsch():
    """Create the Grötzsch graph on vertices 0..10."""
    adj = [set() for _ in range(11)]
    # Cycle 0-1-2-3-4-0
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        adj[u].add(v)
        adj[v].add(u)
    # Mycielski: vertex 5+i connected to i and i+1 (mod 5)... 
    # Standard Mycielski: for each edge (u,v) in C5, add vertex 5+u connected to v,
    # and 5+v connected to u. Then add vertex 10 connected to all 5..9.
    for u, v in cycle:
        adj[u].add(5 + v)
        adj[5 + v].add(u)
        adj[v].add(5 + u)
        adj[5 + u].add(v)
    for i in range(5):
        adj[10].add(5 + i)
        adj[5 + i].add(10)
    return adj

def count_edges(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2

def add_edge(adj, u, v):
    adj[u].add(v)
    adj[v].add(u)

def remove_edge(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    """
    # Start with Grötzsch graph on vertices 0..10
    adj = make_grotzsch()
    n = 16
    # Extend adjacency to 16 vertices
    while len(adj) < n:
        adj.append(set())
    
    # Initial edge count
    edge_count = count_edges(adj, n)
    
    # Now add vertices 11..15 with optimized connections
    # For each new vertex, try all subsets of connections to old vertices
    # that maintain triangle-free property
    
    best_adj = [set(s) for s in adj]
    best_count = edge_count
    
    # Greedy addition of new vertices
    for new_v in range(11, 16):
        # Find all subsets of neighbors from existing vertices that are triangle-free
        existing = list(range(new_v))
        # Try all subsets, but that's 2^11 = 2048, feasible
        best_conn = set()
        best_edges = -1
        for r in range(len(existing) + 1):
            for subset in itertools.combinations(existing, r):
                # Check if connecting new_v to all in subset is triangle-free
                valid = True
                for u in subset:
                    for w in subset:
                        if u != w and w in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Also check that subset itself has no edges among them (else triangle)
                # Actually, if u and w are both in subset and edge (u,w) exists, triangle
                # Let me recheck: triangle is (new_v, u, w) where new_v-u, new_v-w, u-w all exist
                # So we need no edge between any two vertices in subset
                valid = True
                for i in range(len(subset)):
                    for j in range(i + 1, len(subset)):
                        if subset[j] in adj[subset[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                edges = len(subset)
                if edges > best_edges:
                    best_edges = edges
                    best_conn = set(subset)
        
        # Add best connections
        for u in best_conn:
            add_edge(adj, new_v, u)
        
        # Check if graph is still non-3-colorable
        if not is_3_colorable(adj, n):
            new_count = count_edges(adj, n)
            if new_count > best_count:
                best_count = new_count
                best_adj = [set(s) for s in adj]
        else:
            # If it became 3-colorable, we need to keep it that way
            # The connections made it 3-colorable, so revert
            for u in best_conn:
                remove_edge(adj, new_v, u)
    
    # Local search: try adding edges among vertices 11-15 and between old and new
    # to increase edge count while maintaining constraints
    random.seed(42)
    current_count = count_edges(adj, n)
    
    for iteration in range(20000):
        # Randomly pick an edge to try adding
        u = random.randint(0, 15)
        v = random.randint(0, 15)
        if u == v or v in adj[u]:
            continue
        if not is_triangle_free_edge(adj, n, u, v):
            continue
        add_edge(adj, u, v)
        if is_3_colorable(adj, n):
            remove_edge(adj, u, v)
        else:
            current_count = count_edges(adj, n)
            if current_count > best_count:
                best_count = current_count
                best_adj = [set(s) for s in adj]
    
    # Also try adding edges more systematically
    # Try all possible edges among new vertices 11-15
    for u in range(11, 16):
        for v in range(u + 1, 16):
            if v not in adj[u] and is_triangle_free_edge(adj, n, u, v):
                add_edge(adj, u, v)
                if not is_3_colorable(adj, n):
                    current_count = count_edges(adj, n)
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = [set(s) for s in adj]
                else:
                    remove_edge(adj, u, v)
    
    # More local search
    for iteration in range(30000):
        u = random.randint(0, 15)
        v = random.randint(0, 15)
        if u == v or v in adj[u]:
            continue
        if not is_triangle_free_edge(adj, n, u, v):
            continue
        add_edge(adj, u, v)
        if not is_3_colorable(adj, n):
            current_count = count_edges(adj, n)
            if current_count > best_count:
                best_count = current_count
                best_adj = [set(s) for s in adj]
        else:
            remove_edge(adj, u, v)
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
