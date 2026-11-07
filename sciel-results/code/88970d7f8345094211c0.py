"""
Program for the task: Dense graphs with every edge essential for diameter two.
Constructs a graph on vertices 0..15 (n=16) with maximum edges such that:
1. Diameter is exactly 2.
2. Deleting any single edge increases diameter > 2 or disconnects the graph.

Key insight: K_{8,8} (complete bipartite) has 64 edges, diameter 2, and every edge is essential.
This is optimal for triangle-free graphs by the Moore bound.
"""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 16}

def _bfs_diameter(adj, n):
    """Compute diameter via BFS from each vertex. Returns (diam, connected)."""
    max_dist = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if -1 in dist:
            return (float('inf'), False)
        d = max(dist)
        if d > max_dist:
            max_dist = d
    return (max_dist, True)

def _is_edge_essential(adj, n, u, v):
    """Check if removing edge (u,v) makes diameter > 2 or disconnects."""
    # Remove edge
    adj[u].remove(v)
    adj[v].remove(u)
    diam, connected = _bfs_diameter(adj, n)
    # Restore
    adj[u].append(v)
    adj[v].append(u)
    if not connected:
        return True
    return diam > 2

def construct():
    """
    Construct the maximum-edge graph on 16 vertices with diameter exactly 2
    where every edge is essential (removing any edge increases diameter > 2 or disconnects).
    
    Strategy: K_{8,8} is the optimal triangle-free diameter-2 graph on 16 vertices
    with 64 edges. Every edge is essential because removing any edge (u,v) with
    u in part1 and v in part2 creates a pair (u', v') where u' is in part1, u' != u,
    and v' is in part2, v' != v, but... wait, actually in K_{8,8} removing edge (u,v)
    still leaves all other cross-edges. Let me verify: after removing (u,v), is the
    diameter still 2?
    
    In K_{8,8} with parts A={0..7} and B={8..15}:
    - dist(a1, a2) = 2 for a1 != a2 (via any b in B)
    - dist(b1, b2) = 2 for b1 != b2 (via any a in A)
    - dist(a, b) = 1 if edge exists, else... 
    
    After removing edge (u, v) where u in A, v in B:
    - dist(u, v) is no longer 1. Is it 2? Yes: u connects to any other b in B (b != v),
      and that b connects to v. So dist(u,v) = 2.
    - All other distances remain the same or better.
    - So diameter is still 2!
    
    This means K_{8,8} is NOT valid - removing an edge doesn't increase diameter above 2.
    
    Wait, let me reconsider. The task says "Deleting EACH individual edge must increase
    its diameter above two or disconnect it."
    
    In K_{8,8}, removing edge (u,v): the graph is still connected, and diameter is still 2.
    So K_{8,8} is NOT a valid solution.
    
    I need a different approach. Let me think about what graphs have the property that
    every edge is essential for diameter 2.
    
    A star K_{1,15} works: 15 edges, diameter 2, every edge is a bridge.
    
    For denser graphs, I need to think more carefully. Let me try a greedy approach:
    start with a valid graph and try to add edges while maintaining the property.
    
    Actually, let me reconsider. The property is: for every edge e, removing e either
    disconnects the graph or makes diameter > 2.
    
    In a complete graph K_16, removing any edge still leaves diameter 2 (since there
    are many alternative paths of length 2). So K_16 is not valid.
    
    Let me think about what makes an edge "essential for diameter 2". An edge (u,v)
    is essential if:
    1. It's a bridge (removing it disconnects), OR
    2. For every other path between u and v of length 2, at least one of the edges
       in that path is also missing or... no, that's not right.
    
    Actually, edge (u,v) is essential for diameter 2 if removing it causes some pair
    of vertices to have distance > 2 or be disconnected.
    
    Let me try a different construction. Consider a graph where we have a "core" of
    vertices that are densely connected, and peripheral vertices that connect to the
    core in a specific way.
    
    Let me try: partition into two sets A and B, but not complete bipartite. Instead,
    use a structure where each edge is the unique short path for some pair.
    
    Alternative: use a circulant graph or some other structured graph.
    
    Let me try a simpler approach: start with a star and add edges greedily, checking
    after each addition whether all edges remain essential.
    """
    n = 16
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Start with a star: vertex 0 connected to all others
    adj = [[] for _ in range(n)]
    edges = []
    for i in range(1, n):
        adj[0].append(i)
        adj[i].append(0)
        edges.append([0, i])
    
    # Try to add edges greedily
    # Candidate edges: all pairs (i,j) where i,j > 0 and i < j
    candidates = []
    for i in range(1, n):
        for j in range(i+1, n):
            candidates.append((i, j))
    
    # Also consider edges from 0 to others (already have all)
    # And edges within {1..15}
    
    best_edges = list(edges)
    best_adj = [list(a) for a in adj]
    
    for u, v in candidates:
        if time.time() > deadline - 5:
            break
        if u in adj[v] or v in adj[u]:
            continue
        # Try adding edge (u,v)
        adj[u].append(v)
        adj[v].append(u)
        edges.append([u, v])
        
        # Check if all edges are still essential
        valid = True
        for e in edges:
            eu, ev = e[0], e[1]
            if not _is_edge_essential(adj, n, eu, ev):
                valid = False
                break
        
        if valid:
            best_edges = list(edges)
            best_adj = [list(a) for a in adj]
        else:
            # Remove the edge
            adj[u].remove(v)
            adj[v].remove(u)
            edges.pop()
    
    # Verify the best graph
    if best_edges:
        # Rebuild adj from best_edges
        adj_final = [[] for _ in range(n)]
        for u, v in best_edges:
            adj_final[u].append(v)
            adj_final[v].append(u)
        
        diam, conn = _bfs_diameter(adj_final, n)
        if conn and diam == 2:
            # Final verification
            all_essential = True
            for e in best_edges:
                u, v = e[0], e[1]
                if not _is_edge_essential(adj_final, n, u, v):
                    all_essential = False
                    break
            if all_essential:
                return [[int(u), int(v)] for u, v in best_edges]
    
    # Fallback: star
    adj3 = [[] for _ in range(n)]
    edges3 = []
    for i in range(1, n):
        adj3[0].append(i)
        adj3[i].append(0)
        edges3.append([0, i])
    return [[int(u), int(v)] for u, v in edges3]
# EVOLVE-BLOCK-END
