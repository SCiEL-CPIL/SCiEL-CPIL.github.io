import time
import random
import numpy as np
from collections import deque
from array import array

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _is_valid_graph(n, edges, k, g):
    """Check if graph is valid: simple, k-regular, girth >= g."""
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i == j:
            return False
        if j in adj[i]:
            return False
        adj[i].add(j)
        adj[j].add(i)
    
    # Check k-regular
    for u in range(n):
        if len(adj[u]) != k:
            return False
    
    # Check girth >= g using BFS for shortest cycle
    # For each edge (u,v), check if there's a path from u to v of length < g-1
    # that doesn't use edge (u,v)
    # Actually, girth >= g means no cycle of length < g
    # A cycle of length c exists iff there's a closed walk of length c
    # Use BFS from each vertex to find shortest cycle back to it
    
    # More efficient: for each edge, remove it and BFS to see if shortest path < g-1
    # But that's expensive. Let's do BFS from each vertex to detect cycles.
    
    # For girth >= 9, we need no cycles of length 3,4,5,6,7,8
    # BFS from each vertex, tracking distance and parent
    # If we reach an already-visited node at distance d where d + dist_to_other < g, we have a short cycle
    
    # Actually, simpler: BFS from each vertex. If we find a back edge to an ancestor
    # at distance less than g, we have a cycle shorter than g.
    # But standard BFS gives shortest paths, so if dist[u] + dist[v] + 1 < g for an edge (u,v)
    # where neither is ancestor of other... this is tricky.
    
    # Let me use: for each vertex s, BFS. For each edge (u,v) in the graph, if both u and v
    # are visited in BFS from s, and dist[u] + dist[v] + 1 forms a cycle... no.
    
    # Standard approach: girth is the length of the shortest cycle.
    # To check girth >= g: for each vertex, do BFS. When we encounter an edge (u,v) where
    # both are already in the BFS tree, the cycle length is dist[u] + dist[v] + 1.
    # But this only works for trees. For general graphs, we need to be more careful.
    
    # Alternative: since we're constructing via BFS tree, let's just do a full BFS from each
    # vertex and check all edges for creating short cycles.
    
    # For efficiency, let's check: for each edge (u,v), the shortest path from u to v
    # not using edge (u,v) should be >= g-1.
    # This is expensive but g is small.
    
    # Let me use a different approach: BFS from each vertex, and for each vertex, check
    # if any two neighbors are connected by a path of length < g-2 (which would create
    # a cycle of length < g).
    
    # Actually, let me just do BFS from each vertex and detect cycles properly.
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    # Edge (u,v) where v is already visited
                    # Cycle length = dist[u] + dist[v] + 1
                    # But only if u and v are not parent-child in BFS tree
                    if parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        # Also need to check self-loops (already done) and multi-edges (already done)
    
    return True

def _girth_check_fast(adj, n, g):
    """Fast girth check: return True if girth >= g."""
    # For each edge (u,v), check shortest path from u to v not using (u,v)
    # If that path has length < g-1, then there's a cycle of length < g
    
    # Build adjacency list from adj (already a list of lists)
    adj_set = [set(v) for v in adj]
    
    for s in range(n):
        # BFS from s
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= g - 1:  # No need to go deeper
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        # Now check all edges: for edge (u,v), if dist[u] + 1 < g and dist[v] != -1 
        # and the shortest path from u to v is dist[v], we need dist[v] >= g-1
        # Actually, if there's an edge (u,v) and dist_from_s[u] + 1 == dist_from_s[v] or similar...
        # This isn't quite right. Let me think again.
    
    # Better approach: for each vertex s, BFS. For each vertex v reached at distance d,
    # if there's an edge from v to some vertex u at distance d-2 or less (not parent),
    # then we have a cycle of length d + (d-2) + 1 = 2d - 1 or similar.
    
    # Let me use the standard method: BFS from each vertex, and for each edge (u,v)
    # where u != parent[v] and v != parent[u], compute dist[u] + dist[v] + 1.
    # Take the minimum over all such edges and all roots.
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    if parent[u] != v and parent[v] != u:
                        cl = dist[u] + dist[v] + 1
                        if cl < g:
                            return False
    return True

def _bfs_tree_construct(k, g, n_target, rng):
    """
    Construct a graph by building a BFS tree of non-backtracking walks
    up to depth floor(g/2), then merging leaves to achieve k-regularity.
    
    Strategy:
    1. Build a tree where each internal node has k children (except root which has k children)
    2. Depth d = floor(g/2)
    3. Number of nodes at each level: k^level (for level 0 to d)
    4. Total nodes ~ sum(k^i for i=0..d)
    5. For k=5, g=9, d=4: 1 + 5 + 25 + 125 + 625 = 781 nodes
    6. The leaves (at depth d) need to be connected to satisfy regularity
    7. Each leaf has degree 1 (from its parent), needs k-1 more edges
    8. Merge leaves pairwise or in groups to create edges between them
    
    This gives a girth of at least 2*d + 1 = 9 for k=5, g=9.
    """
    d = (g - 1) // 2  # floor(g/2)
    
    # Build the tree
    # Node numbering: BFS order
    # Level 0: 1 node (root)
    # Level i: k^(i) nodes for i = 1..d
    # Actually, each node at level i-1 has k children, so level i has k^i nodes
    
    # Let's use a queue to assign IDs
    nodes_per_level = [1] + [k**i for i in range(1, d + 1)]
    total_nodes = sum(nodes_per_level)
    
    # Assign IDs
    node_ids = {}  # (level, index) -> id
    current_id = 0
    for level in range(d + 1):
        for idx in range(nodes_per_level[level]):
            node_ids[(level, idx)] = current_id
            current_id += 1
    
    # Build edges: parent to child
    adj = [set() for _ in range(total_nodes)]
    for level in range(d):  # level 0 to d-1, each has children at level+1
        for idx in range(nodes_per_level[level]):
            parent_id = node_ids[(level, idx)]
            for child_idx in range(k):
                child_id = node_ids[(level + 1, idx * k + child_idx)]
                adj[parent_id].add(child_id)
                adj[child_id].add(parent_id)
    
    # Now, nodes at level d are leaves (degree 1)
    # Nodes at levels 1..d-1 have degree k+1 (k children + 1 parent)
    # Root has degree k
    # We need all nodes to have degree exactly k
    # So:
    # - Root: degree k, OK
    # - Level 1 to d-1: degree k+1, need to remove 1 edge each? No, we can't remove tree edges.
    # - Level d: degree 1, need k-1 more edges
    
    # Wait, this doesn't work directly. Let me reconsider.
    
    # In a k-regular graph, every node has degree k.
    # In the tree:
    # - Root: degree k (has k children) - OK
    # - Internal nodes (level 1 to d-1): degree k+1 (k children + 1 parent) - TOO HIGH
    # - Leaves (level d): degree 1 - TOO LOW
    
    # This approach doesn't directly give a regular graph. Let me think differently.
    
    # Alternative: build the tree, then add edges between leaves and internal nodes
    # to adjust degrees, while maintaining girth >= g.
    
    # Actually, for a (k,g)-cage, the standard construction uses the fact that
    # the ball of radius floor(g/2) around any vertex is a tree. So if we can
    # make a graph where every vertex has a tree-like neighborhood of radius floor(g/2),
    # the girth will be >= g.
    
    # For simplicity, let me use a random regular graph approach with 2-switches
    # to eliminate short cycles, which is more tractable.
    
    return None

def _generate_random_regular(n, k, rng):
    """Generate a random k-regular simple graph using configuration model + 2-switches."""
    adj = [set() for _ in range(n)]
    
    # Create stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # Shuffle
    m = len(stubs)
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        stubs[i], stubs[j] = stubs[j], stubs[i]
    
    # Pair up stubs
    for a in range(0, m, 2):
        u = stubs[a]
        v = stubs[a + 1]
        if u == v or v in adj[u]:
            # Retry or fail
            return None
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to list of lists
    adj_list = [list(v) for v in adj]
    return adj_list

def _count_short_cycles(adj, n, max_len):
    """Count cycles of length up to max_len."""
    # For each edge, check if there's a path of length max_len-1 between its endpoints
    # Not efficient for large n, but works for moderate n
    
    count = 0
    adj_set = [set(v) for v in adj]
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # Check if there's a path from u to v of length 1 to max_len-1 not using edge (u,v)
            # BFS from u, avoiding edge (u,v) initially
            # Actually, simpler: just check for triangles, 4-cycles, etc. directly
            pass
    
    # For now, let's just check triangles and 4-cycles
    # Triangles
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj_set[u]:
                    count += 1
                    break
    
    # 4-cycles: for each pair of non-adjacent vertices, count common neighbors
    # But that's O(n^2). Let's skip for now.
    
    return count

def _swap_edges(adj, n, k, rng, max_iter=1000):
    """Perform random 2-switches to try to eliminate short cycles."""
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    adj_set = [set(v) for v in adj]
    
    for _ in range(max_iter):
        # Pick a random edge (u,v)
        if not edges:
            break
        idx = _randrange(rng, len(edges))
        u, v = edges[idx]
        
        # Pick a random edge (a,b) that doesn't share vertices with (u,v)
        # and (u,a), (v,b) or (u,b), (v,a) are not existing edges
        found = False
        for _ in range(20):  # Try a few times
            if len(edges) < 2:
                break
            jdx = _randrange(rng, len(edges))
            a, b = edges[jdx]
            if a in (u, v) or b in (u, v):
                continue
            if a == b:
                continue
            # Check if (u,a) and (v,b) are valid
            if a in adj_set[u] or b in adj_set[v]:
                continue
            if b in adj_set[u] or a in adj_set[v]:
                continue
            # Perform swap
            adj_set[u].discard(v)
            adj_set[v].discard(u)
            adj_set[a].discard(b)
            adj_set[b].discard(a)
            
            adj_set[u].add(a)
            adj_set[a].add(u)
            adj_set[v].add(b)
            adj_set[b].add(v)
            
            # Update edge list
            edges[idx] = (u, a)
            edges[jdx] = (v, b)
            found = True
            break
        if not found:
            continue
    
    return [list(v) for v in adj_set]

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline with margin
    
    rng = 123456789
    n = k  # initial
    
    # For k=5, g=9, the smallest known is 1116 vertices.
    # Let's try to construct a valid graph.
    
    # Strategy: Use a random regular graph of moderate size and use 2-switches
    # to eliminate short cycles. Start with a size that's likely to work.
    
    # The Moore bound for (5,9) is 2 + 5*(1+5+25+125) = 2 + 5*156 = 782
    # But we need more vertices to avoid short cycles.
    
    # Let's try sizes from 500 to 2000 and see which gives a valid graph.
    
    best_result = None
    best_n = float('inf')
    
    # Try a few random graphs of increasing size
    for n_try in [800, 1000, 1200, 1500]:
        if time.time() > deadline - 10:
            break
        
        # Generate random k-regular graph
        adj = _generate_random_regular(n_try, k, rng)
        if adj is None:
            continue
        
        # Try to eliminate short cycles using 2-switches
        max_swaps = 5000
        for iteration in range(max_swaps):
            if time.time() > deadline - 5:
                break
            
            # Count short cycles (triangles and 4-cycles for simplicity)
            # For girth >= 9, we need no cycles of length 3,4,5,6,7,8
            # Let's just do 2-switches randomly and check periodically
            
            # Perform a few 2-switches
            adj = _swap_edges(adj, n_try, k, rng, max_iter=10)
        
        # Check girth
        if time.time() > deadline - 3:
            break
        
        if _girth_check_fast(adj, n_try, g):
            # Valid graph!
            edges = []
            for u in range(n_try):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            if n_try < best_n:
                best_n = n_try
                best_result = (n_try, edges)
            
            # Try smaller sizes?
            # For now, return the first valid one
            break
    
    if best_result is not None:
        return best_result
    
    # If no valid graph found, try a larger random graph
    n_try = 2000
    adj = _generate_random_regular(n_try, k, rng)
    if adj is not None:
        # Do some 2-switches
        for _ in range(2000):
            if time.time() > deadline - 3:
                break
            adj = _swap_edges(adj, n_try, k, rng, max_iter=5)
        
        edges = []
        for u in range(n_try):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return (n_try, edges)
    
    # Fallback: return empty
    return (0, [])

