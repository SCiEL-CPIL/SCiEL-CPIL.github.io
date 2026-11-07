"""Dense diameter-2 graph where every edge is critical (minimally 2-edge-connected for diameter 2)."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

N = 16
DEADLINE_MARGIN = 5.0  # seconds to keep before the 60s limit (or use 85s if 90s wall)

def _bfs_distances(adj, start):
    """Return list of distances from start to all nodes. adj is list of sets."""
    dist = [-1] * N
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _diameter(adj):
    """Return diameter of graph, or -1 if disconnected."""
    max_d = 0
    for s in range(N):
        d = _bfs_distances(adj, s)
        if -1 in d:
            return -1
        m = max(d)
        if m > max_d:
            max_d = m
    return max_d

def _edge_is_critical(adj, u, v):
    """Check if removing edge (u,v) makes diameter > 2 or disconnected."""
    # Create a copy of adj without this edge
    adj2 = [set(s) for s in adj]
    adj2[u].discard(v)
    adj2[v].discard(u)
    return _diameter(adj2) != 2

def construct():
    start_time = time.time()
    
    # We need to find a graph on 16 vertices with:
    # 1. Diameter exactly 2
    # 2. Every edge is critical (removing any edge makes diameter > 2 or disconnected)
    # 3. Maximize number of edges
    
    # Strategy: Start with a candidate dense graph, then verify edge-criticality.
    # A good starting point for diameter 2 with many edges: consider graphs close to complete
    # but missing enough edges to make each remaining edge critical.
    
    # Key insight: For an edge (u,v) to be critical, there must be no alternative path of length 2
    # between u and v. That is, u and v must have no common neighbors.
    
    # So we need a graph where for every edge (u,v), u and v share no common neighbor.
    # This is equivalent to: the graph is a "2-edge-critical diameter-2 graph" where each edge
    # is the unique length-2 path or the only connection maintaining diameter 2.
    
    # Actually, let's think more carefully. If we remove edge (u,v), the diameter increases above 2
    # or the graph disconnects. This means that in the original graph, the shortest path between
    # u and v is 1 (the edge itself), and after removal, the shortest path becomes > 2 or infinity.
    # So u and v must have NO common neighbors in the original graph.
    
    # Therefore: for every edge (u,v) in the graph, |N(u) ∩ N(v)| = 0.
    # This is a strong constraint. Let's construct such a graph.
    
    # One approach: use a graph based on a 1-factorization or pairing structure.
    # Consider splitting 16 vertices into pairs. If we connect vertices only within certain
    # structured sets such that adjacent vertices don't share neighbors...
    
    # Alternative: Think of it as a graph where the adjacency matrix A satisfies:
    # For all edges (u,v): (A^2)[u][v] = 0, i.e., no common neighbors.
    # But we also need diameter 2, so for all non-edges (u,v): (A^2)[u][v] >= 1.
    
    # Let me try a construction:
    # Label vertices 0-15. Consider a graph where we have a "core" structure.
    
    # Let's try: take the complete bipartite graph K_{8,8} and add some edges within parts
    # such that the diameter is 2 and every edge is critical.
    
    # Actually, K_{8,8} has diameter 2 already (any two vertices in same part have common neighbors
    # in the other part, and vertices in different parts are adjacent). But edges in K_{8,8} are
    # NOT critical because removing one edge still leaves many paths of length 2.
    
    # We need a sparser structure. Let me think about what graphs satisfy:
    # 1. Diameter 2
    # 2. For every edge (u,v), u and v have no common neighbor
    
    # This means the graph is a "distance-2 critical" graph. Let's try small examples first.
    
    # For n=4: A cycle C4 has diameter 2 and every edge is critical (removing any edge makes it
    # a path of length 3, diameter 3). C4 has 4 edges.
    
    # For n=6: Can we do better? Let's try a hexagon-like structure or other configs.
    
    # Let me try a systematic approach: use simulated annealing or local search to find a graph
    # with many edges satisfying the constraints.
    
    # But first, let me try a structured approach.
    
    # Idea: Use a "blow-up" of a smaller critical graph. If G is a diameter-2 graph where every
    # edge is critical, and we blow up each vertex into a clique of size k, we need to check if
    # the property is preserved. Probably not directly.
    
    # Let me try: start with a random graph and iteratively remove edges that are not critical,
    # then try to add edges back while maintaining the property.
    
    # Actually, the constraint is very tight. Let me think about upper bounds.
    
    # If every edge (u,v) has no common neighbors, then the neighborhoods of adjacent vertices
    # are disjoint (except for the mutual connection). This limits how dense the graph can be.
    
    # For a vertex u with degree d, its d neighbors must have pairwise disjoint neighborhoods
    # (excluding u). Each neighbor v of u has at most (n-1) - 1 - (d-1) = n - d - 1 other neighbors
    # (can't connect to u's other neighbors).
    
    # This suggests maximum degree is around sqrt(n) or so.
    
    # Let me try a concrete construction for n=16:
    # Use a 4x4 grid-like structure or a product of smaller graphs.
    
    # Actually, let me try: take the Cartesian product of two C4 graphs. C4 □ C4 has 16 vertices.
    # Diameter of C4 □ C4 is 2+2=4, too large.
    
    # How about a different approach: use a graph based on a 1-factorization of K_16?
    
    # Let me just try local search with the constraint checking.
    
    # Initialize with a random graph that has diameter 2.
    import random
    random.seed(42)
    
    # Start with a star-like or hub-based structure to ensure diameter 2
    # Actually, let's start with a graph that's guaranteed to have diameter 2:
    # Pick a random spanning structure and add edges.
    
    # Better: start with the complement of a matching or similar sparse structure,
    # which tends to have small diameter.
    
    # Let me try: start with all possible edges, then remove edges that are not critical.
    # But that might lead to too few edges.
    
    # Alternative: start with a moderate-density random graph, check diameter=2,
    # then iteratively remove non-critical edges, then try to add critical edges back.
    
    # Given the time constraint, let me try a few structured candidates.
    
    # Candidate 1: Based on a 1-factorization idea
    # Pair up vertices: (0,1), (2,3), ..., (14,15)
    # Connect pairs in a structured way.
    
    # Let me try: create a graph where vertices are arranged in a 4x4 grid,
    # and connect based on some pattern.
    
    # Actually, let me try the simplest approach: use local search.
    
    def _random_graph_with_diam2():
        """Generate a random graph likely to have diameter 2."""
        # Start with a random permutation-based structure
        edges = set()
        # Ensure connectivity and small diameter by adding a random spanning tree
        perm = list(range(N))
        random.shuffle(perm)
        for i in range(1, N):
            u, v = perm[i], perm[random.randint(0, i-1)]
            edges.add((min(u,v), max(u,v)))
        # Add random extra edges
        all_pairs = list(itertools.combinations(range(N), 2))
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if (min(u,v), max(u,v)) not in edges and random.random() < 0.3:
                edges.add((min(u,v), max(u,v)))
        return edges
    
    def _try_local_search(initial_edges, max_iter=5000):
        """Local search to maximize edges while maintaining diameter 2 and edge-criticality."""
        edges = set(initial_edges)
        
        def build_adj(edge_set):
            adj = [set() for _ in range(N)]
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj
        
        # First, ensure diameter is 2
        adj = build_adj(edges)
        if _diameter(adj) != 2:
            return None
        
        # Remove non-critical edges
        changed = True
        while changed:
            changed = False
            adj = build_adj(edges)
            for u, v in list(edges):
                if _edge_is_critical(adj, u, v):
                    continue
                else:
                    edges.discard((min(u,v), max(u,v)))
                    changed = True
        
        # Now try to add edges back
        all_pairs = list(itertools.combinations(range(N), 2))
        random.shuffle(all_pairs)
        
        for _ in range(max_iter):
            progress = False
            for u, v in all_pairs:
                if (min(u,v), max(u,v)) in edges:
                    continue
                # Try adding this edge
                new_edges = edges | {(min(u,v), max(u,v))}
                adj = build_adj(new_edges)
                if _diameter(adj) != 2:
                    continue
                # Check if all edges are still critical
                ok = True
                for a, b in new_edges:
                    if not _edge_is_critical(adj, a, b):
                        ok = False
                        break
                if ok:
                    edges = new_edges
                    progress = True
                    break
            
            if not progress:
                # Try random restarts of the addition order
                random.shuffle(all_pairs)
        
        return edges
    
    # Try multiple random initializations
    best_edges = None
    best_count = 0
    
    for attempt in range(20):
        if time.time() - start_time > 50:
            break
        
        init = _random_graph_with_diam2()
        result = _try_local_search(init, max_iter=2000)
        if result is not None:
            # Verify
            adj = [set() for _ in range(N)]
            for u, v in result:
                adj[u].add(v)
                adj[v].add(u)
            if _diameter(adj) == 2:
                # Check all edges are critical
                all_critical = True
                for u, v in result:
                    if not _edge_is_critical(adj, u, v):
                        all_critical = False
                        break
                if all_critical:
                    if len(result) > best_count:
                        best_count = len(result)
                        best_edges = result
    
    if best_edges is not None:
        # Convert to list of [u,v]
        return [[u, v] for u, v in sorted(best_edges)]
    else:
        # Fallback: return a known valid graph (e.g., C4 blown up or a simple valid graph)
        # For n=16, let's return a cycle-like structure that has diameter 2... 
        # Actually a simple cycle C16 has diameter 8, not 2.
        # Let's return a star: center 0 connected to all others. Diameter 2.
        # But are edges critical? Removing edge (0,v) disconnects v. So yes, all critical.
        # Star has 15 edges.
        return [[0, v] for v in range(1, N)]

# EVOLVE-BLOCK-END
