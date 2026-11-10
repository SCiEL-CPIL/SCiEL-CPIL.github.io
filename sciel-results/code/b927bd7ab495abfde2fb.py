# EVOLVE-BLOCK-START
import time
from collections import deque

def _hopcroft_karp(adj_left, n_left, n_right):
    """Maximum bipartite matching using Hopcroft-Karp.
    adj_left[u] = list of right nodes adjacent to left node u.
    Returns the size of the maximum matching."""
    if n_left == 0 or n_right == 0:
        return 0
    match_l = [-1] * n_left   # match_l[u] = v or -1
    match_r = [-1] * n_right  # match_r[v] = u or -1
    dist = [0] * n_left
    
    def bfs():
        dq = deque()
        found = False
        for u in range(n_left):
            if match_l[u] == -1:
                dist[u] = 0
                dq.append(u)
            else:
                dist[u] = -1
        while dq:
            u = dq.popleft()
            for v in adj_left[u]:
                mu = match_r[v]
                if mu == -1:
                    found = True
                elif dist[mu] == -1:
                    dist[mu] = dist[u] + 1
                    dq.append(mu)
        return found
    
    def dfs(u):
        for v in adj_left[u]:
            mu = match_r[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = 0
    while bfs():
        for u in range(n_left):
            if match_l[u] == -1:
                if dfs(u):
                    matching += 1
    return matching

def _decompose_graph(edges):
    """Decompose edges into minimum number of edge-disjoint paths.
    edges: list of (u, v) tuples.
    Returns list of paths (each path is a list of vertices)."""
    n_edges = len(edges)
    if n_edges == 0:
        return []
    
    # Build adjacency: for each vertex, list of edge indices incident to it
    from collections import defaultdict
    vert_to_edges = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        vert_to_edges[u].append(i)
        vert_to_edges[v].append(i)
    
    # Build bipartite graph for the reduction
    # Left node i represents edge i "leaving" some vertex
    # Right node j represents edge j "entering" some vertex
    # We can chain edge i into edge j at vertex v if they share vertex v
    # and i uses v as one endpoint and j uses v as the other (they are distinct edges)
    # Actually: at vertex v, edge i=(a,v) can be followed by edge j=(v,b)
    # So left node i is adjacent to right node j if they share a vertex v,
    # and the shared vertex is an "end" of i and a "start" of j in some orientation.
    # But since edges are undirected, we can orient them: at vertex v, 
    # edge i can transition to edge j if both are incident to v and i != j.
    
    # The bipartite graph: left = edge indices, right = edge indices
    # adj_left[i] = list of j such that edge i and edge j share a vertex
    # (and i != j)
    
    adj_left = [[] for _ in range(n_edges)]
    for v, e_list in vert_to_edges.items():
        d = len(e_list)
        for a in range(d):
            for b in range(d):
                if a != b:
                    i = e_list[a]
                    j = e_list[b]
                    adj_left[i].append(j)
    
    # Remove duplicates
    for i in range(n_edges):
        adj_left[i] = list(set(adj_left[i]))
    
    # Find maximum matching
    max_match = _hopcroft_karp(adj_left, n_edges, n_edges)
    
    # Now we need to reconstruct the actual paths
    # The matching tells us which edges are chained together
    # Build the chain structure:
    # match_l[i] = j means edge i is followed by edge j (at some shared vertex)
    # match_r[j] = i means edge j is preceded by edge i
    
    # We need match_l and match_r from the algorithm. Let me redo to get them.
    
    # Actually, let me redo the matching to capture the actual matches.
    match_l = [-1] * n_edges
    match_r = [-1] * n_edges
    dist = [0] * n_edges
    
    def bfs2():
        dq = deque()
        found = False
        for u in range(n_edges):
            if match_l[u] == -1:
                dist[u] = 0
                dq.append(u)
            else:
                dist[u] = -1
        while dq:
            u = dq.popleft()
            for v in adj_left[u]:
                mu = match_r[v]
                if mu == -1:
                    found = True
                elif dist[mu] == -1:
                    dist[mu] = dist[u] + 1
                    dq.append(mu)
        return found
    
    def dfs2(u):
        for v in adj_left[u]:
            mu = match_r[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs2(mu)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = -1
        return False
    
    while bfs2():
        for u in range(n_edges):
            if match_l[u] == -1:
                if dfs2(u):
                    pass
    
    # Now reconstruct paths
    # Each edge i has:
    #   - a "next" edge: match_l[i] (the edge that follows i)
    #   - a "prev" edge: match_r[i] (the edge that precedes i)
    # A path starts at an edge with no prev (match_r[i] == -1)
    # and follows next pointers until no next (match_l[i] == -1)
    
    paths = []
    visited_edges = set()
    
    for start_e in range(n_edges):
        if match_r[start_e] != -1:
            continue  # not a start
        if start_e in visited_edges:
            continue
        
        # Trace the path
        path_edges = []
        e = start_e
        while e != -1 and e not in visited_edges:
            visited_edges.add(e)
            path_edges.append(e)
            e = match_l[e]
        
        if not path_edges:
            continue
        
        # Convert edge list to vertex list
        # We need to figure out the correct order of vertices
        # For each consecutive pair of edges (e1, e2) in path_edges,
        # they share a vertex. The path goes through that shared vertex.
        
        vertices = []
        prev_edge = None
        for idx, e in enumerate(path_edges):
            u, v = edges[e]
            if idx == 0:
                # First edge: start at one endpoint
                # Determine which endpoint is the "start"
                # If there's no prev edge, either endpoint could be start
                # But we need consistency with the next edge
                if len(path_edges) > 1:
                    next_e = path_edges[1]
                    nu, nv = edges[next_e]
                    # The shared vertex between e and next_e
                    if u == nu or u == nv:
                        # shared vertex is u, so start at v
                        vertices.append(v)
                        vertices.append(u)
                    else:
                        # shared vertex is v, so start at u
                        vertices.append(u)
                        vertices.append(v)
                else:
                    vertices.append(u)
                    vertices.append(v)
            else:
                # Determine which endpoint to add
                # The shared vertex with previous edge is already in vertices
                # Add the other endpoint
                if prev_edge is not None:
                    pu, pv = edges[prev_edge]
                    # Find shared vertex between prev_edge and current edge
                    if u == pu or u == pv:
                        # shared is u, add v
                        vertices.append(v)
                    else:
                        # shared is v, add u
                        vertices.append(u)
                else:
                    vertices.append(v)
            prev_edge = e
        
        if len(vertices) >= 2:
            paths.append(vertices)
    
    # Check if all edges are covered
    if len(visited_edges) != n_edges:
        # Some edges weren't visited - this shouldn't happen but handle it
        # Add remaining edges as individual paths
        for e in range(n_edges):
            if e not in visited_edges:
                u, v = edges[e]
                paths.append([u, v])
                visited_edges.add(e)
    
    return paths

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    from random import Random
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        paths = _decompose_graph(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
