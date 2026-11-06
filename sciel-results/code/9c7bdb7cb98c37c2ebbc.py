import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 140
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 10:
            # Fallback: trivial decomposition (each edge is a path)
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = _instance(*spec)
        
        # Solve using Line Graph Matching approach
        best_paths = _solve_line_graph_matching(edges, deadline)
        
        # Fallback to greedy if matching fails or times out
        if best_paths is None:
            best_paths = _greedy_decompose(edges, deadline)
            
        results.append(best_paths)
            
    return results

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _solve_line_graph_matching(edges, deadline):
    if not edges:
        return []
        
    # 1. Build Line Graph L(G)
    # Vertices of L(G) are indices of edges in G
    n_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Adjacency in L(G): edge i and j are connected if they share a vertex in G
    # We can build this by mapping vertices in G to list of incident edges
    vertex_to_edges = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        vertex_to_edges[u].append(i)
        vertex_to_edges[v].append(i)
        
    # L(G) adjacency list
    L_adj = defaultdict(set)
    for v in vertex_to_edges:
        es = vertex_to_edges[v]
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                e1 = es[i]
                e2 = es[j]
                L_adj[e1].add(e2)
                L_adj[e2].add(e1)
                
    # 2. Find Maximum Matching in L(G) with Greedy + Local Search
    # We want to maximize matching size, but also prefer matchings that result in simple paths in G.
    # For now, let's just find a large matching.
    
    best_matching = set()
    best_score = -1
    best_paths = None
    
    # Try multiple random restarts of greedy matching
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # Randomize adjacency order
        adj_list = {u: list(L_adj[u]) for u in L_adj}
        for u in adj_list:
            random.shuffle(adj_list[u])
            
        # Greedy Matching
        matching = set()
        matched = [False] * n_edges
        
        # Priority: maybe pick edges with high degree first? Or random?
        # Let's try random order of vertices
        vertices = list(range(n_edges))
        random.shuffle(vertices)
        
        for u in vertices:
            if matched[u]:
                continue
            for v in adj_list[u]:
                if not matched[v]:
                    # Add edge (u, v) to matching
                    matching.add(tuple(sorted((u, v))))
                    matched[u] = True
                    matched[v] = True
                    break
                    
        # Evaluate this matching
        paths = _reconstruct_paths_from_matching(matching, edges)
        
        if paths is None:
            # Invalid (should not happen if we just check simplicity later, but reconstruction might fail if we don't handle cycles)
            # Actually, reconstruction always succeeds, but paths might not be simple.
            pass
            
        # Check simplicity
        is_simple = True
        for p in paths:
            if len(set(p)) != len(p):
                is_simple = False
                break
                
        if is_simple:
            # Score: minimize number of paths
            score = -len(paths)
            if score > best_score:
                best_score = score
                best_matching = matching
                best_paths = paths
        else:
            # If not simple, we can still use it but it's worse.
            # Or we can try to break the non-simple paths.
            # For now, let's just ignore non-simple matchings in the "best" selection,
            # unless we have no valid simple ones.
            # Actually, we can break them.
            fixed_paths = _fix_non_simple_paths(paths, edges)
            if fixed_paths is not None:
                score = -len(fixed_paths)
                if score > best_score:
                    best_score = score
                    best_matching = matching
                    best_paths = fixed_paths

    if best_paths is not None:
        return best_paths
        
    # If no valid matching found (unlikely), fallback to greedy
    return None

def _reconstruct_paths_from_matching(matching, edges):
    # matching is a set of (i, j) where i, j are edge indices in G
    # Build graph of connections
    # Node: edge index
    # Edge: (i, j) in matching
    adj = defaultdict(list)
    for i, j in matching:
        adj[i].append(j)
        adj[j].append(i)
        
    visited = [False] * len(edges)
    paths = []
    
    for i in range(len(edges)):
        if visited[i]:
            continue
            
        # Start a path
        path_edges = []
        curr = i
        visited[curr] = True
        path_edges.append(curr)
        
        # Traverse
        while adj[curr]:
            next_e = adj[curr][0]
            if visited[next_e]:
                # Cycle detected or already visited
                # If it's a cycle, we stop here (it's a closed loop of edges)
                break
            visited[next_e] = True
            path_edges.append(next_e)
            curr = next_e
            
        # Reconstruct vertex path from edge indices
        # path_edges is a list of edge indices
        # We need to order them such that they form a path
        # Since it's a path in L(G), the edges are connected.
        # We need to determine the direction.
        
        if not path_edges:
            continue
            
        # Get the first edge
        first_e = path_edges[0]
        u, v = edges[first_e]
        
        # Try to build path starting with u
        # We have a sequence of edges. We can check which orientation works.
        # Since it's a path in L(G), consecutive edges share a vertex.
        
        # Let's try to construct the vertex sequence.
        # Start with u, v.
        # Next edge must share v or u.
        
        # Simple approach:
        # Try both orientations of the first edge.
        
        def try_build(start_u, start_v, edge_seq):
            curr_u, curr_v = start_u, start_v
            verts = [curr_u, curr_v]
            
            for k in range(1, len(edge_seq)):
                e_idx = edge_seq[k]
                e_u, e_v = edges[e_idx]
                
                # The new edge must be incident to curr_v (the current end)
                # It could be (curr_v, e_u) or (curr_v, e_v)
                
                if e_u == curr_v:
                    # Extend to e_v
                    curr_u, curr_v = e_u, e_v
                    verts.append(e_v)
                elif e_v == curr_v:
                    # Extend to e_u
                    curr_u, curr_v = e_v, e_u
                    verts.append(e_u)
                else:
                    # Not connected? Should not happen if matching is valid.
                    return None
                    
            return verts
            
        # Try forward
        p1 = try_build(u, v, path_edges)
        if p1:
            paths.append(p1)
            continue
            
        # Try backward (reverse the edge sequence? No, just swap start)
        # If forward failed, maybe the sequence needs to be traversed differently?
        # Actually, if the edges form a path, one of the two ends of the first edge must be the "start"
        # such that the chain continues.
        
        p2 = try_build(v, u, path_edges)
        if p2:
            paths.append(p2)
            continue
            
        # If both fail, something is wrong with the reconstruction logic or matching.
        # This can happen if the "path" in L(G) is not a simple path in terms of vertex continuity
        # due to branching? No, matching ensures degree 1 in L(G).
        # It can happen if the edges don't link up properly.
        # For safety, treat each edge as a separate path if reconstruction fails.
        for e_idx in path_edges:
            paths.append(list(edges[e_idx]))
            
    return paths

def _fix_non_simple_paths(paths, edges):
    # If a path has repeated vertices, break it.
    # This is a heuristic fix.
    fixed_paths = []
    for p in paths:
        if len(set(p)) == len(p):
            fixed_paths.append(p)
        else:
            # Break at first repeat
            seen = set()
            curr_path = []
            for v in p:
                if v in seen:
                    if curr_path:
                        fixed_paths.append(curr_path)
                    curr_path = [v]
                    seen = {v}
                else:
                    curr_path.append(v)
                    seen.add(v)
            if curr_path:
                fixed_paths.append(curr_path)
    return fixed_paths

def _greedy_decompose(edges, deadline):
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        
    paths = []
    
    while remaining:
        if time.time() > deadline:
            break
        # Pick start
        odd_vertices = [v for v in adj if degree.get(v,0) % 2 == 1 and adj[v]]
        if odd_vertices:
            start = random.choice(odd_vertices)
        else:
            start = max(adj.keys(), key=lambda x: degree.get(x,0))
            
        # Find long path
        path = [start]
        visited = {start}
        current = start
        while True:
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            # Warnsdorff
            neighbors.sort(key=lambda x: sum(1 for y in adj[x] if y not in visited))
            next_v = neighbors[0]
            path.append(next_v)
            visited.add(next_v)
            
            # Remove edge
            edge = tuple(sorted((current, next_v)))
            remaining.discard(edge)
            adj[current].discard(next_v)
            adj[next_v].discard(current)
            degree[current] -= 1
            degree[next_v] -= 1
            
            if not adj[current]: del adj[current]
            if not adj[next_v]: del adj[next_v]
            
            current = next_v
            
        paths.append(path)
        
    return paths
