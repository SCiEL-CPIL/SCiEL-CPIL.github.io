import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Construct Hoffman-Singleton graph (50 vertices)
    # Vertices are (i, S) where i in 0..4, S is a 2-subset of 0..4
    # Total vertices: 5 * 10 = 50
    
    vertices = []
    vertex_id = {}
    idx = 0
    for i in range(5):
        for S in itertools.combinations(range(5), 2):
            vertices.append((i, S))
            vertex_id[(i, S)] = idx
            idx += 1
    
    n_total = 50
    edges_hs = []
    
    # Generate edges
    for u in range(n_total):
        i_u, S_u = vertices[u]
        for v in range(u + 1, n_total):
            i_v, S_v = vertices[v]
            if i_u == i_v:
                # Same layer: disjoint sets
                if not (set(S_u) & set(S_v)):
                    edges_hs.append([u, v])
            else:
                # Different layers: j in S_u and i in S_v
                # Here u is (i_u, S_u), v is (i_v, S_v)
                # Condition: i_v in S_u and i_u in S_v
                if i_v in S_u and i_u in S_v:
                    edges_hs.append([u, v])
    
    # We need a subgraph with n=39 vertices
    # Take the first 39 vertices
    n_target = 39
    selected_vertices = set(range(n_target))
    
    edges_final = []
    for e in edges_hs:
        if e[0] in selected_vertices and e[1] in selected_vertices:
            edges_final.append(e)
    
    # Check time
    if time.time() > deadline:
        # Fallback to seed if somehow slow (unlikely)
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
        
    return {'n': n_target, 'edges': edges_final}
