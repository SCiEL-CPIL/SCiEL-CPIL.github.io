import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        if not edges:
            return []
        
        edges_set = set(edges)
        n_vertices = max(max(u, v) for u, v in edges) + 1
        
        def decompose_with_pairing(rng):
            # 1. Create maximal pairing
            edge_partner = {}
            edge_partner_vertex = {}
            
            for v in range(n_vertices):
                v_edges = [e for e in edges if v in e]
                rng.shuffle(v_edges)
                for i in range(0, len(v_edges) - 1, 2):
                    e1 = v_edges[i]
                    e2 = v_edges[i+1]
                    edge_partner[e1] = (e2, v)
                    edge_partner[e2] = (e1, v)
                if len(v_edges) % 2 == 1:
                    edge_partner[v_edges[-1]] = (None, v)
            
            # 2. Find cycles and break them
            visited_edges = set()
            
            for e in edges:
                if e in visited_edges:
                    continue
                
                # Trace component
                comp = []
                current = e
                prev = None
                while True:
                    comp.append(current)
                    visited_edges.add(current)
                    partner, v = edge_partner[current]
                    if partner is None:
                        break
                    prev, current = current, partner
                    if current in comp:
                        # Cycle detected
                        break
                
                # Check if cycle
                # A component is a cycle if no edge in it is unpaired
                is_cyc = all(edge_partner[ce][0] is not None for ce in comp)
                
                if is_cyc:
                    # Break cycle
                    # Pick a random edge in comp
                    # But comp might contain the path part if it wasn't a pure cycle?
                    # No, if it's a cycle, all edges are paired.
                    # So comp is the cycle.
                    
                    # Pick a random edge to break
                    # We need to break a pair.
                    # Let's pick a random edge e_break in comp.
                    e_break = rng.choice(comp)
                    partner, v = edge_partner[e_break]
                    
                    # Unpair
                    edge_partner[e_break] = (None, v)
                    edge_partner[partner] = (None, v)
                    
                    # Note: This might create two new endpoints.
                    # The component is now a path.
            
            # 3. Extract paths
            paths = []
            visited = set()
            
            for e in edges:
                if e in visited:
                    continue
                
                # Find start of path (endpoint)
                # Walk to find an endpoint
                # Since it's a path, we can just walk from e until we hit an endpoint.
                # But we need the order.
                
                # Let's find the two endpoints of this component.
                # Walk from e in both directions?
                # Or just walk until None.
                
                # Since we broke cycles, every component is a path.
                # So there are exactly 2 endpoints (or 1 if single edge).
                
                # Find one endpoint
                current = e
                prev = None
                while True:
                    partner, v = edge_partner[current]
                    if partner is None:
                        break
                    prev, current = current, partner
                    if current in visited:
                        break
                
                # Now 'current' is an endpoint.
                # We need to reconstruct the path starting from 'current'.
                
                # Walk from 'current'
                path_edges = []
                curr = current
                prev = None
                while True:
                    path_edges.append(curr)
                    visited.add(curr)
                    partner, v = edge_partner[curr]
                    if partner is None:
                        break
                    prev, curr = curr, partner
                    if curr in visited:
                        break
                
                # Convert to vertex path
                if not path_edges:
                    continue
                
                # Find start vertex
                # The start vertex is the vertex where the first edge is unpaired.
                start_edge = path_edges[0]
                v_start = edge_partner[start_edge][1]
                
                # Determine the other vertex of start_edge
                u, v = start_edge
                if u == v_start:
                    next_v = v
                else:
                    next_v = u
                
                vertex_path = [v_start, next_v]
                
                # Walk the rest
                # We have the sequence of edges.
                # We can just use the local adjacency logic again to be safe.
                path_edge_set = set(path_edges)
                local_adj = defaultdict(list)
                for ee in path_edge_set:
                    a, b = ee
                    local_adj[a].append(b)
                    local_adj[b].append(a)
                
                curr_v = v_start
                prev_v = None
                while True:
                    neighbors = [n for n in local_adj[curr_v] if n != prev_v]
                    if not neighbors:
                        break
                    next_v = neighbors[0]
                    vertex_path.append(next_v)
                    prev_v, curr_v = curr_v, next_v
                
                paths.append(vertex_path)
            
            return paths
        
        best_paths = None
        best_count = float('inf')
        rng = random.Random(42)
        
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            paths = decompose_with_pairing(rng)
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
        
        if best_paths is None:
            best_paths = decompose_with_pairing(rng)
        
        return best_paths

    def validate_paths(paths, edges):
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(set(path)) != len(path):
                return False
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != edge_set:
            return False
        
        return True

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose(edges)
        
        if not validate_paths(paths, edges):
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
        
        if time.time() > deadline - 1:
            break
    
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])
    
    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
