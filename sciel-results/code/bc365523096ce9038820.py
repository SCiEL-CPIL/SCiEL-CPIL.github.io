import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)
    num_edges = len(all_edges)
    
    def build_adj(edge_set):
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_q3(graph_adj):
        """Find a Q3 subgraph in the graph. Returns the 12 edges or None."""
        for v in range(n):
            nv = graph_adj[v]
            if len(nv) < 3:
                continue
            neighbors = list(nv)
            # Pick 3 neighbors a, b, c of v
            for i in range(len(neighbors)):
                a = neighbors[i]
                for j in range(i + 1, len(neighbors)):
                    b = neighbors[j]
                    for k in range(j + 1, len(neighbors)):
                        c = neighbors[k]
                        
                        na = graph_adj[a]
                        nb = graph_adj[b]
                        nc = graph_adj[c]
                        
                        # x is neighbor of a and b, distinct from v,a,b,c
                        xs = (na & nb) - {v, a, b, c}
                        # y is neighbor of a and c, distinct from v,a,b,c
                        ys = (na & nc) - {v, a, b, c}
                        # z is neighbor of b and c, distinct from v,a,b,c
                        zs = (nb & nc) - {v, a, b, c}
                        
                        # We need distinct x, y, z
                        for x in xs:
                            for y in ys:
                                if x == y:
                                    continue
                                for z in zs:
                                    if x == z or y == z:
                                        continue
                                    
                                    nx = graph_adj[x]
                                    ny = graph_adj[y]
                                    nz = graph_adj[z]
                                    
                                    # w must be neighbor of x, y, z
                                    ws = (nx & ny & nz) - {v, a, b, c, x, y, z}
                                    
                                    for w in ws:
                                        q3_edges = [
                                            (v, a), (v, b), (v, c),
                                            (a, x), (b, x),
                                            (a, y), (c, y),
                                            (b, z), (c, z),
                                            (x, w), (y, w), (z, w)
                                        ]
                                        return q3_edges
        return None

    def repair_and_optimize(current_edges, adj):
        """
        1. Repair: Remove edges to eliminate Q3s.
        2. Optimize: Add edges (1-opt) and Swap edges (2-opt).
        """
        # Phase 1: Repair
        while time.time() < deadline - 5:
            q3_edges = find_q3(adj)
            if q3_edges is None:
                break
            
            # Heuristic: Remove the edge in the Q3 with the lowest degree in the current graph
            min_deg_edge = None
            min_deg = float('inf')
            for u, v in q3_edges:
                deg = len(adj[u]) + len(adj[v])
                if deg < min_deg:
                    min_deg = deg
                    min_deg_edge = (u, v)
            
            if min_deg_edge in current_edges:
                u, v = min_deg_edge
                current_edges.remove(min_deg_edge)
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                break

        # Phase 2: 1-Opt (Add edges)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            missing = [e for e in all_edges if e not in current_edges]
            random.shuffle(missing)
            
            for e in missing:
                if time.time() > deadline - 5:
                    break
                u, v = e
                current_edges.add(e)
                adj[u].add(v)
                adj[v].add(u)
                
                if find_q3(adj) is None:
                    improved = True
                else:
                    current_edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
        
        # Phase 3: 2-Opt (Swap edges)
        max_swaps = 10
        for _ in range(max_swaps):
            if time.time() > deadline - 5:
                break
            
            existing = list(current_edges)
            missing = [e for e in all_edges if e not in current_edges]
            if not existing or not missing:
                break
            
            random.shuffle(existing)
            random.shuffle(missing)
            
            swap_found = False
            for e_out in existing:
                if time.time() > deadline - 5:
                    break
                u1, v1 = e_out
                for e_in in missing:
                    if time.time() > deadline - 5:
                        break
                    if e_in == e_out:
                        continue
                    u2, v2 = e_in
                    
                    current_edges.remove(e_out)
                    current_edges.add(e_in)
                    
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    
                    if find_q3(adj) is None:
                        swap_found = True
                        break
                    else:
                        current_edges.remove(e_in)
                        current_edges.add(e_out)
                        adj[u2].remove(v2)
                        adj[v2].remove(u2)
                        adj[u1].add(v1)
                        adj[v1].add(u1)
                
                if swap_found:
                    break
            
            if not swap_found:
                break

        return current_edges, adj

    best_edges = []
    best_count = 0
    random.seed(42)
    
    densities = [0.7, 0.6, 0.8, 0.5, 0.65, 0.75]
    
    for restart in range(len(densities)):
        if time.time() > deadline - 20:
            break
        
        p = densities[restart]
        current_edges = set()
        for e in all_edges:
            if random.random() < p:
                current_edges.add(e)
                
        adj = build_adj(current_edges)
        
        current_edges, adj = repair_and_optimize(current_edges, adj)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
    return [[u, v] for u, v in best_edges]
