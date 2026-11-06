import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit (or 150s if that's the hard kill)
    
    # 1. Graph Generation (Exact match to verifier)
    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # The seed program loops 4*n*d times. We must replicate this exactly.
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    instances = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = get_instance(*spec)
        n = spec[0]
        # Build adjacency for compatibility checks
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute F(e): vertices that cannot be in the same induced matching as e
        # F(e) = V(e) U {w | exists u in V(e) s.t. (u,w) in E}
        # Actually, if w is in F(e), it means w is either an endpoint of e or adjacent to an endpoint.
        # If a vertex w in V(C) is in F(e), it implies either w is in V(e) (conflict) 
        # or w is adjacent to some u in V(e). If w is in V(C), w is an endpoint of some edge f in C.
        # So (u,w) is an edge in G connecting e and f. This violates induced matching.
        edge_F = {}
        for e in edges:
            u, v = e
            F = set()
            F.add(u)
            F.add(v)
            F.update(adj[u])
            F.update(adj[v])
            edge_F[e] = F
            
        instances.append({
            'n': n,
            'edges': edges,
            'edge_F': edge_F,
            'adj': adj
        })
        
    results = []
    
    def solve_instance(inst):
        n = inst['n']
        edges = inst['edges']
        edge_F = inst['edge_F']
        m = len(edges)
        
        if m == 0:
            return []
        
        # Greedy initialization with smart ordering
        # Sort edges by "constraint" descending: higher degree sum = harder to place
        edge_order = sorted(range(m), key=lambda i: len(inst['adj'][edges[i][0]]) + len(inst['adj'][edges[i][1]]), reverse=True)
        
        colour = [-1] * m
        class_verts = []  # class_verts[c] = set of vertices in colour class c
        
        for idx in edge_order:
            e = edges[idx]
            Fe = edge_F[e]
            placed = False
            for c in range(len(class_verts)):
                if class_verts[c].isdisjoint(Fe):
                    colour[idx] = c
                    class_verts[c].update(e)
                    placed = True
                    break
            if not placed:
                c = len(class_verts)
                colour[idx] = c
                class_verts.append(set(e))
        
        num_classes = len(class_verts)
        
        # Local search: try to reduce number of colours by relocating edges
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Find highest non-empty class
            max_c = -1
            for c in range(num_classes - 1, -1, -1):
                if any(colour[i] == c for i in range(m)):
                    max_c = c
                    break
            
            if max_c <= 0:
                break
            
            # Try to move edges from class max_c to lower classes
            edges_in_max = [i for i in range(m) if colour[i] == max_c]
            
            for idx in edges_in_max:
                e = edges[idx]
                Fe = edge_F[e]
                moved = False
                for c in range(max_c):
                    if class_verts[c].isdisjoint(Fe):
                        # Move edge idx from max_c to c
                        class_verts[max_c].discard(e[0])
                        class_verts[max_c].discard(e[1])
                        colour[idx] = c
                        class_verts[c].update(e)
                        moved = True
                        improved = True
                        break
                if moved:
                    break
            
            # If max_c is now empty, reduce num_classes
            if not any(colour[i] == max_c for i in range(m)):
                num_classes = max_c
        
        # Final result: colours follow sorted edge order
        res = [colour[i] for i in range(m)]
        return res

    # We need to actually implement the optimization properly.
    # The logic above in `solve_instance` was just planning.
    # I will write the full implementation in the final code block.
    
    # Placeholder for now to ensure structure is correct
    for inst in instances:
        results.append(solve_instance(inst))
        
    return results
