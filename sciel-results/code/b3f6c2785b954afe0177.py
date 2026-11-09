import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_graph(n, p, seed):
        edges = instance(n, p, seed)
        edge_set = set(edges)
        
        # Precompute adjacency for fast neighbor checks
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find all triangles
        triangles = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u].intersection(adj[v])
                    for w in common:
                        if u < w and v < w:
                            triangles.append((u, v, w))
                            
        # Find all K4s
        k4s = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common_uv = adj[u].intersection(adj[v])
                    for w in common_uv:
                        if u < w and v < w:
                            # Check if w connects to any other common neighbor? 
                            # No, for K4 we need 4 vertices. 
                            # Let's just check all 4-combinations of neighbors? Too slow.
                            # Alternative: For each triangle (u,v,w), check if there is x connected to u,v,w.
                            pass
        
        # Efficient K4 detection:
        # For each triangle (u, v, w), check if there is a vertex x such that 
        # x is connected to u, v, and w.
        # We can precompute common neighbors for pairs.
        # Actually, let's just use the triangles list.
        # For each triangle (u, v, w), find x in adj[u] & adj[v] & adj[w]
        # If x > max(u,v,w), then (u,v,w,x) is a K4.
        
        k4_candidates = []
        for t in triangles:
            u, v, w = t
            common_3 = adj[u].intersection(adj[v]).intersection(adj[w])
            for x in common_3:
                if x > max(u, v, w):
                    k4_candidates.append((u, v, w, x))
                    
        # Remove duplicate K4s (though the condition x > max ensures uniqueness of representation)
        # Each K4 is generated exactly 4 times? No.
        # If we have K4 {a,b,c,d}, the triangles are (a,b,c), (a,b,d), (a,c,d), (b,c,d).
        # For (a,b,c), common_3 includes d. If d > max(a,b,c), we add (a,b,c,d).
        # For (a,b,d), common_3 includes c. If c > max(a,b,d), we add (a,b,d,c) -> sorted?
        # We should sort the tuple to ensure uniqueness if we were using a set.
        # But with x > max, each K4 is found exactly once (from the triangle of the 3 smallest vertices).
        
        # Greedy Packing
        best_cliques = None
        best_count = float('inf')
        
        # Number of restarts depends on time
        # Estimate time per restart. n=64, E~1000, Tri~2000, K4~100.
        # Very fast. We can do many restarts.
        num_restarts = 200
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Shuffle K4s and Triangles
            shuffled_k4s = k4_candidates[:]
            random.shuffle(shuffled_k4s)
            shuffled_tris = triangles[:]
            random.shuffle(shuffled_tris)
            
            used_edges = set()
            cliques = []
            
            # 1. Pick K4s
            for k4 in shuffled_k4s:
                u, v, w, x = k4
                e1 = (min(u,v), max(u,v))
                e2 = (min(u,w), max(u,w))
                e3 = (min(u,x), max(u,x))
                e4 = (min(v,w), max(v,w))
                e5 = (min(v,x), max(v,x))
                e6 = (min(w,x), max(w,x))
                es = {e1, e2, e3, e4, e5, e6}
                if not used_edges.intersection(es):
                    used_edges.update(es)
                    cliques.append([u, v, w, x])
                    
            # 2. Pick Triangles
            for tri in shuffled_tris:
                u, v, w = tri
                e1 = (min(u,v), max(u,v))
                e2 = (min(u,w), max(u,w))
                e3 = (min(v,w), max(v,w))
                es = {e1, e2, e3}
                if not used_edges.intersection(es):
                    used_edges.update(es)
                    cliques.append([u, v, w])
                    
            # 3. Pick remaining Edges
            for e in edges:
                if e not in used_edges:
                    used_edges.add(e)
                    cliques.append([e[0], e[1]])
                    
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
                
        return best_cliques

    g1 = solve_graph(32, 0.6, 601)
    if time.time() > deadline - 5.0:
         # Fallback if time is tight? Just return what we have or simple edges
         # But solve_graph has internal guard.
         pass
         
    g2 = solve_graph(64, 0.55, 602)
    
    return [g1, g2]
