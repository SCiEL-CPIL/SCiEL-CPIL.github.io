"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict_graph(edges, n):
        """Build conflict graph where edges conflict if they share a vertex or are joined by an edge."""
        m = len(edges)
        edge_set = set(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # For each edge, find all edges that conflict with it
        conflict = [set() for _ in range(m)]
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        for i, (u, v) in enumerate(edges):
            # Edges sharing vertex u or v
            for w in adj.get(u, set()):
                for x in adj.get(w, set()):
                    e2 = (min(w, x), max(w, x))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflict[i].add(j)
                            conflict[j].add(i)
            
            # Also need to check: edges that are joined by an edge
            # Edge (u,v) is joined to edge (a,b) if (u,a), (u,b), (v,a), (v,b) is an edge
            # But we already covered sharing vertices above.
            # "Joined by an edge" means there's a path of length 2 between them through an edge
            # i.e., an edge (p,q) where p is in {u,v} and q is in {a,b}
            # This is exactly what the above loop does: for each neighbor w of u, for each neighbor x of w
            # that gives edge (w,x). But we need to be more careful.
            
            # Let me reconsider: edge e1=(u,v) conflicts with edge e2=(a,b) if:
            # 1. They share a vertex: {u,v} ∩ {a,b} ≠ ∅
            # 2. They are joined by an edge: ∃ edge (p,q) in E such that p∈{u,v}, q∈{a,b}
            #    and (p,q) is not e1 or e2 themselves.
            # Actually, "joined by an edge" typically means there's an edge connecting them, 
            # i.e., an edge (u,a) or (u,b) or (v,a) or (v,b) exists.
            
            # My loop above: for w in adj[u], for x in adj[w], edge (w,x)
            # This finds edges at distance 2 from u via w. But I need edges at distance 1 from {u,v}.
            # Let me redo this more carefully.
        
        # Redo: for each edge (u,v), conflicting edges are those (a,b) where
        # {u,v} ∩ {a,b} ≠ ∅ OR (u,a)∈E OR (u,b)∈E OR (v,a)∈E OR (v,b)∈E
        conflict = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            for j, (a, b) in enumerate(edges):
                if i == j:
                    continue
                # Check if they share a vertex
                if u == a or u == b or v == a or v == b:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if joined by an edge
                if (min(u,a),max(u,a)) in edge_set or (min(u,b),max(u,b)) in edge_set or \
                   (min(v,a),max(v,a)) in edge_set or (min(v,b),max(v,b)) in edge_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict
    
    def dsatur_color(conflict, m):
        """Color using DSATUR heuristic."""
        color = [-1] * m
        # Track for each vertex: set of colors used by neighbors
        neighbor_colors = [set() for _ in range(m)]
        uncolored = list(range(m))
        
        for _ in range(m):
            if time.time() > deadline:
                break
            # Find vertex with highest saturation degree, break ties by degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                sat = len(neighbor_colors[v])
                deg = len(conflict[v])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            
            # Assign smallest available color
            used = neighbor_colors[best_v]
            c = 0
            while c in used:
                c += 1
            color[best_v] = c
            uncolored.remove(best_v)
            
            # Update neighbor colors
            for w in conflict[best_v]:
                if w in uncolored:
                    neighbor_colors[w].add(c)
        
        # If some vertices are uncolored (timeout), assign greedily
        for v in uncolored:
            used = neighbor_colors[v]
            c = 0
            while c in used:
                c += 1
            color[v] = c
            for w in conflict[v]:
                if w not in uncolored and color[w] != -1:
                    pass
                elif w in uncolored:
                    neighbor_colors[w].add(c)
        
        return color
    
    def greedy_color(conflict, m, order):
        """Color using greedy in given order."""
        color = [-1] * m
        neighbor_colors = [set() for _ in range(m)]
        for v in order:
            used = neighbor_colors[v]
            c = 0
            while c in used:
                c += 1
            color[v] = c
            for w in conflict[v]:
                neighbor_colors[w].add(c)
        return color
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n = spec[0]
        m = len(edges)
        conflict = build_conflict_graph(edges, n)
        
        # Try DSATUR
        best_color = dsatur_color(conflict, m)
        best_colors_used = max(best_color) + 1 if m > 0 else 0
        
        # Try a few random orderings with greedy
        rng = Random(42)
        for trial in range(20):
            if time.time() > deadline:
                break
            order = list(range(m))
            rng.shuffle(order)
            color = greedy_color(conflict, m, order)
            colors_used = max(color) + 1 if m > 0 else 0
            if colors_used < best_colors_used:
                best_colors_used = colors_used
                best_color = color
        
        out.append([best_color[i] for i in range(m)])
    
    return out
# EVOLVE-BLOCK-END
