import random
import time
import itertools

def construct():
    deadline = time.time() + 120
    n = 20
    num_colors = 20
    
    best_edges = None
    best_count = 0
    
    def is_valid(edges):
        # Check every 4-cycle has 4 distinct colors
        adj = {}
        for u, v, c in edges:
            if u > v:
                u, v = v, u
            adj[(u, v)] = c
            adj[(v, u)] = c
        
        # Find all 4-cycles
        # A 4-cycle is u-a-b-v-u with all distinct
        # For each pair of vertices (u,v), find all paths u-a-b-v
        for u in range(n):
            for v in range(u+1, n):
                if (u,v) not in adj:
                    continue
                cu = adj[(u,v)]
                # Find all a neighbors of u, b neighbors of v, with (a,b) edge
                for a in range(n):
                    if a == u or a == v:
                        continue
                    if (u,a) not in adj:
                        continue
                    ca = adj[(u,a)]
                    if ca == cu:
                        # Can't form rainbow with this a if ca==cu, but we still need to check
                        # Actually we need all 4 distinct, so if ca==cu already invalid
                        # But let's just collect all C4s and check
                        pass
                    for b in range(n):
                        if b == u or b == v or b == a:
                            continue
                        if (b,v) not in adj:
                            continue
                        if (a,b) not in adj:
                            continue
                        cb = adj[(a,b)]
                        cv = adj[(b,v)]
                        # Check 4 distinct: cu, ca, cb, cv
                        if len({cu, ca, cb, cv}) < 4:
                            return False
        return True
    
    def greedy_build(order_seed):
        rng = random.Random(order_seed)
        # Generate all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(all_edges)
        
        # adj: (u,v) -> color, using sorted tuple
        adj = {}
        edges = []
        
        for (u, v) in all_edges:
            # Try each color
            placed = False
            for c in range(num_colors):
                # Check if adding (u,v,c) creates any non-rainbow C4
                ok = True
                # For each a neighbor of u, b neighbor of v, check if (a,b) is edge
                # and if u-a-b-v-u would be non-rainbow
                for a in range(n):
                    if a == u or a == v:
                        continue
                    if (min(u,a), max(u,a)) not in adj:
                        continue
                    ca = adj[(min(u,a), max(u,a))]
                    if ca == c:
                        # Already have same color, but need to check full C4
                        # Actually just continue checking, the full check will catch it
                        pass
                    for b in range(n):
                        if b == u or b == v or b == a:
                            continue
                        if (min(b,v), max(b,v)) not in adj:
                            continue
                        if (min(a,b), max(a,b)) not in adj:
                            continue
                        cb = adj[(min(a,b), max(a,b))]
                        cv = adj[(min(b,v), max(b,v))]
                        # C4: u-a-b-v-u, edges: (u,a)=ca, (a,b)=cb, (b,v)=cv, (v,u)=c
                        if len({ca, cb, cv, c}) < 4:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    adj[(min(u,v), max(u,v))] = c
                    edges.append([u, v, c])
                    placed = True
                    break
            # If no color works, skip this edge
        
        return edges
    
    # Multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            # Verify
            if not is_valid(best_edges):
                best_count = 0
                best_edges = None
    
    if best_edges is None:
        # Fallback: star
        best_edges = [[0, v, 0] for v in range(1, 20)]
    
    return best_edges
