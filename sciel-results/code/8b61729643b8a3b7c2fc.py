import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 135
    n = 20
    m = 10
    K = 20
    
    left = list(range(m))
    right = list(range(m, 2 * m))
    
    def build_c4free_bipartite():
        """Build a C4-free bipartite graph greedily."""
        adjL = [set() for _ in range(m)]
        adjR = [set() for _ in range(m)]
        edges = []
        all_edges = [(i, j) for i in range(m) for j in range(m)]
        random.shuffle(all_edges)
        
        for i, j in all_edges:
            if time.time() > deadline - 30:
                break
            creates_c4 = False
            for i_prime in adjR[j]:
                if adjL[i] & adjL[i_prime]:
                    creates_c4 = True
                    break
            if not creates_c4:
                adjL[i].add(j)
                adjR[j].add(i)
                edges.append([left[i], right[j], 0])  # placeholder color
        
        return adjL, adjR, edges
    
    def find_valid_color(adj, colors, u, v):
        """Find a color c for edge (u,v) such that all 4-cycles containing (u,v) are rainbow.
        Returns color or None."""
        for c in range(K):
            valid = True
            # 4-cycle: u - a - b - v - u
            # Edges: (u,v)=c, (u,a), (a,b), (b,v)
            for a in range(n):
                if a == u or a == v:
                    continue
                if a not in adj[u]:
                    continue
                ca = colors.get((min(u,a), max(u,a)))
                if ca is None:
                    continue
                for b in range(n):
                    if b == u or b == v or b == a:
                        continue
                    if b not in adj[a]:
                        continue
                    if v not in adj[b]:
                        continue
                    cab = colors.get((min(a,b), max(a,b)))
                    cbv = colors.get((min(b,v), max(b,v)))
                    if cab is None or cbv is None:
                        continue
                    if len({c, ca, cab, cbv}) < 4:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return c
        return None
    
    def is_fully_valid(edges_list):
        """Verify all 4-cycles in the graph are rainbow."""
        adj = {i: set() for i in range(n)}
        colors = {}
        for u, v, c in edges_list:
            adj[u].add(v)
            adj[v].add(u)
            colors[(min(u,v), max(u,v))] = c
        
        # Find all 4-cycles
        for a in range(n):
            for b in range(a + 1, n):
                if b not in adj[a]:
                    continue
                for c_ in range(b + 1, n):
                    if c_ not in adj[b]:
                        continue
                    for d in range(c_ + 1, n):
                        if d not in adj[c_]:
                            continue
                        if a not in adj[d]:
                            continue
                        # Check it's a proper 4-cycle (not a triangle with extra edge)
                        if b in adj[a] and c_ in adj[a]:
                            pass  # Could be K4, still a 4-cycle a-b-c_-d-a
                        ca = colors.get((min(a,b), max(a,b)))
                        cb = colors.get((min(b,c_), max(b,c_)))
                        cd = colors.get((min(c_,d), max(c_,d)))
                        da = colors.get((min(d,a), max(d,a)))
                        if ca is None or cb is None or cd is None or da is None:
                            continue
                        if len({ca, cb, cd, da}) < 4:
                            return False
        return True
    
    best_edges = None
    best_count = 0
    
    # Phase 1: Build C4-free bipartite backbone multiple times, keep best
    for attempt in range(20):
        if time.time() > deadline - 60:
            break
        
        adjL, adjR, bip_edges = build_c4free_bipartite()
        
        if len(bip_edges) <= best_count:
            continue
        
        # Build adjacency and colors for the bipartite graph
        adj = {i: set() for i in range(n)}
        colors = {}
        for u, v, c in bip_edges:
            adj[u].add(v)
            adj[v].add(u)
            colors[(min(u,v), max(u,v))] = c
        
        # Assign colors to bipartite edges (they're C4-free, so any colors work)
        # But we want to leave room for non-bipartite edges
        # Use a systematic coloring to maximize flexibility
        color_idx = 0
        for u, v, c in bip_edges:
            bip_edges[bip_edges.index([u, v, c])] = [u, v, color_idx % K]
            colors[(min(u,v), max(u,v))] = color_idx % K
            color_idx += 1
        
        # Phase 2: Try to add non-bipartite edges
        # Non-bipartite edges: within left, within right, or cross edges not in bipartite
        candidate_edges = []
        # Edges within left partition
        for i in range(m):
            for j in range(i + 1, m):
                candidate_edges.append((left[i], left[j]))
        # Edges within right partition
        for i in range(m):
            for j in range(i + 1, m):
                candidate_edges.append((right[i], right[j]))
        # Cross edges not already in bipartite
        for i in range(m):
            for j in range(m):
                u, v = left[i], right[j]
                if v not in adj[u]:
                    candidate_edges.append((u, v))
        
        random.shuffle(candidate_edges)
        
        for u, v in candidate_edges:
            if time.time() > deadline - 30:
                break
            if u == v:
                continue
            if v in adj[u]:
                continue
            
            c = find_valid_color(adj, colors, u, v)
            if c is not None:
                adj[u].add(v)
                adj[v].add(u)
                colors[(min(u,v), max(u,v))] = c
                bip_edges.append([u, v, c])
        
        if len(bip_edges) > best_count:
            best_count = len(bip_edges)
            best_edges = bip_edges[:]
    
    # Phase 3: If we have a good solution, try to improve by re-coloring
    if best_edges is not None and best_count > 0:
        # Verify the best solution
        if is_fully_valid(best_edges):
            # Try to add more edges
            adj = {i: set() for i in range(n)}
            colors = {}
            for u, v, c in best_edges:
                adj[u].add(v)
                adj[v].add(u)
                colors[(min(u,v), max(u,v))] = c
            
            remaining = []
            for i in range(n):
                for j in range(i + 1, n):
                    if j not in adj[i]:
                        remaining.append((i, j))
            
            random.shuffle(remaining)
            
            for u, v in remaining:
                if time.time() > deadline - 10:
                    break
                c = find_valid_color(adj, colors, u, v)
                if c is not None:
                    adj[u].add(v)
                    adj[v].add(u)
                    colors[(min(u,v), max(u,v))] = c
                    best_edges.append([u, v, c])
                    best_count += 1
        
        # Final verification
        if best_edges and not is_fully_valid(best_edges):
            # Fallback: just use the C4-free bipartite part
            bip_only = []
            for u, v, c in best_edges:
                if (u < m and v >= m) or (v < m and u >= m):
                    bip_only.append([u, v, c])
            if bip_only:
                best_edges = bip_only
                best_count = len(bip_only)
    
    # Ultimate fallback
    if best_edges is None or best_count == 0:
        best_edges = [[0, v, v % K] for v in range(1, n)]
        best_count = n - 1
    
    return best_edges
