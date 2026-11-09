import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 16
    
    def count_edges(g):
        return sum(sum(row) for row in g) // 2

    def find_q3(g):
        """
        Returns a tuple (u, v) of an edge that is part of a Q3, or None.
        """
        for u in range(n):
            for v in range(u+1, n):
                if g[u][v] == 0:
                    continue
                
                N_u = [x for x in range(n) if g[u][x] == 1 and x != v]
                N_v = [x for x in range(n) if g[v][x] == 1 and x != u]
                
                # We need d, e in N_u and f, g in N_v
                # such that (f,d) and (g,e) are edges
                # and there exist h, i distinct from {u,v,d,e,f,g}
                # such that h in N(d) cap N(e) cap N(i)
                # i in N(f) cap N(g) cap N(h)
                # and (h,i) is an edge
                
                # To speed up, we can iterate d, f first
                for d in N_u:
                    for f in N_v:
                        if d == f: continue
                        if g[f][d] == 0: continue
                        
                        # Now we have u-v, u-d, v-f, f-d
                        # We need e in N_u \ {d, f, v, u}
                        # and g in N_v \ {f, d, u, v}
                        # such that (g,e) is an edge
                        
                        for e in N_u:
                            if e == d or e == f: continue
                            for g_v in N_v:
                                if g_v == f or g_v == d: continue
                                if g[g_v][e] == 0: continue
                                
                                # Vertices so far: u, v, d, e, f, g_v
                                # Excluded: {u, v, d, e, f, g_v}
                                # Need i in N(f) cap N(g_v) \ Excluded
                                # Need h in N(d) cap N(e) \ Excluded
                                # Need (h, i) in G
                                
                                C_i = []
                                for i in range(n):
                                    if i in (u, v, d, e, f, g_v): continue
                                    if g[f][i] == 1 and g[g_v][i] == 1:
                                        C_i.append(i)
                                if not C_i: continue
                                
                                C_h = []
                                for h in range(n):
                                    if h in (u, v, d, e, f, g_v): continue
                                    if g[d][h] == 1 and g[e][h] == 1:
                                        C_h.append(h)
                                if not C_h: continue
                                
                                for i in C_i:
                                    for h in C_h:
                                        if g[h][i] == 1:
                                            return (u, v)
        return None

    def can_add(g, u, v):
        if g[u][v] == 1:
            return True
            
        N_u = [x for x in range(n) if g[u][x] == 1 and x != v]
        N_v = [x for x in range(n) if g[v][x] == 1 and x != u]
        
        for d in N_u:
            for f in N_v:
                if d == f: continue
                if g[f][d] == 0: continue
                
                for e in N_u:
                    if e == d or e == f: continue
                    for g_v in N_v:
                        if g_v == f or g_v == d: continue
                        if g[g_v][e] == 0: continue
                        
                        C_i = []
                        for i in range(n):
                            if i in (u, v, d, e, f, g_v): continue
                            if g[f][i] == 1 and g[g_v][i] == 1:
                                C_i.append(i)
                        if not C_i: continue
                        
                        C_h = []
                        for h in range(n):
                            if h in (u, v, d, e, f, g_v): continue
                            if g[d][h] == 1 and g[e][h] == 1:
                                C_h.append(h)
                        if not C_h: continue
                        
                        for i in C_i:
                            for h in C_h:
                                if g[h][i] == 1:
                                    return False
        return True

    best_edges = 0
    best_graph = None
    
    while time.time() < deadline:
        # Start with complete graph
        g = [[1]*n for _ in range(n)]
        for i in range(n):
            g[i][i] = 0
            
        # Remove edges to make Q3-free
        # Randomize removal order to escape local minima
        while True:
            edge = find_q3(g)
            if edge is None:
                break
            u, v = edge
            g[u][v] = 0
            g[v][u] = 0
            
        # Add edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if g[i][j] == 0]
        random.shuffle(missing)
        
        for u, v in missing:
            if can_add(g, u, v):
                g[u][v] = 1
                g[v][u] = 1
                
        curr_edges = count_edges(g)
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_graph = [row[:] for row in g]
            
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i][j] == 1:
                result.append([i, j])
                
    return result
