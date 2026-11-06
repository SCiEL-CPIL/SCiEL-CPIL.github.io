import time

def construct():
    for n in range(17, 0, -1):
        start_time = time.time()
        result = solve_backtracking(n, start_time)
        if result is not None:
            return {'n': n, 'colors': result}
            
    return {'n': 1, 'colors': []}

def solve_backtracking(n, start_time):
    colors = [[-1]*n for _ in range(n)]
    adj = [[set() for _ in range(n)] for _ in range(4)]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
            
    def is_valid_addition(u, v, c):
        # Check if adding (u, v) to color c creates a C6
        # Path v-a-b-d-e-u in adj[c]
        # a,b,d,e distinct, not u, not v.
        
        # Neighbors of v in adj[c]
        N_v = [x for x in adj[c][v] if x != u]
        for a in N_v:
            # Neighbors of a in adj[c], not v, not u
            N_a = [x for x in adj[c][a] if x != v and x != u]
            for b in N_a:
                if b == a: continue
                # Neighbors of b in adj[c], not v, not u, not a
                N_b = [x for x in adj[c][b] if x != v and x != u and x != a]
                for d in N_b:
                    if d == a or d == b: continue
                    # Neighbors of d in adj[c], not v, not u, not a, not b
                    N_d = [x for x in adj[c][d] if x != v and x != u and x != a and x != b]
                    for e in N_d:
                        if e == a or e == b or e == d: continue
                        # Check if e is connected to u in adj[c]
                        if u in adj[c][e]:
                            return True
        return False

    def backtrack(idx):
        if time.time() - start_time > 110:
            return False
            
        if idx == len(edges):
            return True
        
        u, v = edges[idx]
        
        if idx == 0:
            candidates = [0]
        elif idx == 1:
            candidates = [0, 1]
        else:
            candidates = [0, 1, 2, 3]
            
        for c in candidates:
            if is_valid_addition(u, v, c):
                colors[u][v] = c
                colors[v][u] = c
                adj[c][u].add(v)
                adj[c][v].add(u)
                
                if backtrack(idx + 1):
                    return True
                
                colors[u][v] = -1
                colors[v][u] = -1
                adj[c][u].remove(v)
                adj[c][v].remove(u)
        
        return False

    if backtrack(0):
        res = []
        for i in range(n):
            for j in range(i+1, n):
                res.append(colors[i][j])
        return res
    return None
