import time
import random

def _is_valid_coloring(n, colors):
    if n < 2:
        return True
    adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    for c in range(5):
        for u in range(n):
            for v in range(u+1, n):
                if len(adj[c][u] & adj[c][v]) >= 2:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with n=1
    best_n = 1
    best_colors = []
    
    # Try to build up
    n = 1
    adj = [[set() for _ in range(n)] for _ in range(5)]
    colors = []
    
    def try_add(n, adj, colors):
        nonlocal deadline
        if time.time() > deadline:
            return None
        
        new_n = n + 1
        new_adj = [[set() for _ in range(new_n)] for _ in range(5)]
        for c in range(5):
            for u in range(n):
                new_adj[c][u] = set(adj[c][u])
        
        new_v = new_n - 1
        assigned = [-1] * n
        
        # Order: random shuffle for diversity
        order = list(range(n))
        random.shuffle(order)
        
        def bt(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                for u in range(n):
                    c = assigned[u]
                    new_adj[c][new_v].add(u)
                    new_adj[c][u].add(new_v)
                return True
            
            u = order[idx]
            # Try colors in random order
            color_order = list(range(5))
            random.shuffle(color_order)
            for c in color_order:
                ok = True
                for v in range(n):
                    if assigned[v] == c:
                        if adj[c][u] & adj[c][v]:
                            ok = False
                            break
                if ok:
                    assigned[u] = c
                    if bt(idx + 1):
                        return True
                    assigned[u] = -1
            return False
        
        if bt(0):
            new_colors = []
            for i in range(new_n):
                for j in range(i+1, new_n):
                    for c in range(5):
                        if j in new_adj[c][i]:
                            new_colors.append(c)
                            break
            return new_n, new_adj, new_colors
        return None
    
    # Try many times
    for attempt in range(500):
        if time.time() > deadline or best_n >= 29:
            break
        result = try_add(best_n, adj, best_colors)
        if result:
            best_n, adj, best_colors = result
        else:
            # Try with different random seed
            random.seed(attempt * 7919)
    
    return {'n': best_n, 'colors': best_colors}
