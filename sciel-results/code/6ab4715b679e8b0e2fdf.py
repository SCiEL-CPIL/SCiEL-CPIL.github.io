import random
import time

N = 20
C = 20

def _edge_ok(adj_mat, adj_list, colour_of, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
    adj_v = adj_list[v]
    adj_u = adj_list[u]
    
    for x in adj_v:
        if x == u:
            continue
        cvx = colour_of[v][x]
        
        if cvx == c:
            # If cvx == c, any 4-cycle u-v-x-y-u would have c appearing twice
            # So we must ensure no such y exists
            for y in adj_u:
                if y == v or y == x:
                    continue
                if adj_mat[x][y]:
                    return False
            continue

        for y in adj_u:
            if y == v or y == x:
                continue
            if adj_mat[x][y]:
                # Cycle u-v-x-y-u exists
                cxy = colour_of[x][y]
                cyu = colour_of[y][u]
                
                # Check distinctness: c, cvx, cxy, cyu must all be different
                if c == cvx or c == cxy or c == cyu:
                    return False
                if cvx == cxy or cvx == cyu:
                    return False
                if cxy == cyu:
                    return False
    return True

def _greedy_construct(seed, deadline):
    """Run a single greedy construction pass with random edge and color ordering."""
    adj_list = [set() for _ in range(N)]
    adj_mat = [[False] * N for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = random.Random(seed)
    rng.shuffle(potential_edges)
    
    for (u, v) in potential_edges:
        if time.time() > deadline:
            break
        
        cols = list(range(C))
        rng.shuffle(cols)
        
        for c in cols:
            if _edge_ok(adj_mat, adj_list, colour_of, u, v, c):
                adj_list[u].add(v)
                adj_list[v].add(u)
                adj_mat[u][v] = True
                adj_mat[v][u] = True
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    return placed

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin under 120s limit
    
    best_edges = []
    max_edges = 0
    
    seed = 42
    while time.time() < deadline:
        edges = _greedy_construct(seed, deadline)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
    
    return best_edges
