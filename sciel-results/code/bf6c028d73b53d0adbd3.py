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
                cxy = colour_of[x][y]
                cyu = colour_of[y][u]
                
                if c == cvx or c == cxy or c == cyu:
                    return False
                if cvx == cxy or cvx == cyu:
                    return False
                if cxy == cyu:
                    return False
    return True

def _greedy_fill(adj_mat, adj_list, colour_of, placed, rng, exclude_edge=None):
    """Greedily try to add all remaining edges. Returns updated structures."""
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)
                       if not adj_mat[i][j]]
    rng.shuffle(potential_edges)
    
    for (u, v) in potential_edges:
        if exclude_edge is not None and {u, v} == exclude_edge:
            continue
        cols = list(range(C))
        rng.shuffle(cols)
        
        for c in cols:
            if _edge_ok(adj_mat, adj_list, colour_of, u, v, c):
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_mat[u][v] = True
                adj_mat[v][u] = True
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    return adj_mat, adj_list, colour_of, placed

def _greedy_construct(seed):
    """Run a single greedy construction pass."""
    adj_list = [[] for _ in range(N)]
    adj_mat = [[False] * N for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    rng = random.Random(seed)
    
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(potential_edges)
    
    for (u, v) in potential_edges:
        cols = list(range(C))
        rng.shuffle(cols)
        
        for c in cols:
            if _edge_ok(adj_mat, adj_list, colour_of, u, v, c):
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_mat[u][v] = True
                adj_mat[v][u] = True
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    return placed

def _local_search(edges, seed, time_budget):
    """Remove a random edge and re-fill greedily. Return improved edges if found."""
    if len(edges) == 0:
        return edges
    
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Rebuild structures from edges
    adj_list = [[] for _ in range(N)]
    adj_mat = [[False] * N for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    for e in edges:
        u, v, c = e
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_mat[u][v] = True
        adj_mat[v][u] = True
        colour_of[u][v] = c
        colour_of[v][u] = c
    
    best = edges[:]
    best_count = len(edges)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Pick a random edge to remove
        idx = rng.randint(0, len(best) - 1)
        u, v, c = best[idx]
        
        # Remove it
        adj_list[u].remove(v)
        adj_list[v].remove(u)
        adj_mat[u][v] = False
        adj_mat[v][u] = False
        colour_of[u][v] = -1
        colour_of[v][u] = -1
        current_edges = [e for i, e in enumerate(best) if i != idx]
        
        # Re-fill greedily
        _, _, _, current_edges = _greedy_fill(
            adj_mat, adj_list, colour_of, current_edges, rng
        )
        
        if len(current_edges) > best_count:
            best = current_edges
            best_count = len(current_edges)
            # Rebuild structures
            adj_list = [[] for _ in range(N)]
            adj_mat = [[False] * N for _ in range(N)]
            colour_of = [[-1] * N for _ in range(N)]
            for e in best:
                eu, ev, ec = e
                adj_list[eu].append(ev)
                adj_list[ev].append(eu)
                adj_mat[eu][ev] = True
                adj_mat[ev][eu] = True
                colour_of[eu][ev] = ec
                colour_of[ev][eu] = ec
        else:
            # Restore the removed edge
            adj_list[u].append(v)
            adj_list[v].append(u)
            adj_mat[u][v] = True
            adj_mat[v][u] = True
            colour_of[u][v] = c
            colour_of[v][u] = c
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin under 120s limit
    
    best_edges = []
    max_edges = 0
    
    # Phase 1: Multiple greedy restarts
    seed = 42
    while time.time() < deadline - 30.0:
        edges = _greedy_construct(seed)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline:
        remaining = deadline - time.time() - 5.0
        if remaining > 10.0:
            improved = _local_search(best_edges, seed, remaining)
            if len(improved) > max_edges:
                best_edges = improved
                max_edges = len(best_edges)
    
    return best_edges
