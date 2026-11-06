import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    
    # Precompute all 4-cycles in K_n
    # There are 3 distinct 4-cycles for each set of 4 vertices.
    all_cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        all_cycles.append((a, b, c, d))
        all_cycles.append((a, b, d, c))
        all_cycles.append((a, c, b, d))
    
    num_cycles = len(all_cycles)
    
    # Precompute for each edge (u,v), the list of (cycle_index, position) where that edge appears
    # position: 0=u-v, 1=v-x, 2=x-y, 3=y-u in cycle (u,v,x,y)
    edge_to_cycles = {}
    for ci, cycle in enumerate(all_cycles):
        u, v, x, y = cycle
        for pos, (a, b) in enumerate([(u,v), (v,x), (x,y), (y,u)]):
            key = (min(a, b), max(a, b))
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append((ci, pos))
    
    def count_bad_cycles(adj):
        count = 0
        for cycle in all_cycles:
            u, v, x, y = cycle
            if adj[u][v] == -1 or adj[v][x] == -1 or adj[x][y] == -1 or adj[y][u] == -1:
                continue
            c1 = adj[u][v]
            c2 = adj[v][x]
            c3 = adj[x][y]
            c4 = adj[y][u]
            if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                count += 1
        return count

    def get_bad_cycles(adj):
        bad = []
        for ci, cycle in enumerate(all_cycles):
            u, v, x, y = cycle
            if adj[u][v] == -1 or adj[v][x] == -1 or adj[x][y] == -1 or adj[y][u] == -1:
                continue
            c1 = adj[u][v]
            c2 = adj[v][x]
            c3 = adj[x][y]
            c4 = adj[y][u]
            if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                bad.append(ci)
        return bad

    def edge_count(adj):
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    count += 1
        return count

    def make_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    edges.append([u, v, adj[u][v]])
        return edges

    best_edges = [[0, v, 0] for v in range(1, n)]  # Star fallback
    best_count = len(best_edges)
    
    def solve_attempt(adj_init):
        adj = [row[:] for row in adj_init]
        
        # Greedy phase: while there are bad cycles, remove the edge in the most bad cycles
        # but first try recoloring
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            bad_indices = get_bad_cycles(adj)
            if not bad_indices:
                break
            
            # Count how many bad cycles each edge participates in
            edge_bad_count = {}
            for ci in bad_indices:
                u, v, x, y = all_cycles[ci]
                for (a, b) in [(u,v), (v,x), (x,y), (y,u)]:
                    key = (min(a, b), max(a, b))
                    edge_bad_count[key] = edge_bad_count.get(key, 0) + 1
            
            # Pick the edge with the most bad cycles
            worst_edge = None
            max_bad = 0
            for key, cnt in edge_bad_count.items():
                if cnt > max_bad:
                    max_bad = cnt
                    worst_edge = key
            
            if worst_edge is None:
                break
            
            a, b = worst_edge
            old_c = adj[a][b]
            
            # Try to recolor this edge to fix as many bad cycles as possible
            best_new_c = None
            best_delta = 0  # We want to minimize bad cycles, so delta < 0 is good
            
            # Get all cycles involving this edge
            cycles_for_edge = edge_to_cycles.get(worst_edge, [])
            
            for new_c in range(n):
                if new_c == old_c:
                    continue
                
                adj[a][b] = new_c
                adj[b][a] = new_c
                
                # Evaluate delta: count bad cycles among those involving this edge
                new_bad = 0
                for (ci, pos) in cycles_for_edge:
                    u, v, x, y = all_cycles[ci]
                    if adj[u][v] == -1 or adj[v][x] == -1 or adj[x][y] == -1 or adj[y][u] == -1:
                        continue
                    c1 = adj[u][v]
                    c2 = adj[v][x]
                    c3 = adj[x][y]
                    c4 = adj[y][u]
                    if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                        new_bad += 1
                
                # We need the old bad count for these cycles too
                adj[a][b] = old_c
                adj[b][a] = old_c
                
                old_bad = 0
                for (ci, pos) in cycles_for_edge:
                    u, v, x, y = all_cycles[ci]
                    if adj[u][v] == -1 or adj[v][x] == -1 or adj[x][y] == -1 or adj[y][u] == -1:
                        continue
                    c1 = adj[u][v]
                    c2 = adj[v][x]
                    c3 = adj[x][y]
                    c4 = adj[y][u]
                    if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                        old_bad += 1
                
                delta = new_bad - old_bad
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            # Apply best recoloring if it helps
            if best_new_c is not None and best_delta < 0:
                adj[a][b] = best_new_c
                adj[b][a] = best_new_c
            else:
                # No recoloring helps, remove the edge
                adj[a][b] = -1
                adj[b][a] = -1
        
        return adj

    # Define initializers
    def init_random():
        adj = [[-1]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                c = random.randint(0, n-1)
                adj[u][v] = c
                adj[v][u] = c
        return adj
    
    def init_mod_sum():
        adj = [[-1]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                c = (u + v) % n
                adj[u][v] = c
                adj[v][u] = c
        return adj
    
    def init_mod_prod():
        adj = [[-1]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                c = (u * v) % n
                adj[u][v] = c
                adj[v][u] = c
        return adj
    
    def init_structured(offsets):
        adj = [[-1]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                c = (offsets[u] + v) % n
                adj[u][v] = c
                adj[v][u] = c
        return adj
    
    def init_random_structured():
        offsets = list(range(n))
        random.shuffle(offsets)
        return init_structured(offsets)
    
    # Run multiple attempts with different initializers
    attempts = [
        init_mod_sum,
        init_mod_prod,
        init_random,
        init_random_structured,
        init_random_structured,
        init_random_structured,
        init_random_structured,
        init_random_structured,
    ]
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Cycle through initializers, with more random structured ones
        if attempt < 2:
            init_func = attempts[0]
        elif attempt < 4:
            init_func = attempts[1]
        elif attempt < 6:
            init_func = attempts[2]
        else:
            init_func = attempts[3 + (attempt % 5)]
        
        adj = init_func()
        adj = solve_attempt(adj)
        
        if time.time() > deadline:
            break
            
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_edges = make_edges(adj)
            
    return best_edges
