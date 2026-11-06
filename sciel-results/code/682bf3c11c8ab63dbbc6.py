"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0
    N = 20
    C = 20
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Precompute all 4-cycles
    all_4cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    all_4cycles.append((a, b, c, d))
                    all_4cycles.append((a, b, d, c))
                    all_4cycles.append((a, c, b, d))
    
    # Precompute: for each edge (u,v), which 4-cycles contain it
    edge_to_cycles = {}
    for i, cycle in enumerate(all_4cycles):
        a, b, c, d = cycle
        for (e1, e2) in [(a,b), (b,c), (c,d), (d,a)]:
            key = (min(e1,e2), max(e1,e2))
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(i)
    
    def make_initial(density=0.5):
        col = [[-1]*N for _ in range(N)]
        for u in range(N):
            for v in range(u+1, N):
                if random.random() < density:
                    c = random.randint(0, C-1)
                    col[u][v] = c
                    col[v][u] = c
        return col
    
    def get_edges(col):
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if col[u][v] >= 0:
                    edges.append([u, v, col[u][v]])
        return edges
    
    def count_edges(col):
        return sum(1 for u in range(N) for v in range(u+1, N) if col[u][v] >= 0)
    
    def check_edge_violation(col, u, v, c):
        """Check if setting edge (u,v) to colour c creates any violation."""
        key = (min(u,v), max(u,v))
        if key not in edge_to_cycles:
            return False
        for idx in edge_to_cycles[key]:
            a, b, cc, d = all_4cycles[idx]
            colors = []
            for (e1, e2) in [(a,b), (b,cc), (cc,d), (d,a)]:
                if (e1 == u and e2 == v) or (e1 == v and e2 == u):
                    colors.append(c)
                else:
                    if col[e1][e2] < 0:
                        break
                    colors.append(col[e1][e2])
            else:
                if len(set(colors)) < 4:
                    return True
        return False
    
    def fix_violations(col, max_iters=5000):
        for _ in range(max_iters):
            if time_left() < 2.0:
                break
            found_cycle = None
            for cycle in all_4cycles:
                a, b, cc, d = cycle
                if col[a][b] >= 0 and col[b][cc] >= 0 and col[cc][d] >= 0 and col[d][a] >= 0:
                    colors = [col[a][b], col[b][cc], col[cc][d], col[d][a]]
                    if len(set(colors)) < 4:
                        found_cycle = cycle
                        break
            if found_cycle is None:
                break
            a, b, cc, d = found_cycle
            fixed = False
            for (u, v) in [(a,b), (b,cc), (cc,d), (d,a)]:
                if not fixed:
                    current_c = col[u][v]
                    for new_c in range(C):
                        if new_c == current_c:
                            continue
                        if not check_edge_violation(col, u, v, new_c):
                            col[u][v] = new_c
                            col[v][u] = new_c
                            fixed = True
                            break
            if not fixed:
                break
        return col
    
    def try_add_edges(col):
        improved = True
        while improved and time_left() > 2.0:
            improved = False
            for u in range(N):
                for v in range(u+1, N):
                    if col[u][v] >= 0:
                        continue
                    for c in range(C):
                        if not check_edge_violation(col, u, v, c):
                            col[u][v] = c
                            col[v][u] = c
                            improved = True
                            break
        return col
    
    best_col = None
    best_count = 0
    random.seed(42)
    
    densities = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for density in densities:
        if time_left() < 5.0:
            break
        col = make_initial(density)
        col = fix_violations(col)
        col = try_add_edges(col)
        cnt = count_edges(col)
        if cnt > best_count:
            best_count = cnt
            best_col = [row[:] for row in col]
    
    if best_col is not None and time_left() > 5.0:
        col = [row[:] for row in best_col]
        for _ in range(30):
            if time_left() < 3.0:
                break
            edges_list = get_edges(col)
            if not edges_list:
                break
            idx = random.randint(0, len(edges_list)-1)
            u, v, c = edges_list[idx]
            col[u][v] = -1
            col[v][u] = -1
            col = try_add_edges(col)
            cnt = count_edges(col)
            if cnt > best_count:
                best_count = cnt
                best_col = [row[:] for row in col]
    
    if best_col is None:
        return [[0, v, 0] for v in range(1, N)]
    
    return get_edges(best_col)
# EVOLVE-BLOCK-END
