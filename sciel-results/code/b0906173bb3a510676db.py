import random
import time

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # Seed solution for n=9
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_n = n_seed
    best_colors = colors_seed
    
    try:
        def get_edges(n):
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        def solve_n(n):
            edges = get_edges(n)
            colors = {}
            for e in edges:
                colors[e] = random.randint(0, 3)
                
            neighbors = [[set() for _ in range(n)] for _ in range(4)]
            for (u, v), c in colors.items():
                neighbors[c][u].add(v)
                neighbors[c][v].add(u)
                
            def count_paths_4(u, v, c):
                N_u = neighbors[c][u]
                N_v = neighbors[c][v]
                count = 0
                for x in N_u:
                    if x == v: continue
                    for z in N_v:
                        if z == u or z == x: continue
                        common = neighbors[c][x].intersection(neighbors[c][z])
                        for y in common:
                            if y != u and y != v:
                                count += 1
                return count

            def get_delta_e(u, v, c_old, c_new):
                e_old = count_paths_4(u, v, c_old)
                e_new = count_paths_4(u, v, c_new)
                return e_new - e_old

            E = 0
            for u in range(n):
                for v in range(u + 1, n):
                    c = colors[(u, v)]
                    E += count_paths_4(u, v, c)
                    
            best_E = E
            best_colors_local = colors.copy()
            
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            while time.time() - start_time < time_limit - 5:
                if E == 0:
                    break
                    
                u, v = random.choice(edges)
                c_old = colors[(u, v)]
                c_new = random.choice([c for c in range(4) if c != c_old])
                
                delta = get_delta_e(u, v, c_old, c_new)
                
                if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 1e-9)):
                    neighbors[c_old][u].remove(v)
                    neighbors[c_old][v].remove(u)
                    neighbors[c_new][u].add(v)
                    neighbors[c_new][v].add(u)
                    
                    colors[(u, v)] = c_new
                    E += delta
                    
                    if E < best_E:
                        best_E = E
                        best_colors_local = colors.copy()
                    
                T *= cooling
                if T < min_T:
                    T = min_T
            
            if best_E == 0:
                res_colors = [best_colors_local[(i, j)] for i in range(n) for j in range(i + 1, n)]
                return n, res_colors
            else:
                return None, None

        for n in range(10, 18):
            if time.time() - start_time > time_limit - 10:
                break
            
            res_n, res_colors = solve_n(n)
            if res_n is not None:
                best_n = res_n
                best_colors = res_colors
                
    except Exception as e:
        # If anything fails, return the seed
        pass

    return {'n': best_n, 'colors': best_colors}
