import time
import random
import math

def construct():
    deadline = time.time() + 140.0
    random.seed(42)

    def get_seed(n):
        # K11 split into five Hamiltonian cycles by circular distance 1..5
        return [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]

    best_n = 11
    best_colors = get_seed(11)

    def build_structures(colors, n):
        adj = [[set() for _ in range(n)] for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        common_counts = [[None] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                cc = [0] * 5
                for c in range(5):
                    nu = adj[c][u]
                    nv = adj[c][v]
                    cc[c] = len(nu & nv)
                common_counts[u][v] = cc
                common_counts[v][u] = cc
        return adj, common_counts

    def verify(colors, n):
        adj, _ = build_structures(colors, n)
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if len(adj[c][u] & adj[c][v]) >= 2:
                        return False
        return True

    adj, common_counts = build_structures(best_colors, best_n)

    def try_extend(n_curr, adj, common_counts, colors, deadline):
        n_new = n_curr + 1
        new_edge_colors = [random.randint(0, 4) for _ in range(n_curr)]
        
        def calc_cost(new_edge_colors):
            cost = 0
            N_new = [set() for _ in range(5)]
            for i in range(n_curr):
                N_new[new_edge_colors[i]].add(i)
                
            for u in range(n_curr):
                for v in range(u + 1, n_curr):
                    cu = new_edge_colors[u]
                    cv = new_edge_colors[v]
                    if cu == cv:
                        c = cu
                        if common_counts[u][v][c] == 1:
                            cost += 1
                            
            for u in range(n_curr):
                for c in range(5):
                    inter = N_new[c] & adj[c][u]
                    k = len(inter)
                    if k >= 2:
                        cost += math.comb(k, 2)
            return cost

        current_cost = calc_cost(new_edge_colors)
        if current_cost == 0:
            new_colors = colors + new_edge_colors
            new_adj, new_common_counts = build_structures(new_colors, n_new)
            return True, new_colors, new_adj, new_common_counts

        best_cost = current_cost
        best_colors_local = new_edge_colors.copy()
        start_time = time.time()
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline and time.time() - start_time < 2.0:
            if no_improve > max_no_improve:
                break
            i = random.randint(0, n_curr - 1)
            old_c = new_edge_colors[i]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            new_edge_colors[i] = new_c
            new_cost = calc_cost(new_edge_colors)
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors_local = new_edge_colors.copy()
                no_improve = 0
                if best_cost == 0:
                    new_colors = colors + best_colors_local
                    if verify(new_colors, n_new):
                        new_adj, new_common_counts = build_structures(new_colors, n_new)
                        return True, new_colors, new_adj, new_common_counts
                    else:
                        # Should not happen if cost is 0, but just in case
                        return False, None, None, None
            else:
                new_edge_colors[i] = old_c
                no_improve += 1
        return False, None, None, None

    for n_target in range(12, 29):
        if time.time() > deadline - 10:
            break
        success = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            success, new_colors, new_adj, new_common_counts = try_extend(
                best_n, adj, common_counts, best_colors, deadline - 5
            )
            if success:
                best_n = n_target
                best_colors = new_colors
                adj = new_adj
                common_counts = new_common_counts
                break
        if not success:
            break

    if not verify(best_colors, best_n):
        # Fallback to seed
        best_n = 11
        best_colors = get_seed(11)

    return {'n': best_n, 'colors': best_colors}
