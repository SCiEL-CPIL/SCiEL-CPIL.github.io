import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 145.0

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges_total = len(all_edges)

    # For each vertex pair, precompute which other vertex pairs form 4-cycles with them
    # A 4-cycle u-v-w-x-u exists when w is neighbor of v, x is neighbor of u, w!=x, w!=u, x!=v, and edge (w,x) exists
    # We'll use a different approach: for each edge (u,v), maintain a bitmask of colors that would create conflicts

    def check_conflict_fast(u, v, c, edge_colors, adj):
        """
        Check if adding edge (u, v) with color c violates the rainbow 4-cycle property.
        Uses adj lists for speed.
        """
        for w in adj[v]:
            if w == u:
                continue
            col_vw = edge_colors.get((min(v, w), max(v, w)))
            if col_vw is None:
                continue
            for x in adj[u]:
                if x == v or x == w:
                    continue
                key_wx = (min(w, x), max(w, x))
                col_wx = edge_colors.get(key_wx)
                if col_wx is None:
                    continue
                col_xu = edge_colors.get((min(x, u), max(x, u)))
                if len({c, col_vw, col_wx, col_xu}) < 4:
                    return True
        return False

    def check_color_change_fast(u, v, new_c, edge_colors, adj):
        """
        Check if changing color of (u, v) to new_c creates a conflict.
        """
        for w in adj[v]:
            if w == u:
                continue
            col_vw = edge_colors.get((min(v, w), max(v, w)))
            if col_vw is None:
                continue
            for x in adj[u]:
                if x == v or x == w:
                    continue
                key_wx = (min(w, x), max(w, x))
                col_wx = edge_colors.get(key_wx)
                if col_wx is None:
                    continue
                col_xu = edge_colors.get((min(x, u), max(x, u)))
                if len({new_c, col_vw, col_wx, col_xu}) < 4:
                    return True
        return False

    def greedy_initial():
        adj_init = [[] for _ in range(n)]
        ec_init = {}
        edges_list = all_edges[:]
        random.shuffle(edges_list)
        for u, v in edges_list:
            if time.time() > deadline - 10:
                break
            for c in range(num_colors):
                if not check_conflict_fast(u, v, c, ec_init, adj_init):
                    ec_init[(u, v)] = c
                    adj_init[u].append(v)
                    adj_init[v].append(u)
                    break
        return adj_init, ec_init

    def run_sa(adj_sa, ec_sa, time_limit):
        current_edges = len(ec_sa)
        best_edges = current_edges
        best_adj = [list(lst) for lst in adj_sa]
        best_ec = dict(ec_sa)

        T = 2.0
        cooling = 0.999
        min_T = 0.005
        max_iter = 500000
        iter_count = 0

        while T > min_T and time.time() < time_limit and iter_count < max_iter:
            iter_count += 1
            move_type = random.random()

            if move_type < 0.5:
                # Add edge - try multiple missing edges
                missing = [e for e in all_edges if e not in ec_sa]
                if not missing:
                    continue
                # Try up to 3 random missing edges
                tried = 0
                for _ in range(3):
                    u, v = random.choice(missing)
                    c = random.randint(0, num_colors - 1)
                    if not check_conflict_fast(u, v, c, ec_sa, adj_sa):
                        ec_sa[(u, v)] = c
                        adj_sa[u].append(v)
                        adj_sa[v].append(u)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = [list(lst) for lst in adj_sa]
                            best_ec = dict(ec_sa)
                        break

            elif move_type < 0.65:
                # Remove edge (to allow re-adding with different color)
                if not ec_sa:
                    continue
                e = random.choice(list(ec_sa.keys()))
                u, v = e
                c = ec_sa[e]
                del ec_sa[e]
                adj_sa[u].remove(v)
                adj_sa[v].remove(u)
                current_edges -= 1
                # Accept worsening with probability based on temperature
                if random.random() < 0.1 + 0.3 * T:
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [list(lst) for lst in adj_sa]
                        best_ec = dict(ec_sa)
                else:
                    ec_sa[e] = c
                    adj_sa[u].append(v)
                    adj_sa[v].append(u)
                    current_edges += 1

            elif move_type < 0.95:
                # Change color of a random edge
                if not ec_sa:
                    continue
                e = random.choice(list(ec_sa.keys()))
                u, v = e
                old_c = ec_sa[e]
                new_c = random.randint(0, num_colors - 1)
                if new_c == old_c:
                    continue
                if not check_color_change_fast(u, v, new_c, ec_sa, adj_sa):
                    ec_sa[e] = new_c

            else:
                # Swap colors of two edges
                if len(ec_sa) < 2:
                    continue
                e1, e2 = random.sample(list(ec_sa.keys()), 2)
                c1 = ec_sa[e1]
                c2 = ec_sa[e2]
                if c1 == c2:
                    continue

                u1, v1 = e1
                u2, v2 = e2

                # Check if swapping is valid
                if check_color_change_fast(u1, v1, c2, ec_sa, adj_sa):
                    continue
                if check_color_change_fast(u2, v2, c1, ec_sa, adj_sa):
                    continue

                ec_sa[e1] = c2
                ec_sa[e2] = c1

            T *= cooling

        return best_adj, best_ec, best_edges

    # Multi-Start with varying parameters
    total_time = 140.0
    num_restarts = 8
    time_per_restart = total_time / num_restarts

    global_best_edges = 0
    global_best_ec = {}

    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break

        adj_r, ec_r = greedy_initial()
        best_adj_r, best_ec_r, best_edges_r = run_sa(adj_r, ec_r, time.time() + time_per_restart - 2)

        if best_edges_r > global_best_edges:
            global_best_edges = best_edges_r
            global_best_ec = best_ec_r

    result = []
    for (u, v), c in global_best_ec.items():
        result.append([u, v, c])

    return result
