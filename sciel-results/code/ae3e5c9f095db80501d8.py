import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 145.0

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def check_conflict(u, v, c, adj, edge_colors):
        """
        Check if adding edge (u, v) with color c violates the rainbow 4-cycle property.
        A 4-cycle involving (u, v) looks like u-v-w-x-u.
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

    def check_color_change(u, v, new_c, adj, edge_colors, old_c):
        """
        Check if changing color of (u, v) from old_c to new_c creates a conflict.
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
            valid = False
            for c in range(num_colors):
                if not check_conflict(u, v, c, adj_init, ec_init):
                    ec_init[(u, v)] = c
                    adj_init[u].append(v)
                    adj_init[v].append(u)
                    valid = True
                    break
        return adj_init, ec_init

    def run_sa(adj_sa, ec_sa, time_limit):
        current_edges = len(ec_sa)
        best_edges = current_edges
        best_adj = [list(lst) for lst in adj_sa]
        best_ec = dict(ec_sa)

        T = 1.0
        cooling = 0.9995
        min_T = 0.01

        while T > min_T and time.time() < time_limit:
            move_type = random.random()

            if move_type < 0.4:
                # Add edge
                missing = [e for e in all_edges if e not in ec_sa]
                if not missing:
                    continue
                u, v = random.choice(missing)
                c = random.randint(0, num_colors - 1)
                if not check_conflict(u, v, c, adj_sa, ec_sa):
                    ec_sa[(u, v)] = c
                    adj_sa[u].append(v)
                    adj_sa[v].append(u)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [list(lst) for lst in adj_sa]
                        best_ec = dict(ec_sa)
                # else reject

            elif move_type < 0.6:
                # Remove edge
                if not ec_sa:
                    continue
                e = random.choice(list(ec_sa.keys()))
                u, v = e
                c = ec_sa[e]
                del ec_sa[e]
                adj_sa[u].remove(v)
                adj_sa[v].remove(u)
                current_edges -= 1
                if random.random() < 0.05:  # Accept worsening move with low prob
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [list(lst) for lst in adj_sa]
                        best_ec = dict(ec_sa)
                else:
                    ec_sa[e] = c
                    adj_sa[u].append(v)
                    adj_sa[v].append(u)
                    current_edges += 1

            elif move_type < 0.9:
                # Change color
                if not ec_sa:
                    continue
                e = random.choice(list(ec_sa.keys()))
                u, v = e
                old_c = ec_sa[e]
                new_c = random.randint(0, num_colors - 1)
                if new_c == old_c:
                    continue
                if not check_color_change(u, v, new_c, adj_sa, ec_sa, old_c):
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

                # Check if swapping c1 and c2 is valid
                # We need to check conflicts for e1 with c2 and e2 with c1
                # Note: e1 and e2 might be part of the same 4-cycle, so this is tricky.
                # Simple heuristic: check each independently first.
                if check_color_change(u1, v1, c2, adj_sa, ec_sa, c1):
                    continue
                if check_color_change(u2, v2, c1, adj_sa, ec_sa, c2):
                    continue

                # If both pass, apply swap
                ec_sa[e1] = c2
                ec_sa[e2] = c1

            T *= cooling

        return best_adj, best_ec, best_edges

    # Multi-Start
    total_time = 140.0
    num_restarts = 6
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
