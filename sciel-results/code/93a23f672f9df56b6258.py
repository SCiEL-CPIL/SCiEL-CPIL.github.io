import time
import random
import itertools

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 145.0

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_key = lambda u, v: (min(u, v), max(u, v))
    adj = [[] for _ in range(n)]
    edge_colors = {}

    def get_color(u, v):
        k = edge_key(u, v)
        return edge_colors.get(k)

    def has_conflict_with_new_edge(u, v, c):
        """
        Check if adding edge (u,v) with color c creates any non-rainbow 4-cycle.
        For every pair (w, x) such that (w,x) is an edge, with w adjacent to v and x adjacent to u (w != u, x != v),
        the 4-cycle u-v-w-x-u has edges (u,v), (v,w), (w,x), (x,u).
        All these must exist (except (u,v) which we're adding) and their colors must be distinct.
        So for each such w, x:
          - (v,w) must exist
          - (w,x) must exist
          - (x,u) must exist
          - w != u, w != v, x != u, x != v, w != x
          - colors: c (for u,v), col(v,w), col(w,x), col(x,u) must all be distinct
        """
        for w in adj[v]:
            if w == u:
                continue
            col_vw = get_color(v, w)
            if col_vw is None:
                continue
            for x in adj[u]:
                if x == v or x == w:
                    continue
                col_wx = get_color(w, x)
                if col_wx is None:
                    continue
                col_xu = get_color(x, u)
                if col_xu is None:
                    continue
                if len({c, col_vw, col_wx, col_xu}) < 4:
                    return True
        return False

    def count_conflicts_for_edge_change(u, v, old_c, new_c):
        """
        Count how many 4-cycles would be broken if we change color of (u,v) from old_c to new_c.
        A 4-cycle u-v-w-x-u is affected.
        For each w in adj[v] (w != u), x in adj[u] (x != v, x != w):
          if (v,w), (w,x), (x,u) all exist:
            current colors: old_c, col_vw, col_wx, col_xu
            new colors: new_c, col_vw, col_wx, col_xu
            If currently distinct (must be, since valid) but new set has duplicate, it's a conflict.
        """
        conflicts = 0
        for w in adj[v]:
            if w == u:
                continue
            col_vw = get_color(v, w)
            if col_vw is None:
                continue
            for x in adj[u]:
                if x == v or x == w:
                    continue
                col_wx = get_color(w, x)
                if col_wx is None:
                    continue
                col_xu = get_color(x, u)
                if col_xu is None:
                    continue
                # Current set should be 4 distinct
                new_set = {new_c, col_vw, col_wx, col_xu}
                if len(new_set) < 4:
                    conflicts += 1
        return conflicts

    def greedy_initial(seed_order=None):
        nonlocal adj, edge_colors
        adj = [[] for _ in range(n)]
        edge_colors = {}
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if time.time() > deadline - 10:
                break
            found = False
            # Try colors in random order
            colors = list(range(num_colors))
            random.shuffle(colors)
            for c in colors:
                if not has_conflict_with_new_edge(u, v, c):
                    k = edge_key(u, v)
                    edge_colors[k] = c
                    adj[u].append(v)
                    adj[v].append(u)
                    found = True
                    break
        return len(edge_colors)

    def try_add_edge(u, v):
        """Try to add edge (u,v) with a valid color. Returns True if added."""
        k = edge_key(u, v)
        if k in edge_colors:
            return True
        colors = list(range(num_colors))
        random.shuffle(colors)
        for c in colors:
            if not has_conflict_with_new_edge(u, v, c):
                edge_colors[k] = c
                adj[u].append(v)
                adj[v].append(u)
                return True
        return False

    def try_remove_edge(u, v):
        k = edge_key(u, v)
        if k not in edge_colors:
            return False
        del edge_colors[k]
        adj[u].remove(v)
        adj[v].remove(u)
        return True

    def try_change_color(u, v, new_c):
        k = edge_key(u, v)
        if k not in edge_colors:
            return False
        old_c = edge_colors[k]
        if new_c == old_c:
            return True
        # Check if change creates conflict
        conflicts = count_conflicts_for_edge_change(u, v, old_c, new_c)
        if conflicts == 0:
            edge_colors[k] = new_c
            return True
        return False

    def run_sa(time_limit):
        nonlocal adj, edge_colors
        current_edges = len(edge_colors)
        best_edges = current_edges
        best_ec = dict(edge_colors)
        best_adj = [list(lst) for lst in adj]

        T = 2.0
        cooling = 0.9997
        min_T = 0.01

        while T > min_T and time.time() < time_limit:
            move_type = random.random()

            if move_type < 0.45:
                # Add edge
                missing = [e for e in all_edges if e not in edge_colors]
                if not missing:
                    continue
                u, v = random.choice(missing)
                c = random.randint(0, num_colors - 1)
                if not has_conflict_with_new_edge(u, v, c):
                    k = edge_key(u, v)
                    edge_colors[k] = c
                    adj[u].append(v)
                    adj[v].append(u)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_ec = dict(edge_colors)
                        best_adj = [list(lst) for lst in adj]
            elif move_type < 0.65:
                # Remove edge
                if not edge_colors:
                    continue
                e = random.choice(list(edge_colors.keys()))
                u, v = e
                try_remove_edge(u, v)
                current_edges -= 1
                # Probabilistically accept worsening (remove) with small probability based on T
                # Actually, in SA, we consider the move. Let's implement it properly:
                # The move is: remove edge. Delta = -1.
                # Accept with prob exp(-(-1)/T) = exp(1/T) which is >1, so always accept? 
                # Wait, standard SA: we minimize cost. Here we maximize edges.
                # Let's stick to the previous simple heuristic or use a penalty.
                # The previous code had a fixed 0.05 accept rate for removals. Let's keep it simpler.
                if random.random() > 0.05:
                    # Undo removal
                    try_add_edge(u, v)
                    current_edges += 1
            elif move_type < 0.9:
                # Change color
                if not edge_colors:
                    continue
                e = random.choice(list(edge_colors.keys()))
                u, v = e
                old_c = edge_colors[e]
                new_c = random.randint(0, num_colors - 1)
                if new_c != old_c:
                    conflicts = count_conflicts_for_edge_change(u, v, old_c, new_c)
                    if conflicts == 0:
                        edge_colors[e] = new_c
            else:
                # Swap colors
                if len(edge_colors) < 2:
                    continue
                e1, e2 = random.sample(list(edge_colors.keys()), 2)
                c1 = edge_colors[e1]
                c2 = edge_colors[e2]
                if c1 != c2:
                    u1, v1 = e1
                    u2, v2 = e2
                    # This is complex to check perfectly, so just try and revert if conflict increases
                    # For now, skip complex swap to avoid bugs, or do a simple validity check
                    # Let's just use color change for now for safety.
                    pass

            T *= cooling

        # Reconstruct best if needed? No, we track best_ec and best_adj.
        # But the state (adj, edge_colors) is the current one, not best.
        # We should restore best at the end of a run.
        adj[:] = [list(lst) for lst in best_adj]
        edge_colors.clear()
        edge_colors.update(best_ec)
        return best_edges

    # Multi-Start
    total_time = 140.0
    num_restarts = 8
    time_per_restart = total_time / num_restarts

    global_best_edges = 0
    global_best_ec = {}
    global_best_adj = []

    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break

        greedy_initial()
        best_edges_r = run_sa(time.time() + time_per_restart - 1)

        if best_edges_r > global_best_edges:
            global_best_edges = best_edges_r
            global_best_ec = dict(edge_colors)
            global_best_adj = [list(lst) for lst in adj]

    result = []
    for (u, v), c in global_best_ec.items():
        result.append([u, v, c])

    return result
