"""Strong edge colourings via MILP with time budget and greedy fallback."""
import time
import sys

def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def build_instance(n, d, seed):
        from random import Random
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def compute_conflicts(edges, adj):
        """For each edge i, find all edges j that conflict (share vertex or connected by edge)."""
        m = len(edges)
        conflicts = [set() for _ in range(m)]
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        for i, (u, v) in enumerate(edges):
            # Edges sharing vertex u or v
            for w in adj.get(u, set()):
                e2 = (min(u, w), max(u, w))
                j = edge_to_idx.get(e2)
                if j is not None and j != i:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
            for w in adj.get(v, set()):
                e2 = (min(v, w), max(v, w))
                j = edge_to_idx.get(e2)
                if j is not None and j != i:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        return conflicts

    def greedy_color(edges, adj):
        """Greedy first-fit coloring with ordering by degree."""
        m = len(edges)
        # Order edges by sum of endpoint degrees (descending) for better packing
        edge_degree = [(len(adj.get(u, set())) + len(adj.get(v, set())), i) for i, (u, v) in enumerate(edges)]
        edge_degree.sort(reverse=True)
        order = [i for _, i in edge_degree]

        # Precompute conflict sets
        conflicts = compute_conflicts(edges, adj)
        color = [None] * m
        for i in order:
            taken = set()
            for j in conflicts[i]:
                if color[j] is not None:
                    taken.add(color[j])
            c = 0
            while c in taken:
                c += 1
            color[i] = c
        return color

    def milp_color(edges, adj, max_colors, time_limit=25.0):
        """Use scipy MILP to find minimum strong edge coloring."""
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        m = len(edges)
        conflicts = compute_conflicts(edges, adj)

        # Variables: x[i][c] = 1 if edge i gets color c
        # Number of variables: m * max_colors
        n_vars = m * max_colors

        # Objective: minimize sum of colors used = sum_c (1 if any edge uses color c)
        # We use auxiliary variables y[c] >= x[i][c] for all i
        # y[c] >= 0, y[c] <= 1
        # Minimize sum_c y[c]
        n_y = max_colors
        total_vars = n_vars + n_y

        # Objective coefficients
        c_obj = np.zeros(total_vars)
        for c in range(max_colors):
            c_obj[n_vars + c] = 1.0  # minimize y[c]

        # Constraints:
        # 1. Each edge gets exactly one color: sum_c x[i][c] = 1 for all i
        # 2. No two conflicting edges share a color: x[i][c] + x[j][c] <= 1 for all (i,j) in conflicts, all c
        # 3. y[c] >= x[i][c] for all i, c (i.e., x[i][c] - y[c] <= 0)

        A_rows = []
        b_lower = []
        b_upper = []

        # Constraint 1: sum_c x[i][c] = 1
        for i in range(m):
            row = np.zeros(total_vars)
            for c in range(max_colors):
                row[i * max_colors + c] = 1.0
            A_rows.append(row)
            b_lower.append(1.0)
            b_upper.append(1.0)

        # Constraint 2: conflict constraints
        for i in range(m):
            for j in conflicts[i]:
                if i < j:  # avoid duplicates
                    for c in range(max_colors):
                        row = np.zeros(total_vars)
                        row[i * max_colors + c] = 1.0
                        row[j * max_colors + c] = 1.0
                        A_rows.append(row)
                        b_lower.append(-np.inf)
                        b_upper.append(1.0)

        # Constraint 3: y[c] >= x[i][c], i.e., x[i][c] - y[c] <= 0
        for i in range(m):
            for c in range(max_colors):
                row = np.zeros(total_vars)
                row[i * max_colors + c] = 1.0
                row[n_vars + c] = -1.0
                A_rows.append(row)
                b_lower.append(-np.inf)
                b_upper.append(0.0)

        if not A_rows:
            # No edges
            return [0] * m

        A = np.array(A_rows)
        b_low = np.array(b_lower)
        b_high = np.array(b_upper)

        # Bounds: all variables in [0, 1], integer
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        integrality = np.ones(total_vars)  # all binary

        cons = LinearConstraint(A, b_low, b_high)
        bounds = Bounds(lb, ub)

        try:
            res = milp(c_obj, constraints=cons, integrality=integrality, bounds=bounds,
                       options={"time_limit": time_limit, "presolve": True})
            if res.status == 0 and res.x is not None:
                color = [0] * m
                for i in range(m):
                    for c in range(max_colors):
                        if res.x[i * max_colors + c] > 0.5:
                            color[i] = c
                            break
                    else:
                        # Fallback: assign 0
                        color[i] = 0
                return color
        except Exception:
            pass
        return None

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        if time.time() > DEADLINE - 5:
            # Not enough time, use greedy for remaining
            edges = build_instance(n, d, seed)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            color = greedy_color(edges, adj)
            out.append(color)
            continue

        edges = build_instance(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        m = len(edges)
        # Upper bound: greedy gives us a starting point
        greedy_sol = greedy_color(edges, adj)
        greedy_colors = max(greedy_sol) + 1 if m > 0 else 0

        # Try MILP with a few color budgets, starting from greedy and going down
        best_sol = greedy_sol
        best_num_colors = greedy_colors

        # Try to reduce colors using MILP
        # Start from greedy_colors - 1 and go down
        time_per_attempt = max(3.0, (DEADLINE - time.time() - 5.0) / 3.0)
        for nc in range(greedy_colors - 1, 0, -1):
            if time.time() > DEADLINE - 5:
                break
            sol = milp_color(edges, adj, nc, time_limit=time_per_attempt)
            if sol is not None:
                # Verify the solution
                conflicts = compute_conflicts(edges, adj)
                valid = True
                for i in range(m):
                    for j in conflicts[i]:
                        if i < j and sol[i] == sol[j]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_sol = sol
                    best_num_colors = nc
                    break  # Found a valid solution with fewer colors

        out.append(best_sol)

    return out
