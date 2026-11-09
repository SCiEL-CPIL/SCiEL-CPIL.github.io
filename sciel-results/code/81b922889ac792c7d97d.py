import time
import random
import numpy as np
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build graph adjacency for conflict detection
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Conflict sets: two edges conflict iff they share a vertex OR are joined by an edge
        conflict = [set() for _ in range(m)]
        
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict[i].add(j)
                        conflict[j].add(i)

        # Helper to check if i and j conflict
        def is_conflict(i, j):
            return j in conflict[i]

        # 1. Greedy DSATUR to get an initial upper bound (UB)
        def greedy_dsatur():
            colors = [-1] * m
            uncolored = set(range(m))
            for _ in range(m):
                best_i, best_key = None, (-1, -1)
                for i in uncolored:
                    seen = set()
                    for j in conflict[i]:
                        if j in uncolored and colors[j] != -1:
                            seen.add(colors[j])
                    deg = sum(1 for j in conflict[i] if j in uncolored)
                    key = (len(seen), deg)
                    if key > best_key:
                        best_key = key
                        best_i = i
                i = best_i
                taken = set()
                for j in conflict[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        initial_colors = greedy_dsatur()
        ub = max(initial_colors) + 1 if initial_colors else 0
        
        # 2. ILP-based exact coloring using scipy.optimize.milp
        # Variables: x[i][c] = 1 if edge i gets color c
        # Objective: minimize sum of y[c] where y[c] = 1 if color c is used
        # Constraints:
        #   For each edge i: sum_c x[i][c] = 1
        #   For each conflicting pair (i,j) and each color c: x[i][c] + x[j][c] <= 1
        #   y[c] >= x[i][c] for all i, c (linking variables)
        
        # Number of colors to consider: up to ub
        K = ub
        
        # Decision variables:
        # x: m * K binary variables (flattened)
        # y: K binary variables
        # Total: m*K + K variables
        
        # Objective: minimize sum(y)
        c_obj = np.zeros(m * K + K)
        c_obj[m * K:] = 1.0  # minimize sum of y
        
        # Bounds: all variables binary [0, 1]
        integrality = np.ones(m * K + K)
        lb = np.zeros(m * K + K)
        ubound = np.ones(m * K + K)
        
        # Constraints:
        # 1. For each edge i: sum_c x[i][c] = 1  (m constraints)
        # 2. For each conflicting pair (i,j) and each color c: x[i][c] + x[j][c] <= 1
        # 3. y[c] >= x[i][c] for all i, c  =>  x[i][c] - y[c] <= 0
        
        # Build constraint matrix
        # We'll build it row by row
        
        A_rows = []
        b_lows = []
        b_ups = []
        
        # Constraint type 1: sum_c x[i][c] = 1
        for i in range(m):
            row = np.zeros(m * K + K)
            for c in range(K):
                row[i * K + c] = 1.0
            A_rows.append(row)
            b_lows.append(1.0)
            b_ups.append(1.0)
        
        # Constraint type 2: x[i][c] + x[j][c] <= 1 for conflicting pairs
        for i in range(m):
            for j in conflict[i]:
                if j > i:  # avoid duplicates
                    for c in range(K):
                        row = np.zeros(m * K + K)
                        row[i * K + c] = 1.0
                        row[j * K + c] = 1.0
                        A_rows.append(row)
                        b_lows.append(0.0)
                        b_ups.append(1.0)
        
        # Constraint type 3: x[i][c] - y[c] <= 0  =>  x[i][c] <= y[c]
        for i in range(m):
            for c in range(K):
                row = np.zeros(m * K + K)
                row[i * K + c] = 1.0
                row[m * K + c] = -1.0
                A_rows.append(row)
                b_lows.append(0.0)
                b_ups.append(0.0)
        
        # Stack into matrix
        A_mat = np.array(A_rows)
        b_low = np.array(b_lows)
        b_up = np.array(b_ups)
        
        # Check if we have time to run the ILP
        if time.time() > time_limit:
            return initial_colors
        
        try:
            constraints = LinearConstraint(A_mat, b_low, b_up)
            bounds = Bounds(lb, ubound)
            
            # Use HiGHS solver via scipy
            result = milp(
                c=c_obj,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": max(1.0, time_limit - (time.time() - start_time) - 2.0)}
            )
            
            if result.status == 0 and result.x is not None:
                x_sol = result.x[:m * K].reshape(m, K)
                colors = []
                for i in range(m):
                    c_assigned = int(np.argmax(x_sol[i]))
                    colors.append(c_assigned)
                
                # Verify validity
                ok = True
                for c in range(K):
                    members = [i for i in range(m) if colors[i] == c]
                    for a in range(len(members)):
                        for b in range(a + 1, len(members)):
                            if is_conflict(members[a], members[b]):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    # Re-index colors to be compact (0, 1, 2, ...)
                    color_map = {}
                    new_colors = []
                    next_c = 0
                    for c in colors:
                        if c not in color_map:
                            color_map[c] = next_c
                            next_c += 1
                        new_colors.append(color_map[c])
                    return new_colors
                else:
                    return initial_colors
            else:
                return initial_colors
        except Exception:
            return initial_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0.5:
            # Emergency greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            cols = {}
            for e in edges:
                near = set(e)
                for x in e:
                    near |= adj[x]
                taken = set()
                for f, c in cols.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                cols[e] = c
            results.append([cols[e] for e in edges])
            continue
        tl = max(1.0, remaining - 1.0)
        results.append(solve_instance(n, d, seed, tl))
    return results
