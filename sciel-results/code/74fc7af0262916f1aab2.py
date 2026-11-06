import time
import random
from collections import defaultdict
import scipy.optimize
import scipy.sparse
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        conflict = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                if x in (u, v) or y in (u, v):
                    conflict[i].add(j)
                    conflict[j].add(i)
                elif (u in adj[x] or u in adj[y] or v in adj[x] or v in adj[y]):
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    def solve_milp(edges, conflict, k_max, time_limit):
        m = len(edges)
        if m == 0:
            return [], 0
        
        # Variables: x[e, c] for e in 0..m-1, c in 0..k_max-1
        #             z[c] for c in 0..k_max-1
        # Total vars = m * k_max + k_max
        
        n_vars_x = m * k_max
        n_vars_z = k_max
        total_vars = n_vars_x + n_vars_z
        
        # Cost vector
        c = np.zeros(total_vars)
        c[n_vars_x:] = 1.0  # Minimize sum of z_c
        
        # Integrality
        integrality = np.ones(total_vars)
        
        # Bounds
        bounds = [(0, 1)] * total_vars
        
        # Constraints
        # 1. Sum_c x[e, c] = 1 for all e
        # 2. x[e, c] + x[e', c] <= 1 for all conflicts (e, e')
        # 3. x[e, c] <= z[c] for all e, c
        
        rows = []
        cols = []
        vals = []
        lb = []
        ub = []
        
        # Constraint 1: Sum x[e,c] = 1
        for e in range(m):
            row_idx = len(rows)
            for c in range(k_max):
                var_idx = e * k_max + c
                rows.append(row_idx)
                cols.append(var_idx)
                vals.append(1.0)
            lb.append(1.0)
            ub.append(1.0)
            
        # Constraint 2: Conflicts
        # For each conflict pair (e, e') and each colour c:
        # x[e, c] + x[e', c] <= 1
        for e in range(m):
            for e2 in conflict[e]:
                if e < e2: # Avoid duplicates
                    for c in range(k_max):
                        row_idx = len(rows)
                        var1 = e * k_max + c
                        var2 = e2 * k_max + c
                        rows.append(row_idx)
                        cols.append(var1)
                        vals.append(1.0)
                        
                        rows.append(row_idx)
                        cols.append(var2)
                        vals.append(1.0)
                        
                        lb.append(-np.inf)
                        ub.append(1.0)
                        
        # Constraint 3: x[e, c] <= z[c]  =>  x[e, c] - z[c] <= 0
        for e in range(m):
            for c in range(k_max):
                row_idx = len(rows)
                var_x = e * k_max + c
                var_z = n_vars_x + c
                rows.append(row_idx)
                cols.append(var_x)
                vals.append(1.0)
                
                rows.append(row_idx)
                cols.append(var_z)
                vals.append(-1.0)
                
                lb.append(-np.inf)
                ub.append(0.0)
                
        if not rows:
            return [0]*m, 0

        A = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(lb), total_vars))
        
        res = scipy.optimize.milp(
            c=c,
            constraints=scipy.optimize.LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "mip_rel_gap": 0.0}
        )
        
        if res.status == 0: # Optimal
            x_flat = res.x[:n_vars_x]
            z_flat = res.x[n_vars_x:]
            
            colour = [0] * m
            max_c = 0
            for e in range(m):
                for c in range(k_max):
                    if x_flat[e * k_max + c] > 0.5:
                        colour[e] = c + 1
                        if c + 1 > max_c:
                            max_c = c + 1
                        break
            return colour, max_c
        elif res.status == 1: # Feasible (but not optimal)
            # Use the incumbent solution
            x_flat = res.x[:n_vars_x]
            colour = [0] * m
            max_c = 0
            for e in range(m):
                for c in range(k_max):
                    if x_flat[e * k_max + c] > 0.5:
                        colour[e] = c + 1
                        if c + 1 > max_c:
                            max_c = c + 1
                        break
            # Verify validity
            valid = True
            for e in range(m):
                if colour[e] == 0:
                    valid = False
                    break
            if valid:
                return colour, max_c
            else:
                return None, None
        else:
            return None, None

    def solve_instance(n, d, seed):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conflict = build_conflict(edges)
        
        # Estimate K_max
        # Use conjecture bound: 5*d^2/4
        # If d is the limit in gen_instance (max deg d-1), use d-1?
        # Prompt says max degree 4,5,6 for d=4,5,6.
        # Let's use the bound for d.
        k_bound = int(np.ceil(5 * d * d / 4.0)) + 2 # Safety margin
        
        # Try MILP
        time_per_instance = 30.0
        if time.time() > DEADLINE - 10:
            return greedy_fallback(edges, conflict)
            
        colour, max_c = solve_milp(edges, conflict, k_bound, time_per_instance)
        
        if colour is not None:
            return colour
        else:
            return greedy_fallback(edges, conflict)

    def greedy_fallback(edges, conflict):
        m = len(edges)
        if m == 0:
            return []
        
        # Simple greedy
        colour = [0] * m
        for i in range(m):
            forbidden = set()
            for nb in conflict[i]:
                if colour[nb] > 0:
                    forbidden.add(colour[nb])
            c = 1
            while c in forbidden:
                c += 1
            colour[i] = c
        return colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 5.0:
            out.append([])
            continue
        result = solve_instance(*spec)
        out.append(result)

    return out
