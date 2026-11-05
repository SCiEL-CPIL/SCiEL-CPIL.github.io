"""Improved program: Branch-and-bound with LP relaxation for minimum clique transversal."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges

    def bron_kerbosch(adj, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            if P:
                u = max(P, key=lambda v: len(adj[v] & P))
            elif X:
                u = max(X, key=lambda v: len(adj[v] & X))
            else:
                return
            
            # Branch on vertices in P \\ N(u)
            for v in list(P - adj[u]):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bk(frozenset(), set(range(n)), set())
        return cliques

    def greedy_set_cover(cliques, n):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        if not cliques:
            return set()
        
        uncovered = set(range(len(cliques)))
        vertex_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)
        
        selected = set()
        while uncovered:
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = len(vertex_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
            
            selected.add(best_v)
            uncovered -= vertex_cliques[best_v]
        
        return selected

    def branch_and_bound(cliques, n, initial_solution, time_limit):
        """
        Branch-and-bound with LP relaxation for minimum clique transversal.
        
        Variables: x[v] in {0,1} for each vertex v
        Minimize: sum(x[v])
        Subject to: for each clique C, sum(x[v] for v in C) >= 1
                    x[v] >= 0
        """
        m = len(cliques)
        if m == 0:
            return set()
        
        # Build constraint matrix: A[i][j] = 1 if vertex j is in clique i
        A = np.zeros((m, n))
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i][v] = 1.0
        
        b = np.ones(m)
        c = np.ones(n)
        
        # LP relaxation bounds
        def solve_lp(fixed_vars=None):
            """Solve LP relaxation with some variables fixed."""
            if fixed_vars is None:
                fixed_vars = {}
            
            c_vec = c.copy()
            A_mat = A.copy()
            b_vec = b.copy()
            
            # Remove fixed variables and adjust constraints
            free_vars = [v for v in range(n) if v not in fixed_vars]
            fixed_sum = np.zeros(m)
            for v, val in fixed_vars.items():
                if val == 1:
                    fixed_sum += A[:, v]
            
            # Constraints become: sum_{free v in C} x[v] >= 1 - sum_{fixed v in C} x[v]
            # If 1 - fixed_sum[i] <= 0, constraint is satisfied
            # If 1 - fixed_sum[i] > 1, infeasible
            b_new = np.maximum(1 - fixed_sum, 0)
            
            # Remove satisfied constraints
            active = b_new > 0
            if not np.any(active):
                return 0.0, {}  # All constraints satisfied by fixed vars
            
            A_active = A_mat[active][:, free_vars]
            b_active = b_new[active]
            c_active = c_vec[free_vars]
            
            if len(free_vars) == 0:
                return 0.0, {}
            
            # Solve: min c^T x s.t. A_active x >= b_active, 0 <= x <= 1
            # Convert to standard form: min c^T x s.t. -A_active x <= -b_active, 0 <= x <= 1
            res = linprog(
                c_active,
                A_ub=-A_active,
                b_ub=-b_active,
                bounds=[(0, 1)] * len(free_vars),
                method='highs'
            )
            
            if res.status != 0:
                return float('inf'), None
            
            obj = float(res.fun)
            solution = {free_vars[i]: float(res.x[i]) for i in range(len(free_vars))}
            return obj, solution
        
        # Initial upper bound from greedy
        best_obj = len(initial_solution)
        best_solution = set(initial_solution)
        
        # Branch-and-bound
        nodes_explored = 0
        start_bb_time = time.time()
        
        # State: (lower_bound, fixed_vars, free_vars)
        # Use a stack for DFS
        stack = [(0.0, {}, set(range(n)))]
        
        # Sort vertices by "importance" for branching (vertices in more cliques first)
        vertex_clique_count = [0] * n
        for clique in cliques:
            for v in clique:
                vertex_clique_count[v] += 1
        
        # Branch on the most constrained variable
        def get_branch_var(fixed_vars):
            """Choose next variable to branch on."""
            # Try to find a fractional variable in the LP solution
            # For simplicity, pick the unfixed variable with highest clique count
            candidates = [v for v in range(n) if v not in fixed_vars]
            if not candidates:
                return None
            # Pick the one in most cliques
            return max(candidates, key=lambda v: vertex_clique_count[v])
        
        while stack:
            # Time check
            if time.time() > start_bb_time + time_limit:
                break
            
            lower_bound, fixed_vars, free_vars = stack.pop()
            
            if lower_bound >= best_obj:
                continue
            
            # Solve LP relaxation
            lp_obj, lp_sol = solve_lp(fixed_vars)
            if lp_sol is None:
                continue  # Infeasible
            if lp_obj >= best_obj:
                continue
            
            # Check if LP solution is integral
            if lp_sol is not None:
                is_integral = True
                for v, val in lp_sol.items():
                    if abs(val - round(val)) > 1e-6:
                        is_integral = False
                        break
                
                if is_integral:
                    # Found an integer solution
                    sol_set = set(fixed_vars.keys())
                    for v, val in lp_sol.items():
                        if val > 0.5:
                            sol_set.add(v)
                    # Verify it's valid
                    if all(any(v in sol_set for v in clique) for clique in cliques):
                        if len(sol_set) < best_obj:
                            best_obj = len(sol_set)
                            best_solution = sol_set
                    continue
            
            # Branch: pick a variable to branch on
            if not free_vars:
                continue
            
            # Get LP solution for branching
            if lp_sol is None:
                continue
            
            # Find a fractional variable
            frac_var = None
            for v, val in lp_sol.items():
                if 0.1 < val < 0.9:
                    frac_var = v
                    break
            
            if frac_var is None:
                # No fractional vars, use heuristic branching
                frac_var = get_branch_var(fixed_vars)
            
            if frac_var is None or frac_var not in free_vars:
                continue
            
            # Branch: x[frac_var] = 0 and x[frac_var] = 1
            # Push both children onto stack (explore x=1 first for tighter bounds)
            
            # Child 1: x[frac_var] = 1
            fixed1 = dict(fixed_vars)
            fixed1[frac_var] = 1
            free1 = free_vars - {frac_var}
            
            # Child 2: x[frac_var] = 0
            fixed2 = dict(fixed_vars)
            fixed2[frac_var] = 0
            free2 = free_vars - {frac_var}
            
            # Push child 2 first so child 1 is explored first (LIFO)
            if free2:
                stack.append((lp_obj, fixed2, free2))
            if free1:
                stack.append((lp_obj, fixed1, free1))
            
            nodes_explored += 1
        
        return best_solution

    def local_search(selected, cliques, n, time_limit=3.0):
        """Try to improve the solution by removing redundant vertices."""
        if not cliques:
            return selected
        
        selected = set(selected)
        
        vertex_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)
        
        all_cliques = set(range(len(cliques)))
        
        def is_valid(sel):
            cov = set()
            for v in sel:
                cov |= vertex_cliques[v]
            return all_cliques <= cov
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for v in list(selected):
                if len(selected) <= 1:
                    break
                new_sel = selected - {v}
                if is_valid(new_sel):
                    selected = new_sel
                    improved = True
                    break
        
        return selected

    def solve_instance(n, p, seed, time_budget):
        adj, edges = instance(n, p, seed)
        
        # Find all maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        # Greedy set cover for initial solution
        selected = greedy_set_cover(cliques, n)
        
        # Branch-and-bound to improve
        remaining = time_budget - (time.time() - start_time)
        if remaining > 1.0:
            bb_time = min(remaining - 0.5, time_budget * 0.8)
            selected = branch_and_bound(cliques, n, selected, bb_time)
        
        # Final local search polish
        remaining = time_budget - (time.time() - start_time)
        if remaining > 0.5:
            selected = local_search(selected, cliques, n, min(remaining - 0.2, 2.0))
        
        return sorted(selected)

    # Allocate time to each instance
    time_per_instance = 45.0  # 3 instances * 45s = 135s, leaving 15s margin
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        remaining_total = deadline - time.time()
        if remaining_total < 2.0:
            results.append([])
            continue
        # Use remaining time for this instance, capped at time_per_instance
        budget = min(time_per_instance, remaining_total - 1.0)
        result = solve_instance(n, p, seed, budget)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
