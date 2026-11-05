# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return

            if not P:
                return

            # pivot: choose u in P ∪ X maximizing |P ∩ N(u)|
            pivot_candidates = P | X
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            if best_pivot is not None:
                candidates = [v for v in P if v not in adj[best_pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_milp_hitting_set(n, cliques, time_limit):
        """
        Formulates the Minimum Hitting Set problem as a MILP.
        Variables: x_v = 1 if vertex v is selected, 0 otherwise.
        Objective: Minimize sum(x_v)
        Constraints: For each clique C, sum(x_v for v in C) >= 1
        """
        if not cliques:
            return set()
        
        m = len(cliques)
        
        # Build incidence matrix A (m x n)
        # A[i, j] = 1 if vertex j is in clique i
        # Using sparse matrix representation if scipy.sparse is available, 
        # but milp accepts dense or sparse. n is small (<40), m is small.
        # Let's use a dense numpy array for simplicity given small sizes.
        
        A = np.zeros((m, n), dtype=np.float64)
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1.0
        
        # Objective: minimize sum(x)
        c = np.ones(n, dtype=np.float64)
        
        # Constraints: A @ x >= 1 for all rows (cliques)
        # milp expects constraints in form: lb <= A_ub @ x <= ub
        # So we set lb = 1, ub = inf
        # Note: scipy.optimize.milp takes constraints as LinearConstraint(A, lb, ub)
        
        lb = np.ones(m, dtype=np.float64)
        ub = np.full(m, np.inf, dtype=np.float64)
        
        # Variable bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(n, dtype=np.float64), ub=np.ones(n, dtype=np.float64))
        
        # Integrality: all variables are integer (binary)
        integrality = np.ones(n, dtype=np.int8)
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=bounds,
                options={
                    'time_limit': time_limit,
                    'mip_gap': 0.0, # Try for optimal, but time limit stops it
                    # 'verb': 0 # Silent
                }
            )
            
            if res.status == 0: # Optimal found
                # res.x is the solution vector
                sol_set = set(np.where(res.x > 0.5)[0])
                return sol_set
            elif res.status == 1: # Locally optimal (or feasible solution found within time)
                # scipy status 1: "The problem is feasible and a solution was found."
                # Actually, status 0 is Optimal, 1 is Locally Optimal, 2 is Infeasible, etc.
                # For MIP, if time limit hits, it might return status 0 (if optimal proof found) 
                # or a feasible solution. 
                # According to docs: 
                # 0: Optimal solution found.
                # 1: Locally optimal solution found. (For MIP, this usually means a feasible solution was found but optimality not proven, OR it's a specific HiGHS status).
                # Let's check if res.x exists.
                if hasattr(res, 'x') and res.x is not None:
                    sol_set = set(np.where(res.x > 0.5)[0])
                    # Verify validity just in case
                    if is_valid_sol(sol_set, cliques):
                        return sol_set
                return set()
            else:
                # Failed
                return set()
                
        except Exception as e:
            return set()

    def is_valid_sol(sol_set, cliques):
        for c in cliques:
            if not any(v in sol_set for v in c):
                return False
        return True

    def greedy_fallback(n, cliques):
        """
        Fast greedy fallback if MILP fails or takes too long.
        """
        if not cliques:
            return set()
        
        m = len(cliques)
        vertex_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(i)

        uncovered = set(range(m))
        selected = set()
        
        # Precompute counts for speed? n is small.
        while uncovered:
            best_v = -1
            best_gain = 0
            for v in range(n):
                gain = 0
                for i in vertex_cliques[v]:
                    if i in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v == -1 or best_gain == 0:
                break
                
            selected.add(best_v)
            for i in vertex_cliques[best_v]:
                uncovered.discard(i)
                
        # Local search removal
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                test = set(selected)
                test.discard(v)
                valid = True
                for c in cliques:
                    if not any(u in test for u in c):
                        valid = False
                        break
                if valid:
                    selected = test
                    changed = True
                    break
        return selected

    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        current_time = time.time()
        if current_time > deadline - 10:
            # Not enough time, return a valid but potentially suboptimal solution
            # We need to generate the graph and cliques first to get a valid set
            # But if we are this late, we should just return a trivial valid set: all vertices
            # Wait, we must return a list for THIS instance.
            # If we haven't processed previous ones, they are already in results.
            # For this one, if we can't solve, return all vertices? 
            # No, we need to construct the graph to know n. We have n.
            # But we don't know the edges/cliques without generating them.
            # Generating graph/cliques is fast. Solving is slow.
            # If we are out of time, we assume we can at least generate the structure.
            pass

        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            results.append([])
            continue

        elapsed = time.time() - start_time
        remaining_instances = 3 - len(results)
        
        # Allocate time for MILP
        # We want to leave 5s for safety and overhead
        available_time = deadline - 5 - elapsed
        if available_time < 1.0:
             # Emergency: use greedy
             sol = greedy_fallback(n, cliques)
             results.append(sorted(sol))
             continue

        # Give MILP a chunk of the remaining time
        # If it's the last instance, give it almost all remaining time.
        # If it's the first, give it 1/3 of remaining? 
        # MILP is exact, so more time is better. 
        # Let's give it 80% of the available time for this instance, 
        # saving 20% for potential greedy fallback or next instance setup?
        # Actually, if MILP succeeds, it's done. If it fails, we use greedy.
        # Greedy is very fast (<0.1s).
        # So we can allocate most of the time to MILP.
        
        milp_time = available_time * 0.9
        
        sol = solve_milp_hitting_set(n, cliques, milp_time)
        
        if not sol:
            # Fallback to greedy
            sol = greedy_fallback(n, cliques)
            
        # Final verification and repair if needed
        if not is_valid_sol(sol, cliques):
            # If MILP returned an invalid solution (shouldn't happen if status 0/1)
            # or empty set when cliques exist
            sol = greedy_fallback(n, cliques)
            
        results.append(sorted(sol))

    return results
# EVOLVE-BLOCK-END
