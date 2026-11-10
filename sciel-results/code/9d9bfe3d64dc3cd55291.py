import time
import math
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 256
    max_sum = 2 * N

    # 1. Precompute squares up to 2*N
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)

    # 2. Filter vertices: 2*v must not be a square (diagonal constraint)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)

    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]

    # Map vertex to index
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}

    # 3. Build adjacency list for conflicts
    adj = [set() for _ in range(n_vars)]
    conflict_list = []
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj[i].add(j)
                adj[j].add(i)
                conflict_list.append((i, j))

    m_conflicts = len(conflict_list)

    # Helper: check if adding idx to a set causes conflict
    def can_add(idx, sol_set):
        for neighbor in adj[idx]:
            if neighbor in sol_set:
                return False
        return True

    # Try to add all possible vertices
    def try_add_all(sol_set):
        changed = True
        while changed:
            changed = False
            for idx in range(n_vars):
                if idx not in sol_set and can_add(idx, sol_set):
                    sol_set.add(idx)
                    changed = True
        return sol_set

    # Weighted greedy: prefer vertices with fewer conflicts
    def greedy_weighted():
        S = set()
        remaining = set(range(n_vars))
        while remaining:
            # Score: prefer low degree, then low value
            best = min(remaining, key=lambda i: (len(adj[i] & remaining), i))
            if can_add(best, S):
                S.add(best)
                remaining.discard(best)
            else:
                remaining.discard(best)
        return S

    # Greedy by degree (prefer low-degree vertices first)
    def greedy_by_degree():
        degrees = [len(adj[i]) for i in range(n_vars)]
        order = sorted(range(n_vars), key=lambda i: degrees[i])
        S = set()
        for idx in order:
            if can_add(idx, S):
                S.add(idx)
        return S

    # Randomized greedy
    def greedy_randomized(seed=None):
        rng = random.Random(seed)
        S = set()
        order = list(range(n_vars))
        rng.shuffle(order)
        for idx in order:
            if can_add(idx, S):
                S.add(idx)
        return S

    # 1-swap search
    def one_swap_search(sol_set, max_rounds=5):
        for _ in range(max_rounds):
            improved = False
            for idx_to_remove in list(sol_set):
                new_set = sol_set - {idx_to_remove}
                new_set = try_add_all(new_set)
                if len(new_set) > len(sol_set):
                    sol_set = new_set
                    improved = True
                    break
            if not improved:
                break
        return sol_set

    # 2-swap search
    def two_swap_search(sol_set, max_rounds=3):
        for _ in range(max_rounds):
            improved = False
            sol_list = list(sol_set)
            for i in range(len(sol_list)):
                for j in range(i + 1, len(sol_list)):
                    new_set = sol_set - {sol_list[i], sol_list[j]}
                    new_set = try_add_all(new_set)
                    if len(new_set) > len(sol_set):
                        sol_set = new_set
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return sol_set

    # 3-swap search (limited)
    def three_swap_search(sol_set, max_rounds=1):
        for _ in range(max_rounds):
            improved = False
            sol_list = list(sol_set)
            n = len(sol_list)
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        new_set = sol_set - {sol_list[i], sol_list[j], sol_list[k]}
                        new_set = try_add_all(new_set)
                        if len(new_set) > len(sol_set):
                            sol_set = new_set
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return sol_set

    # Branch and bound for maximum independent set
    def branch_and_bound(candidates, current_set, upper_bound, best_solution, best_size):
        if time.time() > deadline - 2:
            return best_solution, best_size
        
        if not candidates:
            if len(current_set) > best_size:
                best_solution = current_set.copy()
                best_size = len(current_set)
            return best_solution, best_size
        
        # Upper bound: current + all candidates
        if len(current_set) + len(candidates) <= best_size:
            return best_solution, best_size
        
        # Pick a vertex to branch on
        v = min(candidates, key=lambda x: len(adj[x] & candidates))
        
        # Branch 1: include v
        new_candidates = candidates - {v}
        new_candidates = {c for c in new_candidates if c not in adj[v]}
        best_solution, best_size = branch_and_bound(
            new_candidates, current_set | {v}, upper_bound, best_solution, best_size
        )
        
        # Branch 2: exclude v
        new_candidates = candidates - {v}
        best_solution, best_size = branch_and_bound(
            new_candidates, current_set, upper_bound, best_solution, best_size
        )
        
        return best_solution, best_size

    # Best solution found so far
    best_solution = set()
    best_size = 0

    # Phase 1: Try several greedy initializations
    initializations = []

    # Weighted greedy
    s = greedy_weighted()
    s = try_add_all(s)
    initializations.append(s)

    # Degree-based greedy
    s = greedy_by_degree()
    s = try_add_all(s)
    initializations.append(s)

    # Randomized greedy (multiple restarts)
    for seed in range(20):
        s = greedy_randomized(seed=seed)
        s = try_add_all(s)
        initializations.append(s)

    for s in initializations:
        if len(s) > best_size:
            best_solution = s.copy()
            best_size = len(s)

    # Phase 2: Local search refinement
    if time.time() < deadline - 10:
        best_solution = one_swap_search(best_solution, max_rounds=5)
        best_size = len(best_solution)

    if time.time() < deadline - 8:
        best_solution = two_swap_search(best_solution, max_rounds=3)
        best_size = len(best_solution)

    if time.time() < deadline - 5:
        best_solution = one_swap_search(best_solution, max_rounds=3)
        best_size = len(best_solution)

    # Phase 3: Try MILP
    if time.time() < deadline - 20:
        try:
            if m_conflicts > 0:
                A_ub = np.zeros((m_conflicts, n_vars), dtype=np.float64)
                b_ub = np.ones(m_conflicts, dtype=np.float64)

                for k, (i, j) in enumerate(conflict_list):
                    A_ub[k, i] = 1.0
                    A_ub[k, j] = 1.0

                c = -np.ones(n_vars, dtype=np.float64)
                bounds = Bounds(lb=0, ub=1)
                integrality = np.ones(n_vars, dtype=np.int8)

                time_left = deadline - time.time() - 15
                if time_left > 10:
                    res = milp(
                        c=c,
                        constraints=[LinearConstraint(A_ub, ub=b_ub)],
                        integrality=integrality,
                        bounds=bounds,
                        options={"time_limit": min(time_left, 30)}
                    )

                    if res.status in (0, 1):
                        milp_set = set()
                        for i in range(n_vars):
                            if res.x[i] > 0.5:
                                milp_set.add(i)
                        milp_set = try_add_all(milp_set)
                        if len(milp_set) > best_size:
                            best_solution = milp_set.copy()
                            best_size = len(milp_set)

        except Exception:
            pass

    # Phase 4: More local search if time permits
    if time.time() < deadline - 5:
        best_solution = one_swap_search(best_solution, max_rounds=5)
        best_size = len(best_solution)

    if time.time() < deadline - 3:
        best_solution = two_swap_search(best_solution, max_rounds=2)
        best_size = len(best_solution)

    # Phase 5: Try branch and bound on remaining candidates
    if time.time() < deadline - 3:
        candidates = set(range(n_vars)) - best_solution
        # Limit B&B to smaller instances
        if len(candidates) < 50:
            best_solution, best_size = branch_and_bound(
                candidates, best_solution.copy(), len(best_solution) + len(candidates),
                best_solution, best_size
            )

    # Phase 6: Final polish
    if time.time() < deadline - 1:
        for idx in range(n_vars):
            if idx not in best_solution and can_add(idx, best_solution):
                best_solution.add(idx)
                best_size = len(best_solution)

    # Convert back to list of values
    result = [valid_vertices[i] for i in sorted(best_solution)]
    if not result:
        result = [valid_vertices[0]]

    return result
