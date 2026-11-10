import time
import math
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

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
    adj = [[] for _ in range(n_vars)]
    conflict_set = set()  # frozenset of (i,j) pairs
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj[i].append(j)
                adj[j].append(i)
                conflict_set.add(frozenset((i, j)))

    # Helper: check if adding idx to a set causes conflict
    def can_add(idx, sol_set):
        for neighbor in adj[idx]:
            if neighbor in sol_set:
                return False
        return True

    # Helper: get size of solution
    def sol_size(sol_set):
        return len(sol_set)

    # Greedy with randomization
    def greedy_randomized():
        S = set()
        order = list(range(n_vars))
        random.shuffle(order)
        for idx in order:
            if can_add(idx, S):
                S.add(idx)
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

    # Greedy by degree (prefer high-degree vertices first)
    def greedy_by_degree_desc():
        degrees = [len(adj[i]) for i in range(n_vars)]
        order = sorted(range(n_vars), key=lambda i: -degrees[i])
        S = set()
        for idx in order:
            if can_add(idx, S):
                S.add(idx)
        return S

    # Local search: try to add any vertex that doesn't conflict
    def try_add_all(sol_set):
        changed = True
        while changed:
            changed = False
            for idx in range(n_vars):
                if idx not in sol_set and can_add(idx, sol_set):
                    sol_set.add(idx)
                    changed = True
        return sol_set

    # Local search: 1-swap (remove one, add many)
    def one_swap_search(sol_set, max_rounds=3):
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

    # Local search: 2-swap (remove two, add many)
    def two_swap_search(sol_set, max_rounds=2):
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

    # Best solution found so far
    best_solution = set()
    best_size = 0

    # Phase 1: Try several greedy initializations
    initializations = []

    # Degree-based greedy
    s = greedy_by_degree()
    s = try_add_all(s)
    initializations.append(s)

    s = greedy_by_degree_desc()
    s = try_add_all(s)
    initializations.append(s)

    # Randomized greedy (multiple restarts)
    for _ in range(5):
        s = greedy_randomized()
        s = try_add_all(s)
        initializations.append(s)

    for s in initializations:
        if len(s) > best_size:
            best_solution = s.copy()
            best_size = len(s)

    # Phase 2: Try MILP
    if time.time() < deadline - 20:
        try:
            # Build constraint matrix
            constraint_pairs = [tuple(p) for p in conflict_set]
            m_constraints = len(constraint_pairs)

            if m_constraints > 0:
                A_ub = np.zeros((m_constraints, n_vars), dtype=np.float64)
                b_ub = np.ones(m_constraints, dtype=np.float64)

                for k, (i, j) in enumerate(constraint_pairs):
                    A_ub[k, i] = 1.0
                    A_ub[k, j] = 1.0

                c = -np.ones(n_vars, dtype=np.float64)
                bounds = Bounds(lb=0, ub=1)
                integrality = np.ones(n_vars, dtype=np.int8)

                time_left = deadline - time.time() - 10
                if time_left > 5:
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

    # Phase 3: Local search refinement on best solution
    if time.time() < deadline - 5:
        # 1-swap search
        best_solution = one_swap_search(best_solution, max_rounds=5)
        best_size = len(best_solution)

        # 2-swap search
        if time.time() < deadline - 5:
            best_solution = two_swap_search(best_solution, max_rounds=3)
            best_size = len(best_solution)

        # More 1-swap rounds
        if time.time() < deadline - 3:
            best_solution = one_swap_search(best_solution, max_rounds=5)
            best_size = len(best_solution)

    # Phase 4: Final polish - try all single additions
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
