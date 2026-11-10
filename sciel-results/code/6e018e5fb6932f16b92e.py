import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    N = 256
    max_sum = 2 * N

    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)

    # 2. Filter vertices: 2*v must not be a square
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)

    if not valid_vertices:
        return [1]

    # 3. Build conflict graph
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    M = len(valid_vertices)

    # List of constraints (i, j) where i < j and valid_vertices[i] + valid_vertices[j] is square
    constraints = []
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            if vi + vj in squares:
                constraints.append((i, j))

    # 4. Solve using SciPy MILP
    milp_solution = None
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Objective: maximize sum(x)
        # milp minimizes, so we minimize -sum(x)
        c = -np.ones(M)

        if len(constraints) == 0:
            # No conflicts, take all
            return list(valid_vertices)

        A = np.zeros((len(constraints), M))
        b = np.ones(len(constraints))

        for k, (i, j) in enumerate(constraints):
            A[k, i] = 1
            A[k, j] = 1

        lincons = LinearConstraint(A, ub=b)
        bounds = Bounds(lb=0, ub=1)

        # Try MILP with tighter tolerances and more iterations if available
        try:
            res = milp(c=c, constraints=lincons, integrality=np.ones(M), bounds=bounds,
                       options={'time_limit': 50, 'mip_gap': 1e-10})
        except TypeError:
            # Older scipy may not support these options
            res = milp(c=c, constraints=lincons, integrality=np.ones(M), bounds=bounds)

        if res.success and res.x is not None:
            # res.x contains the optimal values (0 or 1)
            # Due to floating point, check > 0.5
            selected_indices = [i for i, val in enumerate(res.x) if val > 0.5]
            if selected_indices:
                # Verify the solution is valid
                sol_set = set(valid_vertices[i] for i in selected_indices)
                valid = True
                for v in sol_set:
                    if 2 * v in squares:
                        valid = False
                        break
                    for u in sol_set:
                        if u >= v and u + v in squares:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    milp_solution = list(sol_set)
                else:
                    # Solution invalid, try to fix by removing problematic elements
                    fixed_set = set()
                    for idx in selected_indices:
                        v = valid_vertices[idx]
                        can_add = True
                        if 2 * v in squares:
                            can_add = False
                        else:
                            for u in fixed_set:
                                if u + v in squares:
                                    can_add = False
                                    break
                        if can_add:
                            fixed_set.add(v)
                    if fixed_set:
                        milp_solution = list(fixed_set)
                    else:
                        milp_solution = [valid_vertices[0]]
            else:
                milp_solution = [valid_vertices[0]]
        else:
            # If MILP fails, fall back to a simple greedy or local search
            pass
    except Exception:
        pass

    # 5. Build conflict sets (needed for both fallback and polish)
    conflict = {v: set() for v in valid_vertices}
    for i in range(len(valid_vertices)):
        vi = valid_vertices[i]
        for j in range(i + 1, len(valid_vertices)):
            vj = valid_vertices[j]
            if vi + vj in squares:
                conflict[vi].add(vj)
                conflict[vj].add(vi)

    def is_compatible(v, S):
        if 2 * v in squares:
            return False
        for u in S:
            if u in conflict[v]:
                return False
        return True

    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for v in valid_vertices:
                if v not in S:
                    if is_compatible(v, S):
                        S.add(v)
                        improved = True
            if not improved:
                best_move = None
                best_gain = 0
                for u in list(S):
                    S_temp = S - {u}
                    gain = 0
                    changed = True
                    while changed:
                        changed = False
                        for v in valid_vertices:
                            if v not in S_temp:
                                if is_compatible(v, S_temp):
                                    S_temp.add(v)
                                    changed = True
                                    gain += 1
                    if gain - 1 > best_gain:
                        best_gain = gain - 1
                        best_move = S_temp
                if best_move is not None and best_gain > 0:
                    S = best_move
                    improved = True
        return S

    # If MILP gave us a solution, use it as starting point for polish
    if milp_solution is not None:
        best_S = set(milp_solution)
        # Polish: try to add any compatible elements
        best_S = local_search(best_S)
    else:
        # Fallback: Greedy init + local search
        def greedy_init():
            sorted_v = sorted(valid_vertices, key=lambda x: len(conflict[x]))
            S = set()
            for v in sorted_v:
                if is_compatible(v, S):
                    S.add(v)
            return S

        best_S = greedy_init()
        best_S = local_search(best_S)

    # Quick randomized local search if time permits
    while time.time() < deadline - 2.0:
        if len(best_S) > 3:
            remove_count = random.randint(3, 5)
            removals = random.sample(list(best_S), min(remove_count, len(best_S)))
            perturbed_S = set(best_S)
            for r in removals:
                perturbed_S.remove(r)
            improved_S = local_search(perturbed_S)
            if len(improved_S) > len(best_S):
                best_S = improved_S

    return list(best_S)
