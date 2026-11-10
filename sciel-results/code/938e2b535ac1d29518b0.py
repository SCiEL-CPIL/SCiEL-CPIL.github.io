import time
import math
import random
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

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
    conflict_set = set()
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

    s = greedy_by_degree()
    s = try_add_all(s)
    initializations.append(s)

    s = greedy_by_degree_desc()
    s = try_add_all(s)
    initializations.append(s)

    for _ in range(5):
        s = greedy_randomized()
        s = try_add_all(s)
        initializations.append(s)

    for s in initializations:
        if len(s) > best_size:
            best_solution = s.copy()
            best_size = len(s)

    # Phase 2: Custom Branch-and-Bound with LP relaxation
    if time.time() < deadline - 15:
        try:
            # Build constraint matrix for LP relaxation
            constraint_pairs = [tuple(p) for p in conflict_set]
            m_constraints = len(constraint_pairs)

            if m_constraints > 0:
                A_ub = np.zeros((m_constraints, n_vars), dtype=np.float64)
                b_ub = np.ones(m_constraints, dtype=np.float64)

                for k, (i, j) in enumerate(constraint_pairs):
                    A_ub[k, i] = 1.0
                    A_ub[k, j] = 1.0

                c_obj = -np.ones(n_vars, dtype=np.float64)

                # Solve LP relaxation to get upper bound
                lp_res = linprog(
                    c_obj,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=[(0, 1)] * n_vars,
                    method='highs'
                )

                lp_bound = -lp_res.fun if lp_res.success else float('inf')

                # Branch-and-Bound
                # State: (node_id, fixed_in, fixed_out, lp_bound)
                # We'll use a priority queue to process nodes with highest bound first
                import heapq

                # For efficiency, we'll do a DFS-style B&B with pruning
                # Node: (in_set, out_set, remaining_vars)
                # We branch on the most fractional variable

                best_bb_size = best_size
                best_bb_set = best_solution.copy()

                # Time limit for B&B
                bb_deadline = deadline - 5
                nodes_processed = 0
                max_nodes = 50000  # Safety limit

                # Stack for DFS: (in_set, out_set)
                # in_set: variables forced to be in the solution
                # out_set: variables forced to be out of the solution
                stack = [(set(), set())]

                while stack and time.time() < bb_deadline and nodes_processed < max_nodes:
                    in_set, out_set = stack.pop()
                    nodes_processed += 1

                    # Remaining variables: not in in_set and not in out_set
                    remaining = [i for i in range(n_vars) if i not in in_set and i not in out_set]

                    if not remaining:
                        # Leaf node: evaluate solution
                        sol = in_set.copy()
                        sol = try_add_all(sol)
                        if len(sol) > best_bb_size:
                            best_bb_size = len(sol)
                            best_bb_set = sol.copy()
                        continue

                    # Check if any remaining variable conflicts with in_set
                    # If so, it must be excluded
                    new_out = set()
                    for idx in remaining:
                        for neighbor in adj[idx]:
                            if neighbor in in_set:
                                new_out.add(idx)
                                break

                    if new_out:
                        out_set = out_set | new_out
                        remaining = [i for i in remaining if i not in new_out]
                        if not remaining:
                            sol = in_set.copy()
                            sol = try_add_all(sol)
                            if len(sol) > best_bb_size:
                                best_bb_size = len(sol)
                                best_bb_set = sol.copy()
                            continue

                    # Upper bound: |in_set| + sum of LP solution over remaining
                    # Quick bound: |in_set| + len(remaining) (trivial)
                    # Better: solve LP on subproblem
                    if len(remaining) > 30 and nodes_processed % 10 == 0:
                        # Solve LP relaxation for this subproblem
                        sub_vars = remaining
                        n_sub = len(sub_vars)
                        sub_idx_map = {v: i for i, v in enumerate(sub_vars)}

                        # Build sub-problem constraints
                        sub_A = []
                        sub_b = []
                        for i, vi in enumerate(sub_vars):
                            for j in range(i + 1, n_sub):
                                vj = sub_vars[j]
                                if vi + vj in squares:
                                    row = [0.0] * n_sub
                                    row[i] = 1.0
                                    row[j] = 1.0
                                    sub_A.append(row)
                                    sub_b.append(1.0)

                        if sub_A:
                            sub_A = np.array(sub_A, dtype=np.float64)
                            sub_b = np.array(sub_b, dtype=np.float64)
                            sub_c = -np.ones(n_sub, dtype=np.float64)

                            sub_res = linprog(
                                sub_c,
                                A_ub=sub_A,
                                b_ub=sub_b,
                                bounds=[(0, 1)] * n_sub,
                                method='highs'
                            )

                            if sub_res.success:
                                lp_sub_bound = -sub_res.fun
                                upper_bound = len(in_set) + lp_sub_bound
                                if upper_bound <= best_bb_size:
                                    continue  # Prune this node
                        else:
                            # No conflicts among remaining, take all
                            sol = in_set | set(remaining)
                            sol = try_add_all(sol)
                            if len(sol) > best_bb_size:
                                best_bb_size = len(sol)
                                best_bb_set = sol.copy()
                            continue

                    # Branch on a variable
                    # Choose the variable with highest degree in the subgraph
                    # or the most fractional from LP (simplified: pick highest degree)
                    branch_var = max(remaining, key=lambda x: len([n for n in adj[x] if n in remaining]))

                    # Branch 1: include branch_var
                    in_set1 = in_set | {branch_var}
                    # Exclude all neighbors of branch_var
                    out_set1 = out_set | set(adj[branch_var])

                    # Branch 2: exclude branch_var
                    in_set2 = in_set
                    out_set2 = out_set | {branch_var}

                    # Push to stack (include branch first as it's often better)
                    stack.append((in_set2, out_set2))
                    stack.append((in_set1, out_set1))

                # Update best solution from B&B
                if best_bb_size > best_size:
                    best_solution = best_bb_set.copy()
                    best_size = best_bb_size

        except Exception:
            pass

    # Phase 3: Local search refinement on best solution
    if time.time() < deadline - 5:
        best_solution = one_swap_search(best_solution, max_rounds=5)
        best_size = len(best_solution)

        if time.time() < deadline - 5:
            best_solution = two_swap_search(best_solution, max_rounds=3)
            best_size = len(best_solution)

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
