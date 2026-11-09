"""WalkSAT + DLS hybrid with adaptive restarts and polish for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from collections import defaultdict, deque


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # safety margin before 300s kill

    # Precompute: for each variable, which clauses contain it (and with what sign)
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_clauses[abs(lit) - 1].append((ci, lit > 0))

    # Spectral clustering: partition variables into weakly connected components
    # Build adjacency: two variables are connected if they share a clause
    var_to_clauses = defaultdict(set)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].add(ci)

    # BFS to find connected components based on shared clauses
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        # BFS
        comp = []
        queue = deque([start])
        visited[start] = True
        while queue:
            v = queue.popleft()
            comp.append(v)
            for ci in var_to_clauses[v]:
                for lit in clauses[ci]:
                    u = abs(lit) - 1
                    if not visited[u]:
                        visited[u] = True
                        queue.append(u)
        components.append(comp)

    # Sort components by size (solve larger ones first with more budget)
    components.sort(key=len, reverse=True)
    num_components = len(components)
    print(f"Found {num_components} components, sizes: {[len(c) for c in components[:10]]}...")

    # If there's only one component, proceed as before
    # If multiple, solve each independently
    if num_components > 1:
        # Allocate time budget proportional to component size
        total_size = sum(len(c) for c in components)
        time_budgets = []
        for comp in components:
            # Minimum 1s per component, rest proportional to size
            budget = max(1.0, 280.0 * len(comp) / total_size)
            time_budgets.append(budget)

        # Solve each component independently
        final_assignment = [False] * n
        for comp_idx, comp in enumerate(components):
            comp_vars = comp
            comp_size = len(comp_vars)

            # Build sub-instance for this component
            # Map old var indices to new 0-based indices
            var_map = {old: new for new, old in enumerate(comp_vars)}

            # Find clauses that only involve variables in this component
            comp_clauses = []
            for ci, clause in enumerate(clauses):
                if all(abs(lit) - 1 in var_map for lit in clause):
                    new_clause = [var_map[abs(lit) - 1] + (1 if lit > 0 else -1) for lit in clause]
                    comp_clauses.append(new_clause)

            if not comp_clauses:
                # No constraints, set all to False
                for v in comp_vars:
                    final_assignment[v] = False
                continue

            # Solve the sub-problem
            sub_instance = {
                "n": comp_size,
                "clauses": comp_clauses
            }

            # Use a dedicated solver with time budget
            sub_deadline = time.time() + time_budgets[comp_idx]
            sub_sol = _solve_subproblem(sub_instance, sub_deadline)

            # Map solution back
            for new_idx, old_idx in enumerate(comp_vars):
                final_assignment[old_idx] = sub_sol[new_idx]

        # Verify and return if all clauses satisfied
        if _is_satisfied(final_assignment, clauses):
            return final_assignment

    # If spectral decomposition didn't help or failed, fall back to full problem
    # Precompute clause lengths for DLS
    clause_lens = np.array([len(c) for c in clauses])

    # Build a lookup: for each clause, list of (var_idx, is_positive)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            cv.append((abs(lit) - 1, lit > 0))
        clause_vars.append(cv)

    # Precompute clause satisfaction counts for fast gains/losses
    # sat_count[ci] = number of literals in clause ci that are currently satisfied
    # This allows O(1) check of whether a clause is uniquely satisfied

    def make_satisfied_array(assignment):
        """Return (satisfied bool array, satisfaction count array)."""
        sat = np.zeros(m, dtype=bool)
        sat_count = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            cnt = 0
            for v, is_pos in clause_vars[ci]:
                if assignment[v] == is_pos:
                    cnt += 1
            sat[ci] = cnt > 0
            sat_count[ci] = cnt
        return sat, sat_count

    def flip_var(assignment, var_idx, sat, sat_count):
        """Flip variable var_idx and update sat/sat_count incrementally."""
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val
        for ci, is_pos in var_clauses[var_idx]:
            if is_pos == new_val:
                # This literal now satisfies the clause
                sat_count[ci] += 1
                sat[ci] = True
            else:
                # This literal no longer satisfies; decrement count
                sat_count[ci] -= 1
                if sat_count[ci] == 0:
                    sat[ci] = False

    def walksat_step(assignment, sat, sat_count, rng, p_walk):
        """One WalkSAT step: pick an unsat clause, flip a variable.
        With probability p_walk, use WalkSAT heuristic; otherwise flip randomly."""
        unsat_idx = np.where(~sat)[0]
        if len(unsat_idx) == 0:
            return
        ci = int(rng.choice(unsat_idx))
        clause = clause_vars[ci]
        if rng.random() < p_walk:
            # WalkSAT: pick variable with best (gains - losses)
            best_var = -1
            best_score = -1000
            for v, is_pos in clause:
                gains = 0
                losses = 0
                for c2, is_pos2 in var_clauses[v]:
                    if c2 == ci:
                        continue
                    if is_pos2 == assignment[v]:
                        # Currently satisfies c2; check if uniquely satisfied
                        if sat_count[c2] == 1:
                            losses += 1
                    else:
                        gains += 1
                score = gains - losses
                if score > best_score:
                    best_score = score
                    best_var = v
            if best_var >= 0:
                flip_var(assignment, best_var, sat, sat_count)
            else:
                v, _ = rng.choice(clause)
                flip_var(assignment, v, sat, sat_count)
        else:
            v, _ = rng.choice(clause)
            flip_var(assignment, v, sat, sat_count)

    def dls_step(assignment, sat, sat_count, rng):
        """DLS step: pick the shortest unsatisfied clause, flip a random var in it."""
        unsat_idx = np.where(~sat)[0]
        if len(unsat_idx) == 0:
            return
        # Find shortest unsat clause (use clause_lens for speed)
        min_len = 10
        best_ci = int(unsat_idx[0])
        for ci in unsat_idx:
            l = clause_lens[ci]
            if l < min_len:
                min_len = l
                best_ci = int(ci)
                if l <= 1:
                    break
        clause = clause_vars[best_ci]
        v, _ = rng.choice(clause)
        flip_var(assignment, v, sat, sat_count)

    def polish_pass(assignment, sat, sat_count, rng, max_iters=20000):
        """Aggressive polish: try to satisfy remaining unsat clauses with focused search."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            unsat_idx = np.where(~sat)[0]
            if len(unsat_idx) == 0:
                return assignment
            # Pick a random unsat clause
            ci = int(rng.choice(unsat_idx))
            clause = clause_vars[ci]
            # Try each variable in the clause, pick best gain
            best_var = -1
            best_score = -1000
            for v, is_pos in clause:
                gains = 0
                losses = 0
                for c2, is_pos2 in var_clauses[v]:
                    if c2 == ci:
                        continue
                    if is_pos2 == assignment[v]:
                        if sat_count[c2] == 1:
                            losses += 1
                    else:
                        gains += 1
                score = gains - losses
                if score > best_score:
                    best_score = score
                    best_var = v
            if best_var >= 0:
                flip_var(assignment, best_var, sat, sat_count)
            else:
                v, _ = rng.choice(clause)
                flip_var(assignment, v, sat, sat_count)
            # Check if solved
            if not np.any(~sat):
                return assignment
        return assignment

    def _is_satisfied(assignment, clauses):
        for clause in clauses:
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def _solve_subproblem(sub_inst, sub_deadline):
        """Solve a sub-problem with a dedicated time budget."""
        sn = sub_inst["n"]
        sclauses = sub_inst["clauses"]
        sm = len(sclauses)
        if sm == 0:
            return [False] * sn

        # Precompute for sub-problem
        svar_clauses = [[] for _ in range(sn)]
        for ci, clause in enumerate(sclauses):
            for lit in clause:
                svar_clauses[abs(lit) - 1].append((ci, lit > 0))

        sclause_vars = []
        for clause in sclauses:
            cv = []
            for lit in clause:
                cv.append((abs(lit) - 1, lit > 0))
            sclause_vars.append(cv)

        def make_sat(arr):
            sat = np.zeros(sm, dtype=bool)
            sat_count = np.zeros(sm, dtype=np.int32)
            for ci in range(sm):
                cnt = 0
                for v, is_pos in sclause_vars[ci]:
                    if arr[v] == is_pos:
                        cnt += 1
                sat[ci] = cnt > 0
                sat_count[ci] = cnt
            return sat, sat_count

        def flip(arr, vi, sat, sc):
            old = arr[vi]
            arr[vi] = not old
            for ci, is_pos in svar_clauses[vi]:
                if is_pos == arr[vi]:
                    sc[ci] += 1
                    sat[ci] = True
                else:
                    sc[ci] -= 1
                    if sc[ci] == 0:
                        sat[ci] = False

        best = None
        best_u = sm + 1
        for r in range(200):
            if time.time() > sub_deadline - 1:
                break
            rng = random.Random(r * 7919 + 13)
            arr = [rng.random() < 0.5 for _ in range(sn)]
            sat, sc = make_sat(arr)
            u = int(np.sum(~sat))
            if u < best_u:
                best_u = u
                best = arr[:]
            if u == 0:
                return arr

            for step in range(5000):
                if time.time() > sub_deadline - 1:
                    break
                unsat_idx = np.where(~sat)[0]
                if len(unsat_idx) == 0:
                    return arr
                ci = int(rng.choice(unsat_idx))
                cl = sclause_vars[ci]
                if rng.random() < 0.65:
                    bv, bs = -1, -1000
                    for v, ip in cl:
                        g, l = 0, 0
                        for c2, ip2 in svar_clauses[v]:
                            if c2 == ci:
                                continue
                            if ip2 == arr[v]:
                                if sc[c2] == 1:
                                    l += 1
                            else:
                                g += 1
                        s = g - l
                        if s > bs:
                            bs = s
                            bv = v
                    if bv >= 0:
                        flip(arr, bv, sat, sc)
                    else:
                        v, _ = rng.choice(cl)
                        flip(arr, v, sat, sc)
                else:
                    v, _ = rng.choice(cl)
                    flip(arr, v, sat, sc)
                if step % 20 == 0:
                    u = int(np.sum(~sat))
                    if u == 0:
                        return arr
                    if u < best_u:
                        best_u = u
                        best = arr[:]
        return best if best else [False] * sn

    best_assignment = None
    best_unsat = m + 1

    for restart in range(300):
        if time.time() > deadline:
            break
        rng = random.Random(restart * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat, sat_count = make_satisfied_array(assignment)
        cur_unsat = int(np.sum(~sat))
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        steps_in_restart = 0
        max_steps = 8000
        dls_interval = 150
        stall_count = 0
        prev_unsat = cur_unsat
        p_walk = 0.65

        for step in range(max_steps):
            if time.time() > deadline:
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                return best_assignment if best_assignment else assignment

            # WalkSAT step
            walksat_step(assignment, sat, sat_count, rng, p_walk)
            steps_in_restart += 1

            # Periodic DLS for escape
            if steps_in_restart % dls_interval == 0:
                dls_step(assignment, sat, sat_count, rng)

            # Check progress every 20 steps (more frequent for better adaptation)
            if step % 20 == 0:
                cur_unsat = int(np.sum(~sat))
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                if cur_unsat < prev_unsat:
                    stall_count = 0
                    p_walk = max(0.30, p_walk - 0.02)
                else:
                    stall_count += 1
                prev_unsat = cur_unsat

            # If we're very close (few unsat), do a polish pass
            if cur_unsat <= 5 and step % 100 == 0:
                assignment = polish_pass(assignment, sat, sat_count, rng, max_iters=5000)
                cur_unsat = int(np.sum(~sat))
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]

            # Adaptive restart within this restart attempt
            if stall_count > 12:
                break

        # After main loop, if we got close, try a final polish
        cur_unsat = int(np.sum(~sat))
        if cur_unsat <= 10:
            assignment = polish_pass(assignment, sat, sat_count, rng, max_iters=10000)
            cur_unsat = int(np.sum(~sat))
            if cur_unsat == 0:
                return assignment
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    # Final polish on best assignment found
    if best_assignment is not None and best_unsat <= 15:
        final_rng = random.Random(999999)
        best_sat, best_sat_count = make_satisfied_array(best_assignment)
        best_assignment = polish_pass(best_assignment, best_sat, best_sat_count, final_rng, max_iters=20000)
        if not np.any(~best_sat):
            return best_assignment

    return best_assignment if best_assignment else [False] * n
# EVOLVE-BLOCK-END
