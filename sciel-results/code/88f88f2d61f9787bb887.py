"""Rewritten planted 3-SAT solver: WalkSAT core + simulated-annealing polish."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)

    # Precompute clause data and sign lookup
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            v = abs(l) - 1
            s = (l > 0)
            cd.append((v, s))
        clause_data.append(cd)

    # var_to_clauses: list of clause indices for each variable
    var_to_clauses = [[] for _ in range(n)]
    for ci, cd in enumerate(clause_data):
        for v, s in cd:
            var_to_clauses[v].append(ci)

    # clause_var_sign[ci][v] = True/False/None
    clause_var_sign = [[None] * n for _ in range(num_clauses)]
    for ci, cd in enumerate(clause_data):
        for v, s in cd:
            clause_var_sign[ci][v] = s

    # Deadline guard: 330s wall-clock, leave 15s margin
    start_time = time.time()
    deadline = start_time + 315.0

    def check_time():
        return time.time() < deadline

    def init_state(assignment):
        sat_counts = [0] * num_clauses
        for ci, cd in enumerate(clause_data):
            cnt = 0
            for v, s in cd:
                if assignment[v] == s:
                    cnt += 1
            sat_counts[ci] = cnt
        satisfied = sum(1 for c in sat_counts if c > 0)
        return sat_counts, satisfied

    def flip_var(assignment, var_idx, sat_counts, satisfied):
        """Flip var_idx in assignment (caller already flipped it).
        Update sat_counts and satisfied. Return new satisfied."""
        old_val = not assignment[var_idx]  # the value before flip
        new_val = assignment[var_idx]
        for c_idx in var_to_clauses[var_idx]:
            s_c = clause_var_sign[c_idx][var_idx]
            if s_c is not None:
                was_sat = (old_val == s_c)
                is_sat = (new_val == s_c)
                if was_sat and not is_sat:
                    sat_counts[c_idx] -= 1
                    if sat_counts[c_idx] == 0:
                        satisfied -= 1
                elif not was_sat and is_sat:
                    sat_counts[c_idx] += 1
                    if sat_counts[c_idx] == 1:
                        satisfied += 1
        return satisfied

    def compute_breaks(assignment, var_idx, sat_counts):
        """Return number of currently-satisfied clauses that would become unsatisfied if var_idx is flipped."""
        # var_idx is currently at assignment[var_idx]; flipping changes to not assignment[var_idx]
        new_val = not assignment[var_idx]
        breaks = 0
        for c_idx in var_to_clauses[var_idx]:
            if sat_counts[c_idx] > 0:
                s_c = clause_var_sign[c_idx][var_idx]
                if s_c is not None and assignment[var_idx] == s_c:
                    # This literal is currently satisfied. If it's the only one, flipping breaks the clause.
                    if sat_counts[c_idx] == 1:
                        breaks += 1
        return breaks

    def compute_gains(assignment, var_idx, sat_counts):
        """Return number of currently-unsatisfied clauses that would become satisfied if var_idx is flipped."""
        new_val = not assignment[var_idx]
        gains = 0
        for c_idx in var_to_clauses[var_idx]:
            if sat_counts[c_idx] == 0:
                s_c = clause_var_sign[c_idx][var_idx]
                if s_c is not None and new_val == s_c:
                    gains += 1
        return gains

    def walksat_run(assignment, max_steps, p):
        """WalkSAT: pick random unsat clause, flip literal with prob p greedily (min breaks), else random."""
        sat_counts, satisfied = init_state(assignment)
        if satisfied == num_clauses:
            return True, assignment, satisfied, sat_counts

        for step in range(max_steps):
            if not check_time():
                return False, assignment, satisfied, sat_counts

            # Find unsat clauses
            unsat_indices = [i for i in range(num_clauses) if sat_counts[i] == 0]
            if not unsat_indices:
                return True, assignment, satisfied, sat_counts

            ci = random.choice(unsat_indices)
            cd = clause_data[ci]

            if random.random() < p:
                best_var = -1
                best_cost = float('inf')
                for v, s in cd:
                    cost = compute_breaks(assignment, v, sat_counts)
                    total_cost = cost + random.random()
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_var = v
                v_to_flip = best_var
            else:
                v_to_flip = random.choice(cd)[0]

            assignment[v_to_flip] = not assignment[v_to_flip]
            satisfied = flip_var(assignment, v_to_flip, sat_counts, satisfied)

            if satisfied == num_clauses:
                return True, assignment, satisfied, sat_counts

        return False, assignment, satisfied, sat_counts

    def anneal_run(assignment, max_steps, T_start, T_end, alpha):
        """Simulated annealing on Max-SAT objective (maximize satisfied clauses).
        At each step, pick a random variable, flip it, compute delta in satisfied count.
        Accept with probability 1 if delta >= 0, else exp(delta / T).
        T decays geometrically from T_start to T_end over max_steps.
        """
        sat_counts, satisfied = init_state(assignment)
        if satisfied == num_clauses:
            return True, assignment, satisfied, sat_counts

        best_satisfied = satisfied
        best_assignment = assignment[:]

        for step in range(max_steps):
            if not check_time():
                break

            # Temperature schedule: geometric
            if max_steps > 1:
                T = T_start * (T_end / T_start) ** (step / max_steps)
            else:
                T = T_end

            # Pick random variable
            v = random.randrange(n)

            # Compute delta: change in satisfied count if we flip v
            old_satisfied = satisfied
            # Tentatively flip
            assignment[v] = not assignment[v]
            new_satisfied = flip_var(assignment, v, sat_counts, satisfied)
            delta = new_satisfied - old_satisfied

            # Accept/reject
            if delta >= 0 or (T > 0 and random.random() < math.exp(delta / T)):
                satisfied = new_satisfied
                if satisfied > best_satisfied:
                    best_satisfied = satisfied
                    best_assignment = assignment[:]
            else:
                # Revert flip
                assignment[v] = not assignment[v]
                satisfied = flip_var(assignment, v, sat_counts, satisfied)

            if satisfied == num_clauses:
                return True, assignment, satisfied, sat_counts

        return False, best_assignment, best_satisfied, sat_counts

    # Main search strategy
    best_assignment = None
    best_satisfied = -1

    p_values = [0.5, 0.7, 0.8]
    steps_list = [500, 2000, 10000]

    # Phase 1: WalkSAT restarts
    for p in p_values:
        for steps in steps_list:
            if not check_time():
                break
            while check_time():
                assignment = [random.random() < 0.5 for _ in range(n)]
                solved, assignment, satisfied, _ = walksat_run(assignment, steps, p)
                if satisfied > best_satisfied:
                    best_satisfied = satisfied
                    best_assignment = assignment[:]
                if solved:
                    return assignment

    # Phase 2: Simulated annealing polish from best WalkSAT solution
    if best_assignment is not None and check_time():
        # Run several annealing attempts from the best solution and from random starts
        for attempt in range(10):
            if not check_time():
                break
            if attempt == 0:
                assignment = best_assignment[:]
            else:
                assignment = [random.random() < 0.5 for _ in range(n)]
            T_start = 5.0
            T_end = 0.01
            max_steps = 50000
            solved, assignment, satisfied, _ = anneal_run(assignment, max_steps, T_start, T_end, 0.999)
            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_assignment = assignment[:]
            if solved:
                return assignment

    # Phase 3: More WalkSAT with longer runs if time remains
    if check_time():
        while check_time():
            assignment = [random.random() < 0.5 for _ in range(n)]
            solved, assignment, satisfied, _ = walksat_run(assignment, 50000, 0.7)
            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_assignment = assignment[:]
            if solved:
                return assignment

    if best_assignment is not None:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
