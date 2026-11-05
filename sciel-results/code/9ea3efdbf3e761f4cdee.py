"""Improved planted 3-SAT solver: Fast WalkSAT with tracked unsat clauses + local polish."""
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
        unsat_list = []
        for ci, cd in enumerate(clause_data):
            cnt = 0
            for v, s in cd:
                if assignment[v] == s:
                    cnt += 1
            sat_counts[ci] = cnt
            if cnt == 0:
                unsat_list.append(ci)
        satisfied = num_clauses - len(unsat_list)
        return sat_counts, satisfied, unsat_list

    def flip_var(assignment, var_idx, sat_counts, unsat_list, satisfied):
        """Flip var_idx in assignment (caller already flipped it).
        Update sat_counts, unsat_list, and satisfied. Return new satisfied."""
        old_val = not assignment[var_idx]
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
                        unsat_list.append(c_idx)
                elif not was_sat and is_sat:
                    sat_counts[c_idx] += 1
                    if sat_counts[c_idx] == 1:
                        satisfied += 1
                        # Remove from unsat_list (lazy: mark as -1, clean later)
                        unsat_list.remove(c_idx)
        return satisfied

    def compute_breaks(assignment, var_idx, sat_counts):
        breaks = 0
        for c_idx in var_to_clauses[var_idx]:
            if sat_counts[c_idx] == 1:
                s_c = clause_var_sign[c_idx][var_idx]
                if s_c is not None and assignment[var_idx] == s_c:
                    breaks += 1
        return breaks

    def compute_gains(assignment, var_idx, sat_counts):
        gains = 0
        new_val = not assignment[var_idx]
        for c_idx in var_to_clauses[var_idx]:
            if sat_counts[c_idx] == 0:
                s_c = clause_var_sign[c_idx][var_idx]
                if s_c is not None and new_val == s_c:
                    gains += 1
        return gains

    def walksat_run(assignment, max_steps, p):
        """WalkSAT with tracked unsat clause list for efficiency."""
        sat_counts, satisfied, unsat_list = init_state(assignment)
        if satisfied == num_clauses:
            return True, assignment, satisfied, sat_counts

        for step in range(max_steps):
            if not check_time():
                return False, assignment, satisfied, sat_counts

            # Clean up unsat_list if it has stale entries
            if len(unsat_list) > 0 and random.random() < 0.1:
                unsat_list = [i for i in unsat_list if sat_counts[i] == 0]
            
            if not unsat_list:
                return True, assignment, satisfied, sat_counts

            ci = random.choice(unsat_list)
            # Verify it's still unsat
            if sat_counts[ci] != 0:
                unsat_list.remove(ci)
                continue

            cd = clause_data[ci]

            if random.random() < p:
                # Score each literal: prefer low breaks, high gains
                best_var = -1
                best_score = float('inf')
                for v, s in cd:
                    breaks = compute_breaks(assignment, v, sat_counts)
                    gains = compute_gains(assignment, v, sat_counts)
                    score = breaks - gains + random.random()
                    if score < best_score:
                        best_score = score
                        best_var = v
                v_to_flip = best_var
            else:
                v_to_flip = random.choice(cd)[0]

            assignment[v_to_flip] = not assignment[v_to_flip]
            satisfied = flip_var(assignment, v_to_flip, sat_counts, unsat_list, satisfied)

            if satisfied == num_clauses:
                return True, assignment, satisfied, sat_counts

        return False, assignment, satisfied, sat_counts

    def local_polish(assignment, max_rounds=100):
        """Aggressive local search: try all single flips, then random pairs.
        Only accept improvements. Returns (solved, assignment, satisfied, sat_counts)."""
        sat_counts, satisfied, _ = init_state(assignment)
        if satisfied == num_clauses:
            return True, assignment, satisfied, sat_counts

        for rnd in range(max_rounds):
            if not check_time():
                break
            improved = False

            # Try all single flips
            for v in range(n):
                old_val = assignment[v]
                assignment[v] = not old_val
                new_sat = flip_var(assignment, v, sat_counts, [], satisfied)
                delta = new_sat - satisfied
                if delta > 0:
                    satisfied = new_sat
                    improved = True
                    if satisfied == num_clauses:
                        return True, assignment, satisfied, sat_counts
                else:
                    # Revert
                    assignment[v] = old_val
                    satisfied = flip_var(assignment, v, sat_counts, [], satisfied)

            if not improved:
                # Try some random 2-flips
                found_improvement = False
                for _ in range(min(50, n * (n-1) // 2)):
                    v1 = random.randrange(n)
                    v2 = random.randrange(n)
                    if v1 == v2:
                        continue
                    old1, old2 = assignment[v1], assignment[v2]
                    assignment[v1] = not old1
                    assignment[v2] = not old2
                    s1 = flip_var(assignment, v1, sat_counts, [], satisfied)
                    s2 = flip_var(assignment, v2, sat_counts, [], s1)
                    delta = s2 - satisfied
                    if delta > 0:
                        satisfied = s2
                        improved = True
                        found_improvement = True
                        if satisfied == num_clauses:
                            return True, assignment, satisfied, sat_counts
                    else:
                        assignment[v2] = old2
                        s1 = flip_var(assignment, v2, sat_counts, [], s1)
                        assignment[v1] = old1
                        satisfied = flip_var(assignment, v1, sat_counts, [], s1)
                
                if not improved:
                    break

        return False, assignment, satisfied, sat_counts

    def anneal_run(assignment, max_steps, T_start, T_end):
        """Simulated annealing on Max-SAT objective."""
        sat_counts, satisfied, _ = init_state(assignment)
        if satisfied == num_clauses:
            return True, assignment, satisfied, sat_counts

        best_satisfied = satisfied
        best_assignment = assignment[:]

        for step in range(max_steps):
            if not check_time():
                break

            if max_steps > 1:
                T = T_start * (T_end / T_start) ** (step / max_steps)
            else:
                T = T_end

            v = random.randrange(n)
            old_satisfied = satisfied
            assignment[v] = not assignment[v]
            new_satisfied = flip_var(assignment, v, sat_counts, [], satisfied)
            delta = new_satisfied - old_satisfied

            if delta >= 0 or (T > 0 and random.random() < math.exp(delta / T)):
                satisfied = new_satisfied
                if satisfied > best_satisfied:
                    best_satisfied = satisfied
                    best_assignment = assignment[:]
            else:
                assignment[v] = not assignment[v]
                satisfied = flip_var(assignment, v, sat_counts, [], satisfied)

            if satisfied == num_clauses:
                return True, assignment, satisfied, sat_counts

        return False, best_assignment, best_satisfied, sat_counts

    # Main search strategy
    best_assignment = None
    best_satisfied = -1

    # Phase 1: WalkSAT with varied parameters
    p_values = [0.4, 0.6, 0.7, 0.8, 0.9]
    steps_list = [200, 500, 1000, 5000]

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

    # Phase 2: Local polish on best solution
    if best_assignment is not None and check_time():
        solved, assignment, satisfied, _ = local_polish(best_assignment[:], max_rounds=200)
        if satisfied > best_satisfied:
            best_satisfied = satisfied
            best_assignment = assignment[:]
        if solved:
            return assignment

    # Phase 3: Simulated annealing from best and random starts
    if check_time():
        for attempt in range(15):
            if not check_time():
                break
            if attempt == 0 and best_assignment is not None:
                assignment = best_assignment[:]
            else:
                assignment = [random.random() < 0.5 for _ in range(n)]
            T_start = 5.0
            T_end = 0.01
            max_steps = 30000
            solved, assignment, satisfied, _ = anneal_run(assignment, max_steps, T_start, T_end)
            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_assignment = assignment[:]
            if solved:
                return assignment
            # Polish after annealing
            if check_time():
                solved2, assignment, satisfied2, _ = local_polish(assignment, max_rounds=50)
                if satisfied2 > best_satisfied:
                    best_satisfied = satisfied2
                    best_assignment = assignment[:]
                if solved2:
                    return assignment

    # Phase 4: Long WalkSAT runs with polish
    if check_time():
        while check_time():
            assignment = [random.random() < 0.5 for _ in range(n)]
            solved, assignment, satisfied, _ = walksat_run(assignment, 20000, 0.7)
            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_assignment = assignment[:]
            if solved:
                return assignment
            # Polish
            if check_time():
                solved2, assignment, satisfied2, _ = local_polish(assignment, max_rounds=100)
                if satisfied2 > best_satisfied:
                    best_satisfied = satisfied2
                    best_assignment = assignment[:]
                if solved2:
                    return assignment

    if best_assignment is not None:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
