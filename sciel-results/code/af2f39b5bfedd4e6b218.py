# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)

    # Precompute variable to clause mapping and literal signs
    var_to_clauses = [[] for _ in range(n)]
    clause_var_info = []
    var_signs = [{} for _ in range(n)]

    for c_idx, clause in enumerate(clauses):
        info = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            info.append((v, s))
            var_to_clauses[v].append(c_idx)
            var_signs[v][c_idx] = s
        clause_var_info.append(info)

    # Deadline: 328 seconds to be safe against the 330s kill
    deadline = time.time() + 328.0
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters - tuned for planted 3-SAT
    base_noise = 0.1
    max_noise = 0.7
    restart_threshold = 1500

    while time.time() < deadline:
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Calculate initial clause weights and track unsat clauses
        clause_weights = [0] * len(clauses)
        unsat_set = set()
        for i, clause_info in enumerate(clause_var_info):
            w = 0
            for v, s in clause_info:
                if assignment[v] == (s == 1):
                    w += 1
            clause_weights[i] = w
            if w == 0:
                unsat_set.add(i)

        total_unsat = len(unsat_set)

        if total_unsat == 0:
            return assignment

        steps_since_improvement = 0
        min_unsat_seen = total_unsat
        step_count = 0

        for _ in range(200000):
            step_count += 1
            # Check time every 200 steps to reduce overhead
            if step_count % 200 == 0 and time.time() >= deadline:
                break

            if total_unsat == 0:
                return assignment

            # Track best solution
            if total_unsat < best_unsat_count:
                best_unsat_count = total_unsat
                best_assignment = assignment[:]

            if total_unsat < min_unsat_seen:
                min_unsat_seen = total_unsat
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            if steps_since_improvement > restart_threshold:
                break  # Restart with new random assignment

            if total_unsat == 0:
                return assignment

            # Dynamic noise: increase noise as we get stuck
            progress_ratio = min(steps_since_improvement / restart_threshold, 1.0)
            current_noise = base_noise + (max_noise - base_noise) * progress_ratio

            # Use set-based selection for O(1) membership
            unsat_list = list(unsat_set)
            clause_idx = unsat_list[rng.randrange(total_unsat)]
            clause_info = clause_var_info[clause_idx]

            if rng.random() < current_noise:
                # Random walk step
                v, s = clause_info[rng.randrange(3)]
                var_idx = v
            else:
                # Greedy step: Find variable in clause that gives best gain
                best_gain = float('inf')
                best_var_idx = -1

                for v, s in clause_info:
                    var_idx = v

                    # Calculate delta in total_unsat
                    delta = 0
                    for c_idx in var_to_clauses[var_idx]:
                        old_w = clause_weights[c_idx]
                        c_sign = var_signs[var_idx][c_idx]

                        # Is the literal satisfied currently?
                        is_sat = (assignment[var_idx] == (c_sign == 1))

                        if is_sat:
                            if old_w == 1:
                                delta += 1
                        else:
                            if old_w == 0:
                                delta -= 1

                    if delta < best_gain:
                        best_gain = delta
                        best_var_idx = var_idx
                        if best_gain == -1:
                            break

                var_idx = best_var_idx

            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]

            # Update clause weights and unsat set
            for c_idx in var_to_clauses[var_idx]:
                old_w = clause_weights[c_idx]
                c_sign = var_signs[var_idx][c_idx]

                # Is the literal satisfied NOW (after flip)?
                is_sat = (assignment[var_idx] == (c_sign == 1))

                if is_sat:
                    new_w = old_w + 1
                else:
                    new_w = old_w - 1

                clause_weights[c_idx] = new_w

                # Update unsat set
                if old_w == 0 and new_w > 0:
                    unsat_set.discard(c_idx)
                    total_unsat -= 1
                elif old_w > 0 and new_w == 0:
                    unsat_set.add(c_idx)
                    total_unsat += 1

    return best_assignment
# EVOLVE-BLOCK-END
