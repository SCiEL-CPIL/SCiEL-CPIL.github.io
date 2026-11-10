"""3-SAT solver with hybrid WalkSAT + restarts + deadline guard."""
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["instance_clauses"] if "instance_clauses" in instance else instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 330s kill

    # Preprocess: remove tautological clauses, dedupe
    seen = set()
    filtered_clauses = []
    for c in clauses:
        # Check for tautology (both x and -x in clause)
        lits = set(c)
        if any(-l in lits for l in lits):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            filtered_clauses.append(c)
    clauses = filtered_clauses

    # Build clause index for efficiency
    clause_count = len(clauses)
    if clause_count == 0:
        return [True] * n

    # Variable occurrence counts for heuristics
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
    # Precompute for each clause: list of (var, is_positive) for faster evaluation
    clause_vars = []
    for i, c in enumerate(clauses):
        var_list = []
        for l in c:
            idx = abs(l) - 1
            var_list.append((idx, l > 0))
            if l > 0:
                var_pos[idx].append(i)
            else:
                var_neg[idx].append(i)
        clause_vars.append(var_list)

    best_assignment = None
    best_unsat_count = clause_count + 1

    def count_unsat(assignment):
        count = 0
        for cv in clause_vars:
            sat = False
            for idx, is_pos in cv:
                if assignment[idx] == is_pos:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def satisfied(assignment, c):
        for l in c:
            idx = abs(l) - 1
            if assignment[idx] == (l > 0):
                return True
        return False

    # Phase 1: Try random restarts with WalkSAT using incremental clause tracking
    rng = random.Random(42)
    max_restarts = 5000
    steps_per_restart = 8000
    noise = 0.25  # probability of random flip

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment
        if restart == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Perturb from best or random
            if best_assignment is not None and rng.random() < 0.7:
                assignment = best_assignment[:]
                # Flip a few random variables
                for _ in range(n // 4 + 1):
                    assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        # Build sat_count for each clause and track unsat clauses
        sat_count = [0] * clause_count
        unsat_clauses = []
        for i, cv in enumerate(clause_vars):
            cnt = 0
            for idx, is_pos in cv:
                if assignment[idx] == is_pos:
                    cnt += 1
            sat_count[i] = cnt
            if cnt == 0:
                unsat_clauses.append(i)

        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat_clauses:
                return assignment

            # With some probability, do a random flip
            if rng.random() < noise:
                var = rng.randint(0, n - 1)
                old_val = assignment[var]
                assignment[var] = not old_val

                # Update sat_count and unsat_clauses for affected clauses
                new_unsat = []
                for ci in var_pos[var]:
                    old_sat = sat_count[ci]
                    if old_val:
                        new_sat = old_sat - 1
                    else:
                        new_sat = old_sat + 1
                    sat_count[ci] = new_sat
                    if new_sat == 0:
                        new_unsat.append(ci)
                
                for ci in var_neg[var]:
                    old_sat = sat_count[ci]
                    if old_val:
                        new_sat = old_sat + 1
                    else:
                        new_sat = old_sat - 1
                    sat_count[ci] = new_sat
                    if new_sat == 0:
                        new_unsat.append(ci)
                
                # Rebuild unsat_clauses efficiently
                new_unsat_set = set(new_unsat)
                updated_unsat = []
                for ci in unsat_clauses:
                    if sat_count[ci] == 0:
                        updated_unsat.append(ci)
                for ci in new_unsat:
                    if ci not in updated_unsat:
                        updated_unsat.append(ci)
                unsat_clauses = updated_unsat
                continue

            # Choose a random unsatisfied clause
            clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            clause = clauses[clause_idx]

            # Score each variable in the clause: count how many other clauses it would satisfy
            best_var = -1
            best_score = -1
            for l in clause:
                var = abs(l) - 1
                score = 0
                target_val = (l > 0)
                if assignment[var] != target_val:
                    # Flipping would make this literal true
                    # Count unsat clauses where this is the only unsat literal
                    affected = var_pos[var] if l > 0 else var_neg[var]
                    for ci in affected:
                        if sat_count[ci] == 0:
                            # Check if this literal is the only unsat one in clause ci
                            cv2 = clause_vars[ci]
                            only_unsat = True
                            for idx2, is_pos2 in cv2:
                                if idx2 == var and is_pos2 == target_val:
                                    continue
                                if assignment[idx2] == is_pos2:
                                    only_unsat = False
                                    break
                            if only_unsat:
                                score += 1

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var >= 0:
                var_to_flip = best_var
            else:
                l = clause[rng.randint(0, len(clause) - 1)]
                var_to_flip = abs(l) - 1

            # Apply the flip
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val

            # Update sat_count and unsat_clauses for affected clauses
            new_unsat = []
            for ci in var_pos[var_to_flip]:
                old_sat = sat_count[ci]
                if old_val:
                    new_sat = old_sat - 1
                else:
                    new_sat = old_sat + 1
                sat_count[ci] = new_sat
                if new_sat == 0:
                    new_unsat.append(ci)
            
            for ci in var_neg[var_to_flip]:
                old_sat = sat_count[ci]
                if old_val:
                    new_sat = old_sat + 1
                else:
                    new_sat = old_sat - 1
                sat_count[ci] = new_sat
                if new_sat == 0:
                    new_unsat.append(ci)
            
            # Rebuild unsat_clauses
            new_unsat_set = set(new_unsat)
            updated_unsat = []
            for ci in unsat_clauses:
                if sat_count[ci] == 0:
                    updated_unsat.append(ci)
            for ci in new_unsat:
                if ci not in updated_unsat:
                    updated_unsat.append(ci)
            unsat_clauses = updated_unsat

            # Periodically check unsat count
            if step % 500 == 0:
                current_unsat = len(unsat_clauses)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]

    # Phase 2: Smooth penalty-based local search from best assignment
    if best_assignment is not None:
        assignment = best_assignment[:]
        
        # Smooth local search using soft penalty
        # Objective: minimize sum of exp(-alpha * sat_count) for each clause
        # This is smooth and differentiable in spirit
        alpha = 5.0  # temperature parameter
        max_iter = 50000
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute current penalty and identify unsat clauses
            unsat_clauses = []
            total_penalty = 0.0
            for i, cv in enumerate(clause_vars):
                cnt = 0
                for idx, is_pos in cv:
                    if assignment[idx] == is_pos:
                        cnt += 1
                if cnt == 0:
                    unsat_clauses.append(i)
                    total_penalty += math.exp(alpha)  # fully unsat
                else:
                    total_penalty += math.exp(-alpha * cnt)
            
            if not unsat_clauses:
                return assignment
            
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]
            
            # Greedy: find variable flip that minimizes increase in penalty
            # Limit search to variables in unsat clauses for efficiency
            candidate_vars = set()
            for ci in unsat_clauses[:50]:  # limit to 50 unsat clauses
                for idx, is_pos in clause_vars[ci]:
                    candidate_vars.add(idx)
            
            if not candidate_vars:
                break
            
            best_var = -1
            best_delta = float('inf')
            
            for var in list(candidate_vars)[:200]:  # limit candidates
                old_val = assignment[var]
                # Compute delta penalty if we flip var
                delta = 0.0
                for ci in var_pos[var]:
                    old_cnt = sat_count[ci] if it < max_iter else 0
                    # Recompute current sat count for this clause
                    cnt = 0
                    for idx2, is_pos2 in clause_vars[ci]:
                        if assignment[idx2] == is_pos2:
                            cnt += 1
                    if old_val:
                        new_cnt = cnt - 1
                    else:
                        new_cnt = cnt + 1
                    delta += math.exp(-alpha * max(0, new_cnt)) - math.exp(-alpha * max(0, cnt))
                
                for ci in var_neg[var]:
                    cnt = 0
                    for idx2, is_pos2 in clause_vars[ci]:
                        if assignment[idx2] == is_pos2:
                            cnt += 1
                    if old_val:
                        new_cnt = cnt + 1
                    else:
                        new_cnt = cnt - 1
                    delta += math.exp(-alpha * max(0, new_cnt)) - math.exp(-alpha * max(0, cnt))
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var
            
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
                # Update sat_count
                for ci in var_pos[best_var]:
                    if not assignment[best_var]:  # was True, now False
                        sat_count[ci] -= 1
                    else:
                        sat_count[ci] += 1
                for ci in var_neg[best_var]:
                    if assignment[best_var]:  # was False, now True
                        sat_count[ci] += 1
                    else:
                        sat_count[ci] -= 1
                # Clamp
                for ci in range(clause_count):
                    sat_count[ci] = max(0, min(3, sat_count[ci]))
            
            # Early exit if no improvement possible
            if best_delta >= 0 and len(unsat_clauses) > 0:
                # Try a random perturbation
                if rng.random() < 0.3:
                    var = rng.randint(0, n - 1)
                    assignment[var] = not assignment[var]
                    for ci in var_pos[var]:
                        if not assignment[var]:
                            sat_count[ci] -= 1
                        else:
                            sat_count[ci] += 1
                    for ci in var_neg[var]:
                        if assignment[var]:
                            sat_count[ci] += 1
                        else:
                            sat_count[ci] -= 1
                    for ci in range(clause_count):
                        sat_count[ci] = max(0, min(3, sat_count[ci]))

    # Return best found or random
    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
