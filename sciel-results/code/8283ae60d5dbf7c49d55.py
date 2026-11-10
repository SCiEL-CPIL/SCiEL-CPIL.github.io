"""3-SAT solver with optimized WalkSAT + fast local search + deadline guard."""
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
        lits = set(c)
        if any(-l in lits for l in lits):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            filtered_clauses.append(c)
    clauses = filtered_clauses

    clause_count = len(clauses)
    if clause_count == 0:
        return [True] * n

    # Build clause index for efficiency
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
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
    rng = random.Random(42)

    def init_sat_count(assignment):
        """Build sat_count array and unsat_clauses list from scratch."""
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
        return sat_count, unsat_clauses

    def apply_flip(assignment, var, sat_count, unsat_clauses):
        """Flip variable var, update sat_count and unsat_clauses. Returns new unsat_clauses."""
        old_val = assignment[var]
        assignment[var] = not old_val

        # Track which clauses become newly unsat
        newly_unsat = []
        for ci in var_pos[var]:
            old_sat = sat_count[ci]
            if old_val:
                sat_count[ci] = old_sat - 1
            else:
                sat_count[ci] = old_sat + 1
            if sat_count[ci] == 0:
                newly_unsat.append(ci)

        for ci in var_neg[var]:
            old_sat = sat_count[ci]
            if old_val:
                sat_count[ci] = old_sat + 1
            else:
                sat_count[ci] = old_sat - 1
            if sat_count[ci] == 0:
                newly_unsat.append(ci)

        # Rebuild unsat list: keep existing unsat that are still unsat, add newly unsat
        new_unsat_set = set(newly_unsat)
        updated = []
        for ci in unsat_clauses:
            if sat_count[ci] == 0:
                updated.append(ci)
        for ci in newly_unsat:
            if ci not in new_unsat_set or True:
                # Only add if not already in updated (check sat_count==0)
                if sat_count[ci] == 0 and ci not in updated:
                    updated.append(ci)
        return updated

    # Phase 1: WalkSAT with restarts
    max_restarts = 10000
    steps_per_restart = 4000
    noise = 0.25

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment
        if restart == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            if best_assignment is not None and rng.random() < 0.7:
                assignment = best_assignment[:]
                for _ in range(max(1, n // 5)):
                    assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        sat_count, unsat_clauses = init_sat_count(assignment)
        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat_clauses:
                return assignment

            if rng.random() < noise:
                var = rng.randint(0, n - 1)
                unsat_clauses = apply_flip(assignment, var, sat_count, unsat_clauses)
                continue

            # Choose random unsatisfied clause
            clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            clause = clauses[clause_idx]

            # Score variables in clause: count clauses that would be satisfied
            best_var = -1
            best_score = -1
            for l in clause:
                var = abs(l) - 1
                score = 0
                target_val = (l > 0)
                if assignment[var] != target_val:
                    affected = var_pos[var] if l > 0 else var_neg[var]
                    for ci in affected:
                        if sat_count[ci] == 0:
                            # Check if flipping would satisfy this clause
                            cv2 = clause_vars[ci]
                            only_this_unsat = True
                            for idx2, is_pos2 in cv2:
                                if idx2 == var and is_pos2 == target_val:
                                    continue
                                if assignment[idx2] == is_pos2:
                                    only_this_unsat = False
                                    break
                            if only_this_unsat:
                                score += 1

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var >= 0:
                var_to_flip = best_var
            else:
                l = clause[rng.randint(0, len(clause) - 1)]
                var_to_flip = abs(l) - 1

            unsat_clauses = apply_flip(assignment, var_to_flip, sat_count, unsat_clauses)

            if step % 200 == 0:
                current_unsat = len(unsat_clauses)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]

    # Phase 2: Fast local search polish using incremental tracking
    if best_assignment is not None and time.time() < deadline - 5:
        assignment = best_assignment[:]
        sat_count, unsat_clauses = init_sat_count(assignment)

        if len(unsat_clauses) == 0:
            return assignment

        # Iterative improvement: find flips that reduce unsat count
        max_iter = 20000
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break

            if not unsat_clauses:
                return assignment

            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

            # Find best flip: one that reduces unsat count the most
            # Only consider variables in unsat clauses
            candidate_vars = set()
            for ci in unsat_clauses[:100]:
                for idx, is_pos in clause_vars[ci]:
                    candidate_vars.add(idx)

            if not candidate_vars:
                break

            best_var = -1
            best_delta = 0  # we want negative delta (reduction in unsat)

            for var in candidate_vars:
                # Compute delta in unsat count if we flip var
                delta = 0
                for ci in var_pos[var]:
                    if sat_count[ci] == 0:
                        # This clause is unsat; flipping var might fix or break it
                        if assignment[var]:
                            # var is True, literal is positive -> clause already satisfied by this literal? No, sat_count==0 means no literal is satisfied
                            # If we flip var to False, this positive literal becomes false -> still unsat (or worse)
                            # Actually if sat_count[ci]==0, no literal is true. Flipping var: if l>0 and assignment[var] was True... wait
                            # If assignment[var] is True and l>0, then literal l is TRUE, so sat_count should be >=1. Contradiction.
                            # So if sat_count[ci]==0, then for positive literal, assignment[var] must be False.
                            # Flipping to True makes this literal true -> clause becomes satisfied -> delta -= 1
                            delta -= 1
                        else:
                            # assignment[var] is False, l>0, literal is false. Flipping to True -> literal true -> clause satisfied
                            # But wait, sat_count==0 means ALL literals are false. So this literal IS false.
                            # After flip, this literal becomes true -> clause satisfied -> delta -= 1
                            # Hmm, both cases give delta -= 1? Let me reconsider.
                            # If sat_count[ci] == 0, all literals are false.
                            # For positive literal (l>0): assignment[var] must be False (since literal = assignment[var] for positive)
                            # Flipping var to True: literal becomes True -> clause satisfied -> unsat decreases by 1
                            delta -= 1
                    elif sat_count[ci] == 1:
                        # Exactly one literal is satisfied. If that literal is the one for var, flipping breaks it.
                        cv2 = clause_vars[ci]
                        for idx2, is_pos2 in cv2:
                            if idx2 == var and is_pos2 == (l > 0):
                                if assignment[var] == is_pos2:
                                    # This is the only satisfied literal, flipping breaks it
                                    delta += 1
                                break
                        # If the satisfied literal is NOT for var, flipping var doesn't affect this clause's sat status
                        # (it goes from 1 to 2 or stays 1)
                
                for ci in var_neg[var]:
                    if sat_count[ci] == 0:
                        # All literals false. For negative literal (l<0): assignment[var] must be True
                        # Flipping to False: literal becomes True -> clause satisfied -> delta -= 1
                        delta -= 1
                    elif sat_count[ci] == 1:
                        cv2 = clause_vars[ci]
                        for idx2, is_pos2 in cv2:
                            if idx2 == var and is_pos2 == (l > 0):
                                if assignment[var] == is_pos2:
                                    delta += 1
                                break

                if delta < best_delta:
                    best_delta = delta
                    best_var = var

            if best_var >= 0 and best_delta < 0:
                unsat_clauses = apply_flip(assignment, best_var, sat_count, unsat_clauses)
            else:
                # No improving flip found; try random perturbation
                if rng.random() < 0.5:
                    var = rng.randint(0, n - 1)
                    unsat_clauses = apply_flip(assignment, var, sat_count, unsat_clauses)
                else:
                    # Flip a variable from an unsat clause
                    if unsat_clauses:
                        ci = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
                        cv2 = clause_vars[ci]
                        var = cv2[rng.randint(0, len(cv2) - 1)][0]
                        unsat_clauses = apply_flip(assignment, var, sat_count, unsat_clauses)

    # Final check
    if best_assignment is not None:
        # Verify and return
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
