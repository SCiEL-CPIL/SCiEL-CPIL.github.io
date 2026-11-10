"""3-SAT solver: clause-weighted stochastic local search with majority-sign seeding."""
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["instance_clauses"] if "instance_clauses" in instance else instance["clauses"]
    start_time = time.time()
    deadline = start_time + 288  # safety margin before 330s kill

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

    # Build clause structure
    clause_vars = []
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
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

    # Compute clause frequencies and weights (exponential decay with frequency)
    var_freq = [0] * n
    for cv in clause_vars:
        for idx, _ in cv:
            var_freq[idx] += 1

    # Clause weights: weight_i = 1 / (1 + freq_penalty) where freq_penalty is based on
    # how "common" the variables in the clause are. Rare-variable clauses get higher weight.
    # Actually, the strategy says weights decay exponentially with clause frequency.
    # Let's interpret: clauses that appear in many similar contexts (high variable overlap)
    # get lower weight. Simpler: weight based on inverse of max variable frequency in clause.
    clause_weights = []
    for cv in clause_vars:
        max_freq = max(var_freq[idx] for idx, _ in cv)
        # Weight decays exponentially with max variable frequency
        w = math.exp(-0.1 * max_freq)
        clause_weights.append(w)

    # Majority-sign seeding: for each variable, count positive vs negative occurrences
    pos_count = [0] * n
    neg_count = [0] * n
    for i, c in enumerate(clauses):
        for l in c:
            idx = abs(l) - 1
            if l > 0:
                pos_count[idx] += 1
            else:
                neg_count[idx] += 1

    def majority_seed(rng):
        """Initialize with bias toward majority sign, with some randomness."""
        assignment = []
        for v in range(n):
            if pos_count[v] > neg_count[v]:
                # Bias toward True
                assignment.append(rng.random() < 0.8)
            elif neg_count[v] > pos_count[v]:
                # Bias toward False
                assignment.append(rng.random() < 0.2)
            else:
                assignment.append(rng.random() < 0.5)
        return assignment

    def compute_sat_counts(assignment):
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

    def weighted_unsat(assignment, sat_count, unsat_clauses):
        """Weighted sum of unsatisfied clauses."""
        total = 0.0
        for ci in unsat_clauses:
            total += clause_weights[ci]
        return total

    def flip_var(assignment, var, sat_count, unsat_clauses):
        """Flip variable and update sat_count and unsat_clauses. Returns new unsat_clauses."""
        old_val = assignment[var]
        assignment[var] = not old_val
        new_unsat = []

        for ci in var_pos[var]:
            if old_val:
                sat_count[ci] -= 1
            else:
                sat_count[ci] += 1
            if sat_count[ci] == 0:
                new_unsat.append(ci)

        for ci in var_neg[var]:
            if old_val:
                sat_count[ci] += 1
            else:
                sat_count[ci] -= 1
            if sat_count[ci] == 0:
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
        return unsat_clauses

    rng = random.Random(42)

    best_assignment = None
    best_unsat_count = clause_count + 1
    best_weighted = float('inf')

    max_restarts = 3000
    steps_per_restart = 6000
    noise = 0.2

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize
        if restart == 0:
            assignment = majority_seed(rng)
        elif best_assignment is not None and rng.random() < 0.6:
            assignment = best_assignment[:]
            # Perturb
            num_flips = max(1, n // 5)
            for _ in range(num_flips):
                assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
        else:
            assignment = majority_seed(rng)

        sat_count, unsat_clauses = compute_sat_counts(assignment)
        current_unsat = len(unsat_clauses)

        if current_unsat == 0:
            return assignment

        current_weighted = weighted_unsat(assignment, sat_count, unsat_clauses)
        if current_unsat < best_unsat_count or (current_unsat == best_unsat_count and current_weighted < best_weighted):
            best_unsat_count = current_unsat
            best_weighted = current_weighted
            best_assignment = assignment[:]

        # WalkSAT loop with weighted scoring
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat_clauses:
                return assignment

            # With some probability, do a random flip
            if rng.random() < noise:
                var = rng.randint(0, n - 1)
                unsat_clauses = flip_var(assignment, var, sat_count, unsat_clauses)
                continue

            # Choose a weighted random unsatisfied clause
            # Weighted selection: clauses with higher weight are more likely to be chosen
            total_w = sum(clause_weights[ci] for ci in unsat_clauses)
            if total_w <= 0:
                clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            else:
                r = rng.random() * total_w
                cum = 0.0
                clause_idx = unsat_clauses[-1]
                for ci in unsat_clauses:
                    cum += clause_weights[ci]
                    if r <= cum:
                        clause_idx = ci
                        break

            clause = clauses[clause_idx]

            # Score each variable in the clause
            # Score = weighted sum of unsat clauses that would be satisfied by flipping
            best_var = -1
            best_score = -1.0

            for l in clause:
                var = abs(l) - 1
                target_val = (l > 0)
                if assignment[var] != target_val:
                    # Flipping would satisfy this literal
                    # Count weighted unsat clauses where this is the only unsat literal
                    affected = var_pos[var] if l > 0 else var_neg[var]
                    score = 0.0
                    for ci in affected:
                        if sat_count[ci] == 0:
                            # Check if this is the only unsat literal
                            cv2 = clause_vars[ci]
                            only_unsat = True
                            for idx2, is_pos2 in cv2:
                                if idx2 == var and is_pos2 == target_val:
                                    continue
                                if assignment[idx2] == is_pos2:
                                    only_unsat = False
                                    break
                            if only_unsat:
                                score += clause_weights[ci]
                else:
                    # Variable already has target value; flipping would break this clause
                    # But might satisfy others - count net effect
                    score = 0.0
                    affected = var_pos[var] if l > 0 else var_neg[var]
                    for ci in affected:
                        if sat_count[ci] == 0:
                            # Flipping would make this literal false, so if it was the only sat... 
                            # Wait, if assignment[var] == target_val, the literal is already true.
                            # Flipping makes it false. If sat_count[ci] == 1, it becomes unsat.
                            if sat_count[ci] == 1:
                                # This clause would become unsat - penalty
                                score -= clause_weights[ci]
                    # Also check other clauses where this var appears with opposite sign
                    other_affected = var_neg[var] if l > 0 else var_pos[var]
                    for ci in other_affected:
                        if sat_count[ci] == 0:
                            # Flipping would make this literal true
                            cv2 = clause_vars[ci]
                            only_unsat = True
                            for idx2, is_pos2 in cv2:
                                if idx2 == var:
                                    continue
                                if assignment[idx2] == is_pos2:
                                    only_unsat = False
                                    break
                            if only_unsat:
                                score += clause_weights[ci]

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var >= 0:
                var_to_flip = best_var
            else:
                l = clause[rng.randint(0, len(clause) - 1)]
                var_to_flip = abs(l) - 1

            unsat_clauses = flip_var(assignment, var_to_flip, sat_count, unsat_clauses)

            # Periodically update best
            if step % 1000 == 0:
                current_unsat = len(unsat_clauses)
                if current_unsat < best_unsat_count or (current_unsat == best_unsat_count):
                    current_weighted = weighted_unsat(assignment, sat_count, unsat_clauses)
                    if current_unsat < best_unsat_count or (current_unsat == best_unsat_count and current_weighted < best_weighted):
                        best_unsat_count = current_unsat
                        best_weighted = current_weighted
                        best_assignment = assignment[:]

    # Final polish: try to reduce unsat count further with more careful search
    if best_assignment is not None and best_unsat_count > 0:
        assignment = best_assignment[:]
        sat_count, unsat_clauses = compute_sat_counts(assignment)

        # Try flipping single variables to reduce unsat count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for var in range(n):
                if time.time() > deadline:
                    break
                old_val = assignment[var]
                # Count how many clauses would change
                delta = 0
                for ci in var_pos[var]:
                    if old_val:
                        if sat_count[ci] == 1:
                            delta += 1  # becomes unsat
                    else:
                        if sat_count[ci] == 0:
                            delta -= 1  # becomes sat
                for ci in var_neg[var]:
                    if old_val:
                        if sat_count[ci] == 0:
                            delta -= 1
                    else:
                        if sat_count[ci] == 1:
                            delta += 1

                if delta < 0:
                    # Flipping reduces unsat count
                    unsat_clauses = flip_var(assignment, var, sat_count, unsat_clauses)
                    improved = True

        if len(unsat_clauses) == 0:
            return assignment
        if len(unsat_clauses) < best_unsat_count:
            best_unsat_count = len(unsat_clauses)
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
