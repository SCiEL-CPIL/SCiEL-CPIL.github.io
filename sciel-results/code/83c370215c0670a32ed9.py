"""3-SAT solver with WalkSAT, clause-weighted scoring, and deadline guard."""
import random
import time


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

    # Build variable occurrence lists
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            idx = abs(l) - 1
            if l > 0:
                var_pos[idx].append(i)
            else:
                var_neg[idx].append(i)

    # Variable frequency for weighting (higher freq -> lower weight)
    var_freq = [0] * n
    for i in range(n):
        var_freq[i] = len(var_pos[i]) + len(var_neg[i])
    max_freq = max(var_freq) if max(var_freq) > 0 else 1

    # Clause weights: decay exponentially with clause frequency
    # More frequent clauses get lower weight
    clause_weights = []
    for c in clauses:
        freq = 0
        for l in c:
            freq += var_freq[abs(l) - 1]
        # Simple inverse frequency weighting
        weight = 1.0 / (1.0 + freq * 0.1)
        clause_weights.append(weight)

    best_assignment = None
    best_unsat_count = clause_count + 1
    best_weighted_unsat = float('inf')

    def count_unsat(assignment):
        count = 0
        for c in clauses:
            sat = False
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] == (l > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def weighted_unsat_score(assignment, sat_count):
        """Compute weighted sum of unsatisfied clauses."""
        score = 0.0
        for i in range(clause_count):
            if sat_count[i] == 0:
                score += clause_weights[i]
        return score

    # Phase 1: WalkSAT with clause-weighted scoring
    rng = random.Random(42)
    max_restarts = 5000
    steps_per_restart = 8000
    noise = 0.25  # probability of random flip

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment with majority-sign bias
        if restart == 0:
            # Try majority sign initialization
            majority = [True] * n
            for i in range(n):
                pos_count = len(var_pos[i])
                neg_count = len(var_neg[i])
                if neg_count > pos_count:
                    majority[i] = False
            assignment = majority[:]
            # With some probability, flip a few
            for _ in range(n // 5):
                assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
        else:
            # Perturb from best or random
            if best_assignment is not None and rng.random() < 0.7:
                assignment = best_assignment[:]
                for _ in range(n // 4 + 1):
                    assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        # Build sat_count and unsat_clauses
        sat_count = [0] * clause_count
        unsat_clauses = []
        for i, c in enumerate(clauses):
            cnt = 0
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] == (l > 0):
                    cnt += 1
            sat_count[i] = cnt
            if cnt == 0:
                unsat_clauses.append(i)

        current_unsat = len(unsat_clauses)
        current_weighted = weighted_unsat_score(assignment, sat_count)
        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        elif current_unsat == best_unsat_count and current_weighted < best_weighted_unsat:
            best_weighted_unsat = current_weighted
            best_assignment = assignment[:]
        else:
            best_weighted_unsat = current_weighted

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

                # Update sat_count and unsat_clauses
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
                continue

            # Choose a random unsatisfied clause
            clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            clause = clauses[clause_idx]

            # Score each variable in the clause using weighted gain
            best_var = -1
            best_score = -1.0
            for l in clause:
                var = abs(l) - 1
                target_val = (l > 0)
                if assignment[var] != target_val:
                    # Flipping would make this literal true
                    gain = 0.0
                    loss = 0.0
                    if l > 0:
                        # Flip False -> True
                        for ci in var_pos[var]:
                            if sat_count[ci] == 0:
                                gain += clause_weights[ci]
                            elif sat_count[ci] == 1:
                                loss += clause_weights[ci]
                        for ci in var_neg[var]:
                            if sat_count[ci] == 0:
                                loss += clause_weights[ci]
                            elif sat_count[ci] == 1:
                                gain += clause_weights[ci]
                    else:
                        # Flip True -> False
                        for ci in var_neg[var]:
                            if sat_count[ci] == 0:
                                gain += clause_weights[ci]
                            elif sat_count[ci] == 1:
                                loss += clause_weights[ci]
                        for ci in var_pos[var]:
                            if sat_count[ci] == 0:
                                loss += clause_weights[ci]
                            elif sat_count[ci] == 1:
                                gain += clause_weights[ci]

                    score = gain - loss
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

            # Update sat_count and unsat_clauses
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
                current_weighted = weighted_unsat_score(assignment, sat_count)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                    best_weighted_unsat = current_weighted
                elif current_unsat == best_unsat_count and current_weighted < best_weighted_unsat:
                    best_weighted_unsat = current_weighted
                    best_assignment = assignment[:]

    # Phase 2: Local search from best assignment
    if best_assignment is not None:
        assignment = best_assignment[:]
        for _ in range(20000):
            if time.time() > deadline:
                break
            unsat_clauses = [i for i in range(clause_count) if not any(
                assignment[abs(l) - 1] == (l > 0) for l in clauses[i]
            )]
            if not unsat_clauses:
                return assignment
            # Greedy: find variable that fixes most clauses
            best_var = -1
            best_gain = 0
            for c_idx in unsat_clauses[:100]:
                for l in clauses[c_idx]:
                    var = abs(l) - 1
                    if assignment[var] == (l > 0):
                        continue
                    gain = 0
                    for c2_idx in unsat_clauses[:100]:
                        for l2 in clauses[c2_idx]:
                            if abs(l2) - 1 == var and assignment[var] != (l2 > 0):
                                gain += 1
                                break
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
                current_unsat = count_unsat(assignment)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment

    # Return best found or random
    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
