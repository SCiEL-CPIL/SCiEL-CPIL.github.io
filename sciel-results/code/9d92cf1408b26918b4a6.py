"""CDCL-inspired 3-SAT solver with clause learning and efficient incremental tracking."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)

    if num_clauses == 0:
        return [False] * n

    # Preprocess: convert to 0-based (var, sign) tuples
    processed_clauses = []
    for clause in clauses:
        processed = [(abs(lit) - 1, 1 if lit > 0 else -1) for lit in clause]
        processed_clauses.append(processed)

    # Build occurrence lists: var_occ[var] = list of (clause_idx, sign)
    var_occ = [[] for _ in range(n)]
    for ci, clause in enumerate(processed_clauses):
        for var, sign in clause:
            var_occ[var].append((ci, sign))

    # For fast satisfaction check, precompute per-clause var and sign arrays
    clause_vars = [[v for v, s in c] for c in processed_clauses]
    clause_signs = [[s for v, s in c] for c in processed_clauses]

    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin

    rng = random.Random(12345)

    # --- Core state ---
    # assignment: list of bools
    # clause_sat_count[ci] = number of true literals in clause ci
    # unsat_count = number of clauses with sat_count == 0

    def init_state(assignment):
        """Initialize clause_sat_count and unsat_count from assignment."""
        clause_sat_count = [0] * len(processed_clauses)
        unsat = 0
        for ci in range(len(processed_clauses)):
            vs = clause_vars[ci]
            ss = clause_signs[ci]
            cnt = 0
            for i in range(len(vs)):
                v = vs[i]
                s = ss[i]
                if s == 1:
                    if assignment[v]:
                        cnt += 1
                else:
                    if not assignment[v]:
                        cnt += 1
            clause_sat_count[ci] = cnt
            if cnt == 0:
                unsat += 1
        return clause_sat_count, unsat

    def flip_var(assignment, var, clause_sat_count, unsat_count):
        """Flip variable var, update clause_sat_count and unsat_count incrementally.
        Returns new unsat_count."""
        old_val = assignment[var]
        assignment[var] = not old_val
        new_unsat = unsat_count
        for ci, sign in var_occ[var]:
            old_sat = clause_sat_count[ci]
            # Determine if this literal was true before flip
            if sign == 1:
                was_true = old_val
            else:
                was_true = not old_val
            if was_true:
                # Literal was true, now false
                clause_sat_count[ci] = old_sat - 1
                if old_sat == 1:
                    new_unsat += 1
            else:
                # Literal was false, now true
                clause_sat_count[ci] = old_sat + 1
                if old_sat == 0:
                    new_unsat -= 1
        return new_unsat

    # --- Learned clauses (nogoods) ---
    # Each learned clause is a list of (var, sign) that must be satisfied
    # We track them separately and check them in the unsat count
    learned_clauses = []  # list of list of (var, sign)

    def check_learned(assignment):
        """Count how many learned clauses are unsatisfied."""
        cnt = 0
        for lc in learned_clauses:
            sat = False
            for var, sign in lc:
                if sign == 1:
                    if assignment[var]:
                        sat = True
                        break
                else:
                    if not assignment[var]:
                        sat = True
                        break
            if not sat:
                cnt += 1
        return cnt

    def extract_conflict(assignment, clause_sat_count):
        """Extract a conflict clause from the current assignment.
        Find an unsatisfied clause and use its literals as a nogood.
        Returns a list of (var, sign) representing the clause that should be
        satisfied (i.e., at least one literal must be true)."""
        # Find an unsatisfied clause
        for ci in range(len(processed_clauses)):
            if clause_sat_count[ci] == 0:
                # All literals in this clause are false
                # The conflict is: NOT(all these literals are false)
                # = at least one literal is true
                # Represent as the clause itself (must have >= 1 true)
                conflict = [(v, s) for v, s in processed_clauses[ci]]
                return conflict
        return None

    # --- WalkSAT with learned clauses ---
    max_restarts = 200
    base_steps = 8000

    best_assignment = None
    best_unsat = float('inf')

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count, unsat_count = init_state(assignment)
        learned_unsat = check_learned(assignment)
        total_unsat = unsat_count + learned_unsat

        if total_unsat == 0:
            return assignment

        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_assignment = assignment[:]

        # Adaptive steps: increase with restart number up to a cap
        steps = min(base_steps + restart * 500, 25000)

        # Adaptive noise: start high, decrease
        noise_prob = max(0.1, 0.4 - restart * 0.005)

        for step in range(steps):
            if time.time() > deadline:
                break

            # Find an unsatisfied clause (original or learned)
            # Prefer original clauses for conflict extraction
            unsat_ci = None
            for ci in range(len(processed_clauses)):
                if clause_sat_count[ci] == 0:
                    unsat_ci = ci
                    break

            if unsat_ci is None:
                # All original clauses satisfied, check learned
                if learned_unsat == 0:
                    return assignment
                # Find an unsatisfied learned clause
                for li, lc in enumerate(learned_clauses):
                    sat = False
                    for var, sign in lc:
                        if sign == 1:
                            if assignment[var]:
                                sat = True
                                break
                        else:
                            if not assignment[var]:
                                sat = True
                                break
                    if not sat:
                        # Flip a random variable in this learned clause
                        var, sign = rng.choice(lc)
                        unsat_count = flip_var(assignment, var, clause_sat_count, unsat_count)
                        learned_unsat = check_learned(assignment)
                        total_unsat = unsat_count + learned_unsat
                        if total_unsat == 0:
                            return assignment
                        if total_unsat < best_unsat:
                            best_unsat = total_unsat
                            best_assignment = assignment[:]
                        break
                continue

            # Got an unsatisfied original clause
            clause = processed_clauses[unsat_ci]

            # Choose variable to flip
            if rng.random() < (1.0 - noise_prob):
                # Gain-based: flip variable that maximizes gain
                best_var = None
                best_gain = -999
                for var, sign in clause:
                    # Compute gain: how many clauses' satisfaction changes
                    gain = 0
                    for ci2, sign2 in var_occ[var]:
                        old_sat = clause_sat_count[ci2]
                        if sign2 == 1:
                            was_true = assignment[var]
                        else:
                            was_true = not assignment[var]
                        if was_true:
                            # Flipping makes this literal false
                            if old_sat == 1:
                                gain -= 1
                        else:
                            # Flipping makes this literal true
                            if old_sat == 0:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                if best_var is not None:
                    unsat_count = flip_var(assignment, best_var, clause_sat_count, unsat_count)
            else:
                # Random flip
                var, sign = rng.choice(clause)
                unsat_count = flip_var(assignment, var, clause_sat_count, unsat_count)

            learned_unsat = check_learned(assignment)
            total_unsat = unsat_count + learned_unsat

            if total_unsat == 0:
                return assignment

            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment[:]

        # End of restart: extract conflict clause if we didn't solve
        if total_unsat > 0 and restart > 0:
            conflict = extract_conflict(assignment, clause_sat_count)
            if conflict is not None:
                # Check if this conflict is already learned (avoid duplicates)
                conflict_set = frozenset(conflict)
                is_dup = False
                for lc in learned_clauses:
                    if frozenset(lc) == conflict_set:
                        is_dup = True
                        break
                if not is_dup:
                    learned_clauses.append(conflict)
                    # Cap learned clauses to prevent bloat
                    if len(learned_clauses) > 500:
                        learned_clauses = learned_clauses[-300:]

    # Final check
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
