"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    DEADLINE = 310.0  # safety margin before 330s kill

    # Preprocess: convert to 0-based variable indices and signs
    # clause_vars[i] = list of 3 var indices (0-based)
    # clause_sigs[i] = list of 3 signs (+1 or -1)
    clause_vars = []
    clause_sigs = []
    for c in clauses:
        cv = [abs(l) - 1 for l in c]
        cs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(cv)
        clause_sigs.append(cs)

    # Build var -> list of (clause_idx, pos_in_clause) for efficient updates
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i][j]].append((i, j))

    rng = random.Random(42)

    def make_random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def is_satisfied(assignment, clause_idx):
        cv = clause_vars[clause_idx]
        cs = clause_sigs[clause_idx]
        for j in range(3):
            if assignment[cv[j]] == (cs[j] > 0):
                return True
        return False

    def count_unsat(assignment):
        count = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            sat = False
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def get_unsat_clauses(assignment):
        unsat = []
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            sat = False
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat

    def walk_sat_run(assignment, max_flips=8000, p=0.55):
        """WalkSAT with clause-level bookkeeping for speed."""
        # Initialize clause satisfaction status
        clause_sat_count = [0] * m  # number of satisfied literals in each clause
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            cnt = 0
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    cnt += 1
            clause_sat_count[i] = cnt

        unsat_set = set(i for i in range(m) if clause_sat_count[i] == 0)

        for flip in range(max_flips):
            if time.time() - start_time > DEADLINE:
                return None

            if len(unsat_set) == 0:
                return assignment[:]

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))

            # Choose literal to flip
            if rng.random() < p:
                # Random literal in the clause
                li = rng.randint(0, 2)
            else:
                # Weighted: prefer literals that satisfy more other clauses
                # For each literal in the unsat clause, count how many other
                # unsatisfied clauses it appears in (flipping would help)
                best_li = 0
                best_weight = -1
                for li in range(3):
                    var = clause_vars[ci][li]
                    sig = clause_sigs[ci][li]
                    # Flipping this var: it will satisfy clause ci (since it's unsat)
                    # Count how many other unsat clauses contain this var
                    weight = 0
                    for (oi, oj) in var_to_clauses[var]:
                        if oi == ci:
                            continue
                        if oi in unsat_set:
                            weight += 1
                    if weight > best_weight:
                        best_weight = weight
                        best_li = li
                li = best_li

            var = clause_vars[ci][li]
            old_val = assignment[var]
            new_val = not old_val
            assignment[var] = new_val

            # Update clause_sat_count for all clauses containing this var
            for (oi, oj) in var_to_clauses[var]:
                osig = clause_sigs[oi][oj]
                # Check if this literal was satisfied before and after
                was_sat = (old_val == (osig > 0))
                now_sat = (new_val == (osig > 0))
                if was_sat and not now_sat:
                    clause_sat_count[oi] -= 1
                    if clause_sat_count[oi] == 0:
                        unsat_set.add(oi)
                elif not was_sat and now_sat:
                    clause_sat_count[oi] += 1
                    if clause_sat_count[oi] > 0 and oi in unsat_set:
                        unsat_set.remove(oi)

        # Return None if not solved (caller handles restart)
        if len(unsat_set) == 0:
            return assignment[:]
        return None

    best_assignment = None
    best_unsat = m + 1

    max_restarts = 300
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        assignment = make_random_assignment()
        result = walk_sat_run(assignment, max_flips=8000, p=0.55)
        if result is not None:
            return result

        # Track best
        uc = count_unsat(assignment)
        if uc < best_unsat:
            best_unsat = uc
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment

    return [True] * n
# EVOLVE-BLOCK-END
