# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # Deadline guard: 320s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 320.0

    # Preprocess: convert to 0-based, separate positive/negative indices
    # clause[i] = (pos_idx, neg_idx) where pos_idx = var-1 if literal>0 else -1
    #            neg_idx = var-1 if literal<0 else -1
    clause_data = []
    for c in clauses:
        pos = -1
        neg = -1
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos = v
            else:
                neg = v
        clause_data.append((pos, neg))

    # Build variable-to-clause incidence for efficient updates
    # var_clauses[var] = list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for i, (pos, neg) in enumerate(clause_data):
        if pos >= 0:
            var_clauses[pos].append((i, True))
        if neg >= 0:
            var_clauses[neg].append((i, False))

    def make_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]

    def is_satisfied(assignment):
        for pos, neg in clause_data:
            sat = False
            if pos >= 0 and assignment[pos]:
                sat = True
            elif neg >= 0 and not assignment[neg]:
                sat = True
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        cnt = 0
        for pos, neg in clause_data:
            sat = False
            if pos >= 0 and assignment[pos]:
                sat = True
            elif neg >= 0 and not assignment[neg]:
                sat = True
            if not sat:
                cnt += 1
        return cnt

    def get_unsat_clauses(assignment):
        unsat = []
        for i, (pos, neg) in enumerate(clause_data):
            sat = False
            if pos >= 0 and assignment[pos]:
                sat = True
            elif neg >= 0 and not assignment[neg]:
                sat = True
            if not sat:
                unsat.append(i)
        return unsat

    def flip(assignment, var):
        assignment[var] = not assignment[var]

    def walk_sat(assignment, rng, max_steps):
        # Maintain clause satisfaction counts for efficiency
        # sat_count[i] = number of literals in clause i that are satisfied
        sat_count = np.zeros(m, dtype=np.int32)
        for i, (pos, neg) in enumerate(clause_data):
            c = 0
            if pos >= 0 and assignment[pos]:
                c += 1
            if neg >= 0 and not assignment[neg]:
                c += 1
            sat_count[i] = c

        unsat_set = set(np.where(sat_count == 0)[0].tolist())

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment

            if not unsat_set:
                return assignment

            # Choose an unsatisfied clause (random)
            clause_idx = rng.choice(list(unsat_set))
            pos, neg = clause_data[clause_idx]

            # Candidate variables in this clause
            candidates = []
            if pos >= 0:
                candidates.append((pos, True))
            if neg >= 0:
                candidates.append((neg, False))

            if not candidates:
                continue

            # Score each candidate: number of clauses that would become satisfied minus those that would become unsatisfied
            best_var = None
            best_score = -10**9
            best_is_pos = None

            for var, is_pos in candidates:
                # If we flip var, which clauses are affected?
                # For each clause j that contains var:
                #   if var appears positively in j: flipping var from True->False makes this literal unsatisfied, flipping False->True makes it satisfied
                #   if var appears negatively in j: flipping var from True->False makes this literal satisfied, flipping False->True makes it unsatisfied
                score = 0
                current_val = assignment[var]
                for (j, j_is_pos) in var_clauses[var]:
                    if j_is_pos:
                        # var appears as +var in clause j
                        # Currently satisfied if current_val is True
                        # After flip: satisfied if not current_val
                        if current_val:
                            # Was satisfied, will be unsatisfied -> lose 1
                            if sat_count[j] == 1:
                                score -= 1  # clause becomes unsatisfied
                        else:
                            # Was unsatisfied, will be satisfied -> gain 1
                            if sat_count[j] == 0:
                                score += 1  # clause becomes satisfied
                    else:
                        # var appears as -var in clause j
                        # Currently satisfied if current_val is False
                        # After flip: satisfied if not current_val (i.e., current_val becomes True means -var is unsatisfied)
                        if not current_val:
                            # Was satisfied, will be unsatisfied -> lose 1
                            if sat_count[j] == 1:
                                score -= 1
                        else:
                            # Was unsatisfied, will be satisfied -> gain 1
                            if sat_count[j] == 0:
                                score += 1

                if score > best_score:
                    best_score = score
                    best_var = var
                    best_is_pos = is_pos

            if best_var is None:
                # Random flip
                var, is_pos = rng.choice(candidates)
                best_var = var

            # Apply flip
            old_val = assignment[best_var]
            flip(assignment, best_var)

            # Update sat_counts for all clauses containing best_var
            for (j, j_is_pos) in var_clauses[best_var]:
                if j_is_pos:
                    # var appears as +var
                    if old_val:
                        # Was True, now False: literal goes from satisfied to unsatisfied
                        sat_count[j] -= 1
                    else:
                        # Was False, now True: literal goes from unsatisfied to satisfied
                        sat_count[j] += 1
                else:
                    # var appears as -var
                    if not old_val:
                        # Was False, now True: literal goes from satisfied to unsatisfied
                        sat_count[j] -= 1
                    else:
                        # Was True, now False: literal goes from unsatisfied to satisfied
                        sat_count[j] += 1

                # Update unsat_set
                if sat_count[j] == 0:
                    unsat_set.add(j)
                elif sat_count[j] > 0 and j in unsat_set:
                    unsat_set.discard(j)

        return assignment

    # Multi-restart WalkSAT with increasing effort
    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    restarts = 0
    base_steps = 5000
    while time.time() < deadline:
        assignment = make_assignment(rng)

        # Quick check
        if is_satisfied(assignment):
            return assignment

        # Run WalkSAT
        steps = base_steps * (1 + restarts // 5)
        if restarts > 50:
            steps = 50000

        assignment = walk_sat(assignment, rng, steps)

        if is_satisfied(assignment):
            return assignment

        # Track best
        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

        restarts += 1

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return make_assignment(rng)
# EVOLVE-BLOCK-END
