"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Deadline: leave a safety margin from the 330s limit
    deadline = time.time() + 325.0

    # Precompute var -> list of (clause_idx, signed literal)
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_to_clauses[var].append((i, lit))

    # Precompute clause literal variable indices for fast satisfaction checks
    clause_vars = [[abs(lit) - 1 for lit in clause] for clause in clauses]

    def is_satisfied(assignment):
        for i in range(m):
            clause = clauses[i]
            for lit, var in zip(clause, clause_vars[i]):
                if assignment[var] == (lit > 0):
                    break
            else:
                return False
        return True

    def walk_sat(assignment, max_steps, rng, p, use_weights):
        """
        Optimized WalkSAT with incremental updates and optional clause weighting.
        """
        clause_sat_count = [0] * m
        unsat = [False] * m
        unsat_count = 0

        for i in range(m):
            count = 0
            clause = clauses[i]
            for lit, var in zip(clause, clause_vars[i]):
                if assignment[var] == (lit > 0):
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                unsat[i] = True
                unsat_count += 1

        if unsat_count == 0:
            return True

        # Clause weights for biasing selection toward persistent conflicts
        w = [1.0] * m if use_weights else None
        W = float(m) if use_weights else None

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return False

            if unsat_count == 0:
                return True

            # Pick an unsatisfied clause (weighted by conflict frequency)
            if use_weights:
                # Weighted selection: sample proportional to w
                total = 0.0
                cum = []
                for i in range(m):
                    if unsat[i]:
                        total += w[i]
                        cum.append((i, total))
                    else:
                        cum.append((i, total))
                if total <= 0.0:
                    # Fallback: uniform among unsat
                    idx = rng.randrange(m)
                    while not unsat[idx]:
                        idx = rng.randrange(m)
                    c_idx = idx
                else:
                    r = rng.random() * total
                    lo, hi = 0, len(cum) - 1
                    # binary search for first cum >= r
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if cum[mid][1] >= r:
                            hi = mid
                        else:
                            lo = mid + 1
                    c_idx = cum[lo][0]
                    # In case of floating issues, ensure unsat
                    if not unsat[c_idx]:
                        idx = rng.randrange(m)
                        while not unsat[idx]:
                            idx = rng.randrange(m)
                        c_idx = idx
            else:
                # Uniform among unsat via rejection
                idx = rng.randrange(m)
                while not unsat[idx]:
                    idx = rng.randrange(m)
                c_idx = idx

            clause = clauses[c_idx]

            # Evaluate flips for literals in this clause
            best_gain = -1000
            best_var = -1
            flip_vars = []

            for lit, var in zip(clause, clause_vars[c_idx]):
                current_val = assignment[var]
                desired_val = (lit > 0)

                # Since c_idx is unsat, all its literals are unsatisfied.
                # Flipping 'var' satisfies 'lit' in c_idx. Gain += 1.
                gain = 1

                for (ci2, lit2) in var_to_clauses[var]:
                    if ci2 == c_idx:
                        continue
                    is_sat2 = assignment[var] == (lit2 > 0)
                    if is_sat2:
                        # Currently satisfied; flipping makes it unsatisfied.
                        if clause_sat_count[ci2] == 1:
                            gain -= 1
                    else:
                        # Currently unsatisfied; flipping satisfies it.
                        if clause_sat_count[ci2] == 0:
                            gain += 1

                flip_vars.append(var)
                if gain > best_gain:
                    best_gain = gain
                    best_var = var

            # Select variable to flip
            if best_var == -1:
                var = flip_vars[rng.randrange(len(flip_vars))]
            elif rng.random() < p:
                var = flip_vars[rng.randrange(len(flip_vars))]
            else:
                var = best_var

            # Apply flip
            assignment[var] = not assignment[var]

            # Update clause_sat_count and unsat tracking
            for (ci2, lit2) in var_to_clauses[var]:
                old_count = clause_sat_count[ci2]
                new_is_sat = assignment[var] == (lit2 > 0)
                old_is_sat = not new_is_sat
                if new_is_sat and not old_is_sat:
                    clause_sat_count[ci2] = old_count + 1
                    if old_count == 0:
                        unsat[ci2] = False
                        unsat_count -= 1
                elif not new_is_sat and old_is_sat:
                    clause_sat_count[ci2] = old_count - 1
                    if old_count == 1:
                        unsat[ci2] = True
                        unsat_count += 1

            # Update clause weights: boost unsat clauses, decay all
            if use_weights:
                for i in range(m):
                    if unsat[i]:
                        w[i] = w[i] * 1.5 + 1.0
                    else:
                        w[i] = w[i] * 0.9
                # Periodic renormalization to avoid overflow
                if step % 200 == 0 and step > 0:
                    mx = max(w)
                    if mx > 1e6:
                        for i in range(m):
                            w[i] /= mx

        return unsat_count == 0

    # Strategy: Multiple restarts with varying parameters
    base_steps = max(2000, n * 100)

    restart_count = 0
    # Vary p and weighting across restarts to escape local minima
    p_values = [0.2, 0.3, 0.5, 0.6, 0.8]
    weight_flags = [True, False, True, True, False]

    while time.time() < deadline - 5.0:
        restart_count += 1
        rng = random.Random(restart_count * 1000003 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Quick check
        if is_satisfied(assignment):
            return assignment

        p = p_values[restart_count % len(p_values)]
        use_weights = weight_flags[restart_count % len(weight_flags)]

        # Run WalkSAT
        if walk_sat(assignment, base_steps, rng, p, use_weights):
            return assignment

    # Fallback: return a random assignment
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
