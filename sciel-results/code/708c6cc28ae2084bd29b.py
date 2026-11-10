"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random


def solve(instance):
    """Simulated annealing with incremental clause tracking and smart variable selection."""
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # safety margin before 330s kill

    # Precompute literal indices (0-based) and sign for each clause
    clause_lits = []
    for c in clauses:
        lits = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_lits.append(lits)

    # For each variable, track which clauses contain it
    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for var, sign in lits:
            if sign == 1:
                pos_clauses[var].append(ci)
            else:
                neg_clauses[var].append(ci)

    # Initialize assignment with slight bias toward satisfying more clauses
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Compute initial clause satisfaction status
    clause_sat = [0] * m
    for ci, lits in enumerate(clause_lits):
        sat_count = 0
        for var, sign in lits:
            if assignment[var] == (sign == 1):
                sat_count += 1
        clause_sat[ci] = sat_count

    unsat_count = sum(1 for s in clause_sat if s == 0)

    # Track best solution found
    best_unsat = unsat_count
    best_assignment = assignment[:]

    # Build list of unsatisfied clause indices for smart variable selection
    unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]

    # Simulated annealing with restarts
    T_start = 3.0
    T_end = 0.005
    total_steps = 400000
    max_restarts = 5
    restart_threshold = 50000  # steps without improvement before restart

    step = 0
    steps_since_improvement = 0
    restart_count = 0

    while time.time() < deadline:
        # Current temperature (exponential cooling)
        progress = min(step / total_steps, 1.0)
        T = T_start * math.exp(-8.0 * progress)
        if T < T_end:
            T = T_end

        # If all clauses satisfied, return immediately
        if unsat_count == 0:
            return assignment

        # Smart variable selection: prefer variables in unsatisfied clauses
        # with probability based on temperature
        if unsat_clauses and rng.random() < 0.7:
            # Pick a random unsatisfied clause and pick a variable from it
            ci = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            v = clause_lits[ci][rng.randint(0, 2)][0]
        else:
            v = rng.randint(0, n - 1)

        # Compute delta: change in unsat_count if we flip v
        delta = 0
        if assignment[v]:
            for ci in pos_clauses[v]:
                if clause_sat[ci] == 1:
                    delta += 1
        else:
            for ci in pos_clauses[v]:
                if clause_sat[ci] == 0:
                    delta -= 1

        if not assignment[v]:
            for ci in neg_clauses[v]:
                if clause_sat[ci] == 1:
                    delta += 1
        else:
            for ci in neg_clauses[v]:
                if clause_sat[ci] == 0:
                    delta -= 1

        # Accept or reject
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            # Apply flip
            assignment[v] = not assignment[v]

            # Update clause_sat for all clauses containing v
            if assignment[v]:
                for ci in pos_clauses[v]:
                    clause_sat[ci] += 1
                for ci in neg_clauses[v]:
                    clause_sat[ci] -= 1
            else:
                for ci in neg_clauses[v]:
                    clause_sat[ci] += 1
                for ci in pos_clauses[v]:
                    clause_sat[ci] -= 1

            # Update unsat_count and unsat_clauses list
            unsat_count += delta

            # Update unsat_clauses efficiently
            # Rebuild periodically or track changes
            if delta != 0:
                # For simplicity, rebuild unsat_clauses every 1000 steps
                if step % 1000 == 0:
                    unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]

            # Track best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

        step += 1

        # Restart if stuck for too long
        if steps_since_improvement > restart_threshold and restart_count < max_restarts:
            restart_count += 1
            # Reinitialize with some perturbation from best
            assignment = best_assignment[:]
            clause_sat = [0] * m
            for ci, lits in enumerate(clause_lits):
                sat_count = 0
                for var, sign in lits:
                    if assignment[var] == (sign == 1):
                        sat_count += 1
                clause_sat[ci] = sat_count
            unsat_count = sum(1 for s in clause_sat if s == 0)
            unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]

            # Perturb a few random variables
            for _ in range(n // 10 + 1):
                pv = rng.randint(0, n - 1)
                assignment[pv] = not assignment[pv]
                if assignment[pv]:
                    for ci in pos_clauses[pv]:
                        clause_sat[ci] += 1
                    for ci in neg_clauses[pv]:
                        clause_sat[ci] -= 1
                else:
                    for ci in neg_clauses[pv]:
                        clause_sat[ci] += 1
                    for ci in pos_clauses[pv]:
                        clause_sat[ci] -= 1
                unsat_count = sum(1 for s in clause_sat if s == 0)
                unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]

            steps_since_improvement = 0
            step = 0  # Reset step counter for new annealing schedule

    # Final polish phase: try to fix remaining unsatisfied clauses
    if best_unsat > 0 and time.time() < deadline - 10:
        # Focus on unsatisfied clauses and try flipping variables that appear in them
        unsat_clauses = [ci for ci in range(m) if clause_sat[ci] == 0]
        polish_deadline = time.time() + min(20, deadline - time.time())
        polish_assignment = best_assignment[:]
        polish_clause_sat = [0] * m
        for ci, lits in enumerate(clause_lits):
            sat_count = 0
            for var, sign in lits:
                if polish_assignment[var] == (sign == 1):
                    sat_count += 1
            polish_clause_sat[ci] = sat_count
        polish_unsat = sum(1 for s in polish_clause_sat if s == 0)

        while time.time() < polish_deadline and polish_unsat > 0:
            # Pick a random unsatisfied clause
            if not unsat_clauses:
                break
            ci = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            # Try flipping each variable in this clause
            best_v = -1
            best_delta = 0
            for var, sign in clause_lits[ci]:
                # Compute delta for flipping var
                d = 0
                if polish_assignment[var]:
                    for cj in pos_clauses[var]:
                        if polish_clause_sat[cj] == 1:
                            d += 1
                else:
                    for cj in pos_clauses[var]:
                        if polish_clause_sat[cj] == 0:
                            d -= 1
                if not polish_assignment[var]:
                    for cj in neg_clauses[var]:
                        if polish_clause_sat[cj] == 1:
                            d += 1
                else:
                    for cj in neg_clauses[var]:
                        if polish_clause_sat[cj] == 0:
                            d -= 1
                if d < best_delta:
                    best_delta = d
                    best_v = var

            if best_v >= 0 and best_delta < 0:
                # Apply the best flip
                v = best_v
                polish_assignment[v] = not polish_assignment[v]
                if polish_assignment[v]:
                    for cj in pos_clauses[v]:
                        polish_clause_sat[cj] += 1
                    for cj in neg_clauses[v]:
                        polish_clause_sat[cj] -= 1
                else:
                    for cj in neg_clauses[v]:
                        polish_clause_sat[cj] += 1
                    for cj in pos_clauses[v]:
                        polish_clause_sat[cj] -= 1
                polish_unsat += best_delta
                if polish_unsat < best_unsat:
                    best_unsat = polish_unsat
                    best_assignment = polish_assignment[:]
                # Rebuild unsat_clauses
                unsat_clauses = [cj for cj in range(m) if polish_clause_sat[cj] == 0]

    return best_assignment
# EVOLVE-BLOCK-END
