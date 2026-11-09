"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    """Satisfying assignment for a locally generated planted satisfiable 3-CNF.

    instance: dict with 'n' (int) and 'clauses' (list of lists of signed 1-based ints).
    Returns: list of booleans (length n) such that every clause is satisfied.
    """
    n = instance["n"]
    clauses = instance["clauses"]

    # Deadline guard: evaluator kills at 330s; keep a safety margin.
    start = time.time()
    deadline = start + 295.0  # 35s margin

    rng = random.Random(0)

    # Precompute literal arrays for vectorized evaluation.
    # lit_pos[i] = variable index (0-based) of literal i
    # lit_sign[i] = +1 if positive, -1 if negative
    m = len(clauses)
    lit_pos = np.empty((m, 3), dtype=np.int64)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for i, cl in enumerate(clauses):
        for j, l in enumerate(cl):
            lit_pos[i, j] = abs(l) - 1
            lit_sign[i, j] = 1 if l > 0 else -1

    # Precompute variable -> list of clause indices map for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = lit_pos[i, j]
            var_to_clauses[v].append(i)

    def build_value(assignment):
        # value[v] = +1 if var v is True, -1 if False
        return np.where(np.asarray(assignment, dtype=bool), 1, -1).astype(np.int8)

    def clause_sat_mask(value):
        # Returns boolean array of length m: True if clause satisfied.
        # A clause is satisfied if any literal matches: sign * value[var] == 1
        vals = value[lit_pos]  # (m,3)
        return np.any(lit_sign * vals == 1, axis=1)

    def unsat_count(value):
        return int(np.count_nonzero(~clause_sat_mask(value)))

    # Fast initial random assignment; check if already satisfied.
    assignment = [rng.random() < 0.5 for _ in range(n)]
    value = build_value(assignment)
    if unsat_count(value) == 0:
        return assignment

    # WalkSAT with a clause-counting heuristic + noise, and periodic restarts.
    # Track best assignment found (fewest unsatisfied clauses) as a fallback.
    best_assignment = None
    best_unsat = m + 1

    max_total_steps = 2000000
    noise = 0.35  # probability of a random flip instead of the best

    while time.time() < deadline:
        # Random restart assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        value = build_value(assignment)
        unsat_mask = ~clause_sat_mask(value)
        cur_unsat = int(np.count_nonzero(unsat_mask))

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # If we've used a lot of total steps and found a good best, bias restarts
        # toward best_assignment occasionally (light annealing).
        if best_unsat <= 2 and rng.random() < 0.3:
            assignment = best_assignment[:]
            value = build_value(assignment)
            unsat_mask = ~clause_sat_mask(value)
            cur_unsat = best_unsat

        for _ in range(max_total_steps):
            if time.time() >= deadline:
                break

            if cur_unsat == 0:
                return assignment

            # Pick a random unsatisfied clause.
            unsat_indices = np.nonzero(unsat_mask)[0]
            ci = rng.choice(unsat_indices.tolist())

            # For each of the 3 literals in this clause, compute the net gain
            # if we flip that variable.
            # Gain = (# unsat clauses containing v that become sat) - 
            #        (# sat clauses containing v that become unsat)
            # 
            # A clause k containing v with sign s_k and current value val_v:
            #   Current literal value = s_k * val_v
            #   After flip: -s_k * val_v
            # 
            # For an unsat clause to become sat, the flipped literal must be the
            # ONLY one that can satisfy it (i.e., other literals are already false
            # and won't change). Actually, more precisely: after flip, if the
            # flipped literal becomes true (s_k * new_val_v == 1), the clause is sat.
            # But if other literals are already true, it was already sat.
            #
            # Let's use a simpler but effective heuristic: count for each variable
            # v in the clause, how many unsatisfied clauses contain v where v's
            # literal is currently false (so flipping might fix them), minus how
            # many satisfied clauses contain v where v's literal is currently true
            # (so flipping would break them if no other literal saves them).
            #
            # For efficiency, we'll compute:
            #   fix_count[v] = number of unsat clauses containing v where the
            #                  literal of v is currently false
            #   break_count[v] = number of sat clauses containing v where the
            #                    literal of v is currently true AND it's the
            #                    only true literal (so flipping makes clause unsat)
            #
            # Computing break_count exactly is expensive. Let's use an approximation:
            # break_count[v] ≈ number of sat clauses containing v where v's literal
            # is true, weighted by 1/(number of true literals in that clause)
            # to estimate probability it's the only true one.
            #
            # Actually, let's just use the standard WalkSAT "score" which is:
            # For each variable v in the unsat clause, count how many OTHER unsat
            # clauses contain v (these might be fixed by flipping v).
            # This is cheaper: we just need to know if v appears in other unsat clauses.
            
            gains = np.zeros(3, dtype=np.int64)
            for j in range(3):
                v = lit_pos[ci, j]
                s = lit_sign[ci, j]
                # Current literal value
                cur_lit_val = s * value[v]
                
                # If this literal is currently true, flipping it will definitely
                # make this clause still unsat (since we picked an unsat clause,
                # but wait - if this literal is true, the clause would be sat.
                # So in an unsat clause, all literals are false. Thus cur_lit_val == -1.
                # So flipping will make this literal true, satisfying this clause.
                
                # Count how many other unsat clauses contain v
                # We can approximate by counting unsat clauses in var_to_clauses[v]
                # But that's O(degree). Let's just do it.
                count = 0
                for k in var_to_clauses[v]:
                    if k != ci and unsat_mask[k]:
                        count += 1
                gains[j] = count

            # Choose literal with max gain + noise.
            # Fallback: random literal if gains all zero.
            if rng.random() < noise:
                j = rng.randrange(3)
            else:
                # pick argmax of gains; ties broken randomly
                maxg = gains.max()
                candidates = np.nonzero(gains == maxg)[0]
                j = rng.choice(candidates.tolist())

            v = lit_pos[ci, j]
            old = value[v]
            new = -old
            value[v] = new
            assignment[v] = new > 0

            # Update unsat_mask incrementally for clauses containing v.
            for k in var_to_clauses[v]:
                # recompute satisfaction of clause k
                sat = np.any(lit_sign[k] * value[lit_pos[k]] == 1)
                if sat:
                    if unsat_mask[k]:
                        cur_unsat -= 1
                        unsat_mask[k] = False
                else:
                    if not unsat_mask[k]:
                        cur_unsat += 1
                        unsat_mask[k] = True

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    # Return best found.
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
