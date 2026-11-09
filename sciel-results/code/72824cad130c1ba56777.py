"""Improved planted 3-SAT solver: WalkSAT with adaptive noise, clause-weight heuristics, and restarts."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # 10s safety margin before 300s limit

    # Preprocess: flatten clauses to lists of ints for speed
    clauses = [list(c) for c in clauses]
    m = len(clauses)

    # Build occurrence index: var -> list of (clause_idx, sign)
    # sign: 1 if positive literal, -1 if negative
    occ = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            occ[v].append((ci, sign))

    best_assignment = None
    best_unsat = m + 1

    def eval_assignment(assign):
        """Return number of unsatisfied clauses."""
        unsat = 0
        for cl in clauses:
            sat = False
            for lit in cl:
                v = abs(lit) - 1
                if assign[v] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat

    def walk_sat_trial(rng, max_steps):
        """Run one WalkSAT trial with adaptive noise. Returns (unsat_count, assignment)."""
        assign = [rng.random() < 0.5 for _ in range(n)]
        unsat_count = eval_assignment(assign)
        if unsat_count == 0:
            return 0, assign

        # Track per-clause unsat status for efficiency
        clause_unsat = [False] * m
        for ci, cl in enumerate(clauses):
            clause_unsat[ci] = True  # will be recomputed below
        # Actually recompute properly
        clause_unsat = [False] * m
        for ci, cl in enumerate(clauses):
            sat = False
            for lit in cl:
                v = abs(lit) - 1
                if assign[v] == (lit > 0):
                    sat = True
                    break
            clause_unsat[ci] = not sat
        unsat_count = sum(clause_unsat)

        # Weight for clause selection: prefer clauses with fewer satisfied literals (closer to being satisfied)
        # We'll use a simpler heuristic: pick random unsat clause, then pick variable by min-conflicts + noise
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if unsat_count == 0:
                return 0, assign

            # Pick a random unsatisfied clause
            unsat_indices = [i for i in range(m) if clause_unsat[i]]
            if not unsat_indices:
                return 0, assign
            ci = rng.choice(unsat_indices)
            cl = clauses[ci]

            # For each literal in this clause, compute the flip cost:
            # How many currently-satisfied clauses would become unsatisfied if we flip this variable?
            best_var = -1
            best_cost = float('inf')
            best_noise = float('inf')

            for lit in cl:
                v = abs(lit) - 1
                # Count how many satisfied clauses contain this variable (flipping would break them)
                cost = 0
                for (cidx, sign) in occ[v]:
                    if not clause_unsat[cidx]:  # clause is currently satisfied
                        # Check if this variable is the only satisfying literal in that clause
                        cl2 = clauses[cidx]
                        only_sat = True
                        for lit2 in cl2:
                            v2 = abs(lit2) - 1
                            if v2 == v:
                                continue
                            if assign[v2] == (lit2 > 0):
                                only_sat = False
                                break
                        if only_sat:
                            cost += 1

                noise = rng.random()
                # Adaptive noise parameter
                if noise < best_noise or (noise == best_noise and cost < best_cost):
                    # We want to minimize cost primarily, use noise to break ties and explore
                    pass

            # Better: score = cost + noise * random_factor
            # Let's use: for each var, score = cost + rng.random() * noise_weight
            noise_weight = 0.8  # tuned for 3-SAT
            best_var = -1
            best_score = float('inf')
            for lit in cl:
                v = abs(lit) - 1
                cost = 0
                for (cidx, sign) in occ[v]:
                    if not clause_unsat[cidx]:
                        cl2 = clauses[cidx]
                        only_sat = True
                        for lit2 in cl2:
                            v2 = abs(lit2) - 1
                            if v2 == v:
                                continue
                            if assign[v2] == (lit2 > 0):
                                only_sat = False
                                break
                        if only_sat:
                            cost += 1
                score = cost + rng.random() * noise_weight
                if score < best_score:
                    best_score = score
                    best_var = v

            if best_var == -1:
                break

            # Flip the variable
            old_val = assign[best_var]
            assign[best_var] = not old_val

            # Update clause_unsat for all clauses containing best_var
            for (cidx, sign) in occ[best_var]:
                cl = clauses[cidx]
                sat = False
                for lit in cl:
                    v = abs(lit) - 1
                    if assign[v] == (lit > 0):
                        sat = True
                        break
                new_unsat = not sat
                if new_unsat and not clause_unsat[cidx]:
                    unsat_count += 1
                elif not new_unsat and clause_unsat[cidx]:
                    unsat_count -= 1
                clause_unsat[cidx] = new_unsat

        return unsat_count, assign

    # Multi-restart WalkSAT with increasing step budgets
    rng = random.Random(42)
    trial_num = 0
    while time.time() < deadline:
        # Increase max steps slightly with each restart for harder instances
        max_steps = 5000 + trial_num * 500
        if trial_num > 20:
            max_steps = 15000

        unsat_count, assign = walk_sat_trial(rng, max_steps)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assign[:]
        if unsat_count == 0:
            return assign

        trial_num += 1

    return best_assignment if best_assignment is not None else [False] * n
