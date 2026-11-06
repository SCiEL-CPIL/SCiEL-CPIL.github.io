"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280  # 300s limit, 20s safety margin

    # Precompute: for each variable, list of (clause_idx, literal_value) where
    # literal_value is True if the clause contains +var, False if -var.
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occurrences[var].append((ci, lit > 0))

    def dpll_with_unit_prop(seed, time_budget=10):
        """DPLL with unit propagation and randomized restarts."""
        rng = random.Random(seed)
        start_time = time.time()
        
        # Work with a copy of clauses as lists of literals
        # Each literal is a signed int (1-based)
        
        def propagate(clauses_list, assignment):
            """Unit propagation. Returns (new_assignment, new_clauses_list, contradiction)"""
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses_list:
                    # Check if clause is satisfied, unit, or still open
                    sat = False
                    unit_lit = None
                    open_lits = []
                    for lit in clause:
                        var = abs(lit) - 1
                        if var in assignment:
                            val = assignment[var]
                            if (lit > 0 and val) or (lit < 0 and not val):
                                sat = True
                                break
                            # else: literal is false, skip
                        else:
                            open_lits.append(lit)
                    
                    if sat:
                        continue  # clause satisfied, drop it
                    elif len(open_lits) == 0:
                        return assignment, clauses_list, True  # contradiction
                    elif len(open_lits) == 1:
                        unit_lit = open_lits[0]
                        var = abs(unit_lit) - 1
                        val = unit_lit > 0
                        if var in assignment:
                            if assignment[var] != val:
                                return assignment, clauses_list, True
                        else:
                            assignment[var] = val
                            changed = True
                        # Clause becomes satisfied by unit propagation, drop it
                        continue
                    else:
                        new_clauses.append(open_lits)
            
            return assignment, new_clauses, False
        
        # Main DPLL with restarts
        best_result = None
        best_unsat = float('inf')
        
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break
            
            # Random initial assignment for some variables to guide search
            assignment = {}
            # Assign a few random variables to help break symmetry
            num_initial = min(n // 5, 30)
            for var in rng.sample(range(n), num_initial):
                assignment[var] = rng.random() < 0.5
            
            clauses_copy = [list(c) for c in clauses]
            
            # Apply unit propagation
            assignment, clauses_copy, contradiction = propagate(clauses_copy, assignment)
            
            if contradiction:
                # Try again with different random seed
                continue
            
            # If all clauses satisfied, we're done
            if not clauses_copy:
                # Build full assignment
                result = [False] * n
                for var in range(n):
                    result[var] = assignment.get(var, rng.random() < 0.5)
                return result
            
            # Now do a local search to finish
            result = [assignment.get(var, rng.random() < 0.5) for var in range(n)]
            
            # Count unsatisfied clauses
            unsat_count = 0
            for clause in clauses_copy:
                if not any(result[abs(l)-1] == (l > 0) for l in clause):
                    unsat_count += 1
            
            if unsat_count == 0:
                return result
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_result = result[:]
        
        return best_result if best_result else [False] * n

    def random_walksat(seed):
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # sat_count[c] = number of satisfied literals in clause c
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1

        unsat_set = set(i for i in range(m) if sat_count[i] == 0)

        # If already satisfied
        if not unsat_set:
            return assignment

        # WalkSAT with clause-count tracking
        # Parameters: noise probability, max steps
        noise = 0.3
        max_steps = 500000

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if not unsat_set:
                break

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]

            # Score each literal in the clause:
            # gain = number of other unsat clauses that would become satisfied
            # loss = number of currently-sat clauses that would become unsat
            best_var = -1
            best_score = -1
            best_is_noise = False

            # Evaluate each literal in the clause
            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                # If we set assignment[var] = want_val:
                # For each other clause containing this var:
                #   if currently sat_count == 1 and the only sat literal is this var with current value:
                #     flipping makes it unsat (loss)
                #   if currently sat_count == 0:
                #     if this literal would be satisfied after flip: gain
                gain = 0
                loss = 0
                for (oc, lit_val) in var_occurrences[var]:
                    if oc == ci:
                        continue
                    if lit_val == want_val:
                        # This literal would be satisfied after flip
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        # This literal would be unsatisfied after flip
                        if sat_count[oc] == 1:
                            loss += 1
                score = gain - loss
                candidates.append((score, var, want_val))

            # With probability noise, pick random; else pick best
            if rng.random() < noise:
                _, var, want_val = rng.choice(candidates)
            else:
                best = max(candidates, key=lambda x: x[0])
                var, want_val = best[1], best[2]

            # Apply flip
            old_val = assignment[var]
            if old_val == want_val:
                # No change needed
                continue
            assignment[var] = want_val

            # Update sat_counts for all clauses containing var
            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == want_val:
                    # This literal is now satisfied
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    # This literal is now unsatisfied
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)

            if not unsat_set:
                break

        return assignment

    # First try DPLL with unit propagation
    result = dpll_with_unit_prop(0, time_budget=30)
    unsat_count = sum(1 for c in clauses if not any(
        result[abs(l)-1] == (l > 0) for l in c))
    if unsat_count == 0:
        return result
    
    best_assignment = result
    best_unsat = unsat_count

    # If DPLL didn't solve it, try WalkSAT from the DPLL result
    if time.time() < deadline - 10:
        # Initialize WalkSAT from the best DPLL result
        rng = random.Random(0)
        assignment = best_assignment[:]
        
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1
        
        unsat_set = set(i for i in range(m) if sat_count[i] == 0)
        
        if not unsat_set:
            return assignment
        
        noise = 0.3
        max_steps = 500000
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if not unsat_set:
                break
            
            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]
            
            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                gain = 0
                loss = 0
                for (oc, lit_val) in var_occurrences[var]:
                    if oc == ci:
                        continue
                    if lit_val == want_val:
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        if sat_count[oc] == 1:
                            loss += 1
                score = gain - loss
                candidates.append((score, var, want_val))
            
            if rng.random() < noise:
                _, var, want_val = rng.choice(candidates)
            else:
                best = max(candidates, key=lambda x: x[0])
                var, want_val = best[1], best[2]
            
            old_val = assignment[var]
            if old_val == want_val:
                continue
            assignment[var] = want_val
            
            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == want_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)
            
            if not unsat_set:
                break
        
        unsat_count = sum(1 for c in clauses if not any(
            assignment[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment

    # Try more DPLL restarts with different seeds
    for seed in range(1, 20):
        if time.time() > deadline - 5:
            break
        result = dpll_with_unit_prop(seed, time_budget=5)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Try more WalkSAT restarts
    for seed in range(1, 30):
        if time.time() > deadline - 3:
            break
        result = random_walksat(seed)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment
# EVOLVE-BLOCK-END
