"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    # Deadline guard: 320s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 320.0

    # Pre-process clauses for incremental updates
    # clause_vars[i] = list of (var_idx, is_positive)
    # var_to_clauses[v] = list of clause indices containing v
    clause_vars = []
    clause_polarity = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        vars_in_c = []
        pols_in_c = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            vars_in_c.append(v)
            pols_in_c.append(p)
            var_to_clauses[v].append(i)
        clause_vars.append(vars_in_c)
        clause_polarity.append(pols_in_c)

    # State variables
    assignment = [False] * n
    clause_sat_count = [0] * m  # Number of satisfied literals in each clause
    clause_is_sat = [False] * m
    unsat_clauses = []  # List of indices of unsatisfied clauses

    def update_clause_sat(ci):
        """Recalculate satisfaction status for clause ci based on current assignment."""
        cvars = clause_vars[ci]
        cpols = clause_polarity[ci]
        count = 0
        for v, p in zip(cvars, cpols):
            if assignment[v] == p:
                count += 1
        clause_sat_count[ci] = count
        clause_is_sat[ci] = (count > 0)
        
    def initialize_state(assignment):
        """Initialize all state variables for a given assignment."""
        unsat_clauses.clear()
        for i in range(m):
            update_clause_sat(i)
            if not clause_is_sat[i]:
                unsat_clauses.append(i)

    def flip_variable(v):
        """Flip variable v and incrementally update state."""
        assignment[v] = not assignment[v]
        # Update all clauses containing v
        for ci in var_to_clauses[v]:
            was_sat = clause_is_sat[ci]
            
            # Recalculate count for this clause
            cvars = clause_vars[ci]
            cpols = clause_polarity[ci]
            new_count = 0
            for vv, pp in zip(cvars, cpols):
                if assignment[vv] == pp:
                    new_count += 1
            
            clause_sat_count[ci] = new_count
            is_sat = (new_count > 0)
            clause_is_sat[ci] = is_sat
            
            # Update unsat_clauses list
            if was_sat and not is_sat:
                # Became unsat
                unsat_clauses.append(ci)
            elif not was_sat and is_sat:
                # Became sat, remove from list
                # Using list remove is O(N) but N is small enough for 150 vars usually
                # To be safer, we can swap-remove if we don't care about order
                try:
                    unsat_clauses.remove(ci)
                except ValueError:
                    pass

    def calculate_gain(v, target_clause):
        """
        Calculate the gain of flipping variable v.
        Gain = (# unsat clauses containing v that become sat) - (# sat clauses containing v that become unsat)
        Note: We assume v is in target_clause and target_clause is currently unsat.
        """
        gain = 0
        for ci in var_to_clauses[v]:
            cvars = clause_vars[ci]
            cpols = clause_polarity[ci]
            
            # Check if flipping v satisfies this clause
            # Currently, clause ci has count `clause_sat_count[ci]`.
            # If v is currently satisfying ci, count includes v. If not, it doesn't.
            
            # Does v currently satisfy ci?
            v_satisfies_ci = False
            for vv, pp in zip(cvars, cpols):
                if vv == v:
                    if assignment[v] == pp:
                        v_satisfies_ci = True
                    break
            
            if not clause_is_sat[ci]:
                # Clause is currently unsat.
                # If v does NOT currently satisfy it, flipping v might make it satisfy (if v is the literal needed).
                # Wait, if clause is unsat, NO literal satisfies it.
                # So v does NOT satisfy ci.
                # If we flip v, does v satisfy ci?
                # The literal in ci for v has polarity p.
                # Current assignment[v] != p.
                # New assignment[v] = not assignment[v] == p.
                # So YES, flipping v will satisfy ci.
                gain += 1
            else:
                # Clause is currently sat.
                # If v currently satisfies ci, and it's the ONLY satisfying literal, flipping v breaks it.
                # If v currently satisfies ci, but other literals also satisfy it, flipping v keeps it sat.
                # If v currently does NOT satisfy ci, flipping v doesn't change the fact that others satisfy it.
                
                if v_satisfies_ci:
                    # Check if v is the ONLY satisfying literal
                    if clause_sat_count[ci] == 1:
                        gain -= 1
                    # else gain += 0
                # else gain += 0
        
        return gain

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    max_restarts = 1000
    max_steps_per_restart = 50000
    noise_prob = 0.2

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        initialize_state(assignment)
        
        current_unsat = len(unsat_clauses)
        
        if current_unsat == 0:
            return assignment[:]
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        steps = 0
        while steps < max_steps_per_restart:
            if time.time() > deadline - 5.0:
                break
            
            if not unsat_clauses:
                return assignment[:]

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cvars = clause_vars[ci]
            
            # Calculate gain for each variable in the clause
            best_v = -1
            best_gain = -1000
            
            # Optional: Add noise by picking a random var sometimes
            if rng.random() < noise_prob:
                v = rng.choice(cvars)
            else:
                for v in cvars:
                    g = calculate_gain(v, ci)
                    if g > best_gain:
                        best_gain = g
                        best_v = v
                v = best_v
            
            flip_variable(v)
            steps += 1
            
            current_unsat = len(unsat_clauses)
            
            if current_unsat == 0:
                return assignment[:]
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                
            # Early exit if stuck for too long
            if steps % 1000 == 0 and current_unsat == best_unsat_count and steps > 5000:
                break

    if best_assignment:
        return best_assignment
    return assignment

# EVOLVE-BLOCK-END
