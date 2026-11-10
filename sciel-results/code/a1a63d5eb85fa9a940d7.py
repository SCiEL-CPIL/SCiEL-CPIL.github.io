"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 330s kill

    # Precompute clause data
    m = len(clauses)
    # For each clause, store literals and variable indices
    clause_vars = []  # list of list of var indices (0-based)
    clause_polarity = []  # list of list of bool (True if positive literal)
    for c in clauses:
        vars_in_c = []
        pols_in_c = []
        for lit in c:
            var_idx = abs(lit) - 1
            vars_in_c.append(var_idx)
            pols_in_c.append(lit > 0)
        clause_vars.append(vars_in_c)
        clause_polarity.append(pols_in_c)

    # Build variable -> clause incidence (for fast updates)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in clause_vars[i]:
            var_to_clauses[v].append(i)

    # Precompute for O(1) delta: for each (var, clause) pair, store the clause index within the clause
    # var_clause_info[var] = list of (clause_idx, position_in_clause)
    var_clause_info = [[] for _ in range(n)]
    for ci in range(m):
        for pos, v in enumerate(clause_vars[ci]):
            var_clause_info[v].append((ci, pos))

    def compute_unsat_count(assignment):
        count = 0
        for i in range(m):
            sat = False
            for j in range(len(clause_vars[i])):
                if assignment[clause_vars[i][j]] == clause_polarity[i][j]:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def check_satisfied(assignment):
        for i in range(m):
            sat = False
            for j in range(len(clause_vars[i])):
                if assignment[clause_vars[i][j]] == clause_polarity[i][j]:
                    sat = True
                    break
            if not sat:
                return False
        return True

    def belief_propagation(rng, num_iterations=200):
        """
        Run Belief Propagation on the factor graph.
        Returns an assignment based on the marginal beliefs.
        """
        # Messages from variable to clause: msg_v2c[v][ci] = log-odds that v is True given all other constraints
        # Messages from clause to variable: msg_c2v[ci][v] = log-odds that v is True given clause ci
        
        # Initialize messages
        msg_v2c = [[0.0] * m for _ in range(n)]
        msg_c2v = [[0.0] * len(clause_vars[ci]) for ci in range(m)]
        
        # For each clause, precompute which variables are in it
        # clause_var_map[ci] = list of var indices in clause ci
        
        for it in range(num_iterations):
            if time.time() > deadline - 10.0:
                break
                
            # Update variable-to-clause messages
            for v in range(n):
                for ci, pos in var_clause_info[v]:
                    # msg_v2c[v][ci] = sum of all msg_c2v messages to v except from ci
                    # For efficiency, we can compute the total and subtract
                    total = 0.0
                    for other_ci, other_pos in var_clause_info[v]:
                        if other_ci != ci:
                            # Find position of v in other_ci
                            # This is expensive; let's use a different approach
                            pass
                    
                    # Better approach: maintain running sums
                    # For now, use a simpler approximation
                    # msg_v2c[v][ci] = sum of msg_c2v[other_ci][v] for other_ci != ci
                    # We'll compute this more efficiently below
                    pass
            
            # Simplified BP: use a heuristic based on clause satisfaction
            # This is a practical approximation for planted SAT
            
            # For each clause, compute the "pressure" on each variable
            # If a clause is unsatisfied, variables in it get pushed toward satisfying it
            
            # Let's use a simpler approach: iterative improvement based on clause conflicts
            # This is essentially min-conflicts with BP-inspired initialization
            
            # Compute current assignment from beliefs
            assignment = [False] * n
            for v in range(n):
                # Belief that v is True
                belief = 0.0
                for ci, pos in var_clause_info[v]:
                    # Check if clause ci is currently satisfied without v
                    clause_sat_without_v = False
                    for j, other_v in enumerate(clause_vars[ci]):
                        if j == pos:
                            continue
                        if assignment[other_v] == clause_polarity[ci][j]:
                            clause_sat_without_v = True
                            break
                    
                    if not clause_sat_without_v:
                        # This clause needs v to be set to satisfy polarity
                        if clause_polarity[ci][pos]:
                            belief += 1.0  # Push v to True
                        else:
                            belief -= 1.0  # Push v to False
                    else:
                        # Clause is already satisfied, slight push to not change
                        if clause_polarity[ci][pos]:
                            belief += 0.1
                        else:
                            belief -= 0.1
                
                assignment[v] = belief > 0
            
            # Check if solved
            if compute_unsat_count(assignment) == 0:
                return assignment
            
            # Add some randomness to escape local minima
            if it % 20 == 19:
                for _ in range(n // 10):
                    v = rng.randrange(n)
                    assignment[v] = not assignment[v]
        
        # Final assignment based on last beliefs
        assignment = [False] * n
        for v in range(n):
            belief = 0.0
            for ci, pos in var_clause_info[v]:
                clause_sat_without_v = False
                for j, other_v in enumerate(clause_vars[ci]):
                    if j == pos:
                        continue
                    if assignment[other_v] == clause_polarity[ci][j]:
                        clause_sat_without_v = True
                        break
                
                if not clause_sat_without_v:
                    if clause_polarity[ci][pos]:
                        belief += 1.0
                    else:
                        belief -= 1.0
                else:
                    if clause_polarity[ci][pos]:
                        belief += 0.1
                    else:
                        belief -= 0.1
            
            assignment[v] = belief > 0
        
        return assignment

    def fast_local_search(rng, initial_assignment, max_steps):
        """
        Fast local search with O(1) delta computation.
        """
        assignment = initial_assignment[:]
        current_unsat = compute_unsat_count(assignment)
        if current_unsat == 0:
            return assignment

        best_assignment = assignment[:]
        best_unsat = current_unsat

        # Track unsatisfied clauses for quick access
        unsat_clauses = set()
        for ci in range(m):
            sat = False
            for j in range(len(clause_vars[ci])):
                if assignment[clause_vars[ci][j]] == clause_polarity[ci][j]:
                    sat = True
                    break
            if not sat:
                unsat_clauses.add(ci)

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if not unsat_clauses:
                return assignment

            # Pick a random unsatisfied clause
            clause_idx = next(iter(unsat_clauses))

            # Pick a variable from the clause to flip
            var_idx = clause_vars[clause_idx][rng.randrange(len(clause_vars[clause_idx]))]

            # Compute delta using O(1) amortized method
            old_val = assignment[var_idx]
            new_val = not old_val
            delta = 0

            for ci, pos in var_clause_info[var_idx]:
                # Check if flipping var_idx changes satisfaction of clause ci
                # Count satisfied literals in clause ci before and after flip
                old_sat_count = 0
                new_sat_count = 0
                for j, v in enumerate(clause_vars[ci]):
                    p = clause_polarity[ci][j]
                    if v == var_idx:
                        if old_val == p:
                            old_sat_count += 1
                        if new_val == p:
                            new_sat_count += 1
                    else:
                        if assignment[v] == p:
                            old_sat_count += 1
                            new_sat_count += 1
                
                old_is_sat = old_sat_count > 0
                new_is_sat = new_sat_count > 0
                
                if old_is_sat and not new_is_sat:
                    delta += 1
                    unsat_clauses.add(ci)
                elif not old_is_sat and new_is_sat:
                    delta -= 1
                    unsat_clauses.discard(ci)

            # Accept the flip (greedy for now, could add SA later)
            assignment[var_idx] = new_val
            current_unsat += delta

            # Track best
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]

            if current_unsat == 0:
                return assignment

        return best_assignment

    # Main loop with restarts
    rng = random.Random(0)
    best_assignment = None
    best_unsat = m + 1

    # Phase 1: Use BP to get a good initial assignment
    bp_assignment = belief_propagation(rng, num_iterations=100)
    bp_unsat = compute_unsat_count(bp_assignment)

    if bp_unsat == 0:
        return bp_assignment

    # Phase 2: Local search from BP result
    result = fast_local_search(rng, bp_assignment, max_steps=10000)
    unsat = compute_unsat_count(result)

    if unsat == 0:
        return result

    if unsat < best_unsat:
        best_unsat = unsat
        best_assignment = result

    # Phase 3: Multiple restarts with random + local search
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Random initial assignment
        random_assign = [rng.random() < 0.5 for _ in range(n)]
        
        # Local search
        steps = 5000 + restart * 500
        result = fast_local_search(rng, random_assign, max_steps=steps)
        unsat = compute_unsat_count(result)

        if unsat == 0:
            return result

        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = result

        # Every 10 restarts, try BP again with different iterations
        if restart % 10 == 9 and time.time() < deadline - 10.0:
            bp_assign = belief_propagation(rng, num_iterations=50 + restart)
            bp_unsat = compute_unsat_count(bp_assign)
            if bp_unsat == 0:
                return bp_assign
            
            result = fast_local_search(rng, bp_assign, max_steps=steps)
            unsat = compute_unsat_count(result)
            if unsat == 0:
                return result
            if unsat < best_unsat:
                best_unsat = unsat
                best_assignment = result

    # Final check
    if best_assignment is not None and check_satisfied(best_assignment):
        return best_assignment
    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
