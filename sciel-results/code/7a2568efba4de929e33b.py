"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    DEADLINE = 290.0  # Safety margin before 300s limit

    # Preprocess: convert to 0-based variable indices and signs
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

    # Precompute variable degrees for static heuristic
    var_degree = [0] * n
    for i in range(m):
        for j in range(3):
            var_degree[clause_vars[i][j]] += 1

    # Precompute clause weights (sum of degrees of its variables)
    clause_weights = [0] * m
    for i in range(m):
        w = 0
        for j in range(3):
            w += var_degree[clause_vars[i][j]]
        clause_weights[i] = w

    rng = random.Random(42)

    def make_random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def walk_sat_run(assignment, max_flips, p=0.55):
        """WalkSAT with optimized list-based unsat clause tracking and static clause weighting."""
        
        # Initialize clause satisfaction status
        clause_sat_count = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            cnt = 0
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    cnt += 1
            clause_sat_count[i] = cnt

        # Initialize unsat_list and unsat_pos for O(1) selection and updates
        unsat_list = []
        unsat_pos = [-1] * m  # -1 if not in unsat_list, else index in unsat_list
        
        for i in range(m):
            if clause_sat_count[i] == 0:
                unsat_pos[i] = len(unsat_list)
                unsat_list.append(i)

        for flip in range(max_flips):
            if time.time() - start_time > DEADLINE:
                return None, assignment

            if len(unsat_list) == 0:
                return assignment[:], assignment

            # Pick an unsatisfied clause
            # Strategy: With high probability, pick the clause with highest static weight.
            # With low probability, pick randomly to escape local minima.
            if rng.random() < 0.8:
                # Find max weight in unsat_list
                best_ci = -1
                best_w = -1
                # Optimization: if list is small, just scan. If large, sampling might be better, but m is small.
                for idx in unsat_list:
                    if clause_weights[idx] > best_w:
                        best_w = clause_weights[idx]
                        best_ci = idx
                ci = best_ci
            else:
                idx = rng.randint(0, len(unsat_list) - 1)
                ci = unsat_list[idx]

            # Choose literal to flip
            if rng.random() < p:
                # Random literal in the clause
                li = rng.randint(0, 2)
            else:
                # Weighted: prefer literals that satisfy more other unsat clauses
                best_li = 0
                best_weight = -1
                for li in range(3):
                    var = clause_vars[ci][li]
                    weight = 0
                    for (oi, oj) in var_to_clauses[var]:
                        if oi == ci:
                            continue
                        # Check if oi is currently unsat
                        if unsat_pos[oi] != -1:
                            weight += 1
                    if weight > best_weight:
                        best_weight = weight
                        best_li = li
                li = best_li

            var = clause_vars[ci][li]
            old_val = assignment[var]
            new_val = not old_val
            assignment[var] = new_val

            # Update clause_sat_count and unsat_list for all clauses containing this var
            for (oi, oj) in var_to_clauses[var]:
                osig = clause_sigs[oi][oj]
                was_sat = (old_val == (osig > 0))
                now_sat = (new_val == (osig > 0))
                
                if was_sat and not now_sat:
                    # Clause became unsat
                    clause_sat_count[oi] -= 1
                    if clause_sat_count[oi] == 0:
                        # Add to unsat_list
                        unsat_pos[oi] = len(unsat_list)
                        unsat_list.append(oi)
                elif not was_sat and now_sat:
                    # Clause became sat
                    clause_sat_count[oi] += 1
                    if clause_sat_count[oi] > 0:
                        # Remove from unsat_list if it was there
                        if unsat_pos[oi] != -1:
                            pos = unsat_pos[oi]
                            last_idx = len(unsat_list) - 1
                            last_clause = unsat_list[last_idx]
                            
                            # Swap with last
                            unsat_list[pos] = last_clause
                            unsat_pos[last_clause] = pos
                            
                            # Pop last
                            unsat_list.pop()
                            unsat_pos[oi] = -1

        # Return None if not solved, but return the final assignment for tracking
        if len(unsat_list) == 0:
            return assignment[:], assignment
        return None, assignment

    def count_unsat(assignment):
        unsat_count = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            sat = False
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        return unsat_count

    def polish(assignment, unsat_clauses):
        """Try to fix a small number of unsat clauses by flipping variables."""
        if not unsat_clauses:
            return assignment
        # If there are very few unsat clauses, try flipping each variable in them
        # and see if it reduces the count further.
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment
        
        # Collect unique variables from unsat clauses
        vars_to_try = set()
        for ci in unsat_clauses:
            for j in range(3):
                vars_to_try.add(clause_vars[ci][j])
        
        best_assign = assignment[:]
        best_count = current_unsat
        
        for var in vars_to_try:
            # Flip var
            assignment[var] = not assignment[var]
            new_count = count_unsat(assignment)
            if new_count < best_count:
                best_count = new_count
                best_assign = assignment[:]
            # Unflip var
            assignment[var] = not assignment[var]
            
        return best_assign

    best_assignment = None
    best_unsat = m + 1

    # Luby sequence for restarts: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
    # Base flip budget
    base_flips = 100
    
    # Generate Luby sequence on the fly
    def luby(k):
        # k is 0-indexed
        # Find largest power of 2 less than or equal to k+1
        # Luby sequence is defined recursively.
        # Simpler: precompute a few or use a generator.
        # For simplicity, we'll use a geometric progression with randomization, 
        # or just a standard Luby implementation.
        pass

    # Let's use a simpler adaptive strategy: 
    # Restart budget increases as we fail.
    # budget = base * (restart + 1) ** 1.5 ?
    # Or just keep a list of budgets.
    
    max_restarts = 1000
    # We will use a Luby-like sequence of multipliers
    # 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8
    # Multipliers:
    multipliers = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    # Cycle through these, but also allow increasing the base if we are stuck.
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        # Determine flip budget
        # Use Luby multiplier
        mult = multipliers[restart % len(multipliers)]
        # Also increase base budget slightly with restart to ensure deeper search eventually
        # base_flips increases every 10 restarts
        current_base = base_flips + (restart // 10) * 100
        flips = current_base * mult
        
        # Cap flips to avoid wasting time on a single run
        if flips > 5000:
            flips = 5000

        assignment = make_random_assignment()
        
        p = 0.55
        result, final_assignment = walk_sat_run(assignment, max_flips=flips, p=p)
        
        if result is not None:
            return result

        # Track best
        unsat_count = count_unsat(final_assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = final_assignment[:]
            
            # If we are very close, try polishing
            if unsat_count <= 3:
                # Get list of unsat clauses
                unsat_clauses = []
                for i in range(m):
                    cv = clause_vars[i]
                    cs = clause_sigs[i]
                    sat = False
                    for j in range(3):
                        if final_assignment[cv[j]] == (cs[j] > 0):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(i)
                
                polished = polish(final_assignment, unsat_clauses)
                polished_unsat = count_unsat(polished)
                if polished_unsat == 0:
                    return polished
                if polished_unsat < best_unsat:
                    best_unsat = polished_unsat
                    best_assignment = polished[:]

    if best_assignment is not None:
        return best_assignment

    return [True] * n
# EVOLVE-BLOCK-END
