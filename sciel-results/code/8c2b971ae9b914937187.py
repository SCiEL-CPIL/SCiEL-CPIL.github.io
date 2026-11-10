import random
import time
import math
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
        
    deadline = time.time() + 290

    # 1. Preprocessing: Build Factor Graph Structure
    # clause_vars[c] = list of (var_idx, sign)
    # var_occurrences[v] = list of (clause_idx, sign)
    clause_vars = []
    var_occurrences = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        lits = []
        for l in clause:
            vi = abs(l) - 1
            sign = 1 if l > 0 else -1
            lits.append((vi, sign))
            var_occurrences[vi].append((ci, sign))
        clause_vars.append(lits)

    # 2. Fast initialization helpers

    def compute_counts(assignment):
        """Compute satisfaction counts for all clauses."""
        counts = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            counts[ci] = cnt
        return counts

    def get_minconflicts_init(rng, steps=500):
        """
        MinConflicts-style warm-up: repeatedly pick a random unsatisfied clause
        and flip the variable that breaks the fewest other clauses.
        This is a fast, effective way to get close to a solution.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        counts = compute_counts(assignment)
        
        for _ in range(steps):
            if time.time() > deadline:
                break
                
            # Find unsatisfied clauses
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                break
                
            # Pick random unsat clause
            ci = rng.choice(unsat)
            
            # Find variable in this clause that breaks the fewest other clauses
            best_vi = None
            min_breaks = float('inf')
            
            for vi, sign in clause_vars[ci]:
                breaks = 0
                for oci, osign in var_occurrences[vi]:
                    if oci == ci:
                        continue
                    # If the literal in oci is currently satisfied, and it's the only one,
                    # flipping vi breaks oci.
                    if assignment[vi] == (osign == 1):
                        if counts[oci] == 1:
                            breaks += 1
                            
                if breaks < min_breaks:
                    min_breaks = breaks
                    best_vi = vi
                    
            if best_vi is not None:
                vi = best_vi
                old_val = assignment[vi]
                new_val = not old_val
                assignment[vi] = new_val
                for oci, osign in var_occurrences[vi]:
                    was_sat = old_val == (osign == 1)
                    now_sat = new_val == (osign == 1)
                    if was_sat and not now_sat:
                        counts[oci] -= 1
                    elif not was_sat and now_sat:
                        counts[oci] += 1
                        
        return assignment, counts

    def walk_sat(assignment, counts, max_steps, rng, deadline):
        """
        WalkSAT with weighted variable selection:
        - With probability p, pick a variable from the unsat clause that minimizes breaks
        - With probability (1-p), pick a random variable from the unsat clause
        - With small probability, pick a completely random variable
        """
        p = 0.55  # Probability of choosing min-breaks variable from unsat clause
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
                
            # Find unsatisfied clauses
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts
                
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)
            
            # Decide which variable to flip
            r = rng.random()
            if r < p:
                # Choose from literals in the unsat clause with min-breaks heuristic
                best_vi = None
                min_breaks = float('inf')
                
                for vi, sign in clause_vars[ci]:
                    breaks = 0
                    for oci, osign in var_occurrences[vi]:
                        if oci == ci:
                            continue
                        if assignment[vi] == (osign == 1):
                            if counts[oci] == 1:
                                breaks += 1
                                
                    if breaks < min_breaks:
                        min_breaks = breaks
                        best_vi = vi
                vi = best_vi
            elif r < p + 0.4:
                # Random variable from the unsat clause
                vi, _ = clause_vars[ci][rng.randrange(len(clause_vars[ci]))]
            else:
                # Completely random variable
                vi = rng.randrange(n)
                
            # Flip variable vi
            old_val = assignment[vi]
            new_val = not old_val
            assignment[vi] = new_val
            
            # Update counts
            for oci, osign in var_occurrences[vi]:
                was_sat = old_val == (osign == 1)
                now_sat = new_val == (osign == 1)
                if was_sat and not now_sat:
                    counts[oci] -= 1
                elif not was_sat and now_sat:
                    counts[oci] += 1
                    
        return False, assignment, counts

    def simulated_annealing(assignment, counts, max_steps, rng, deadline):
        """
        Simulated annealing: accept worsening moves with probability exp(-delta/T).
        This helps escape local minima that trap pure local search.
        """
        T = 1.0  # Initial temperature
        cooling = 0.9995  # Cooling rate per step
        min_T = 0.01
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
                
            # Find unsatisfied clauses
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts
                
            cur_unsat = len(unsat)
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)
            
            # Pick a random variable from this clause to flip
            vi, sign = clause_vars[ci][rng.randrange(len(clause_vars[ci]))]
            
            # Try the flip
            old_val = assignment[vi]
            new_val = not old_val
            assignment[vi] = new_val
            
            # Update counts and compute delta
            delta = 0
            for oci, osign in var_occurrences[vi]:
                was_sat = old_val == (osign == 1)
                now_sat = new_val == (osign == 1)
                if was_sat and not now_sat:
                    counts[oci] -= 1
                    if counts[oci] == 0:
                        delta += 1  # Became unsatisfied
                elif not was_sat and now_sat:
                    counts[oci] += 1
                    if counts[oci] == 1:
                        delta -= 1  # Became satisfied (was unsat)
                        
            new_unsat = cur_unsat + delta
            
            # Accept or reject
            if delta <= 0:
                # Improvement or neutral: always accept
                pass
            else:
                # Worsening: accept with probability exp(-delta/T)
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        pass  # Accept
                    else:
                        # Reject: undo the flip
                        assignment[vi] = old_val
                        for oci, osign in var_occurrences[vi]:
                            was_sat = old_val == (osign == 1)
                            now_sat = new_val == (osign == 1)
                            if was_sat and not now_sat:
                                counts[oci] -= 1
                            elif not was_sat and now_sat:
                                counts[oci] += 1
                else:
                    # Reject at low temperature
                    assignment[vi] = old_val
                    for oci, osign in var_occurrences[vi]:
                        was_sat = old_val == (osign == 1)
                        now_sat = new_val == (osign == 1)
                        if was_sat and not now_sat:
                            counts[oci] -= 1
                        elif not was_sat and now_sat:
                            counts[oci] += 1
                
            # Cool down
            if T > min_T:
                T *= cooling
                
        return False, assignment, counts

    def polish(assignment, counts, rng, deadline, max_rounds=50):
        """
        Final polish: repeatedly try to fix the most-constrained unsatisfied clauses.
        For each unsat clause, try flipping each variable and keep the best improvement.
        """
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
                
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts
            
            # Pick the unsat clause with the most "potential" (variables that could help)
            ci = rng.choice(unsat)
            
            best_vi = None
            best_delta = float('inf')
            
            for vi, sign in clause_vars[ci]:
                # Try flipping vi and compute net change
                delta = 0
                for oci, osign in var_occurrences[vi]:
                    was_sat = assignment[vi] == (osign == 1)
                    now_sat = not was_sat
                    if was_sat and not now_sat:
                        if counts[oci] == 1:
                            delta += 1
                    elif not was_sat and now_sat:
                        if counts[oci] == 0:
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_vi = vi
            
            if best_vi is not None and best_delta < 0:
                # Apply the flip
                vi = best_vi
                old_val = assignment[vi]
                new_val = not old_val
                assignment[vi] = new_val
                for oci, osign in var_occurrences[vi]:
                    was_sat = old_val == (osign == 1)
                    now_sat = new_val == (osign == 1)
                    if was_sat and not now_sat:
                        counts[oci] -= 1
                    elif not was_sat and now_sat:
                        counts[oci] += 1
            else:
                # No improving flip found; break
                break
                
        unsat = [ci for ci in range(m) if counts[ci] == 0]
        return len(unsat) == 0, assignment, counts

    # 3. Main Solver Loop
    best_unsat = m + 1
    best_assignment = None
    rng = random.Random(12345)
    
    max_restarts = 2000
    walk_steps = 3000
    sa_steps = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
            
        # Alternate between different initialization strategies
        if restart % 4 == 0:
            # Pure random init
            assignment = [rng.random() < 0.5 for _ in range(n)]
            counts = compute_counts(assignment)
        elif restart % 4 == 1:
            # Short MinConflicts warm-up
            assignment, counts = get_minconflicts_init(rng, steps=200)
        elif restart % 4 == 2:
            # Longer MinConflicts warm-up
            assignment, counts = get_minconflicts_init(rng, steps=500)
        else:
            # Medium MinConflicts warm-up
            assignment, counts = get_minconflicts_init(rng, steps=350)
        
        # Check if solved
        cur_unsat = sum(1 for c in counts if c == 0)
        if cur_unsat == 0:
            return assignment
            
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        # Run WalkSAT
        solved, assignment, counts = walk_sat(assignment, counts, walk_steps, rng, deadline)
        
        if solved:
            return assignment
            
        cur_unsat = sum(1 for c in counts if c == 0)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        # If we're close but not solved, try simulated annealing to escape local min
        if cur_unsat <= m * 0.1:  # Less than 10% unsatisfied
            sa_solved, assignment, counts = simulated_annealing(assignment, counts, sa_steps, rng, deadline)
            if sa_solved:
                return assignment
                
            cur_unsat = sum(1 for c in counts if c == 0)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
            
            # Try polish
            polish_solved, assignment, counts = polish(assignment, counts, rng, deadline)
            if polish_solved:
                return assignment
                
            cur_unsat = sum(1 for c in counts if c == 0)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return [False] * n
