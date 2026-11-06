"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance using a multi-strategy WalkSAT approach.
    
    Combines:
    1. Efficient local search with incremental clause tracking
    2. Adaptive random-walk probability
    3. Multiple restarts with different seeds and parameters
    4. Time management to stay within budget
    
    For planted SAT instances, solutions are typically findable via local search
    since the instances are generated from a satisfying assignment.
    """
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: leave 5s safety margin from 330s limit
    deadline = time.time() + 325.0
    
    # Build a literal-to-clauses index for efficient updates
    # var_to_clauses[var] = list of (clause_idx, literal_sign)
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_to_clauses[var].append((i, lit))
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        cnt = 0
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def walk_sat(assignment, max_steps, rng, p=0.5):
        """
        WalkSAT with adaptive random-walk probability.
        Returns True if solution found, False otherwise.
        """
        # Compute initial unsat count and per-clause satisfaction status
        clause_sat = []
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            clause_sat.append(sat)
        
        unsat_count = sum(1 for s in clause_sat if not s)
        if unsat_count == 0:
            return True
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return False
            
            if unsat_count == 0:
                return True
            
            # Pick a random unsatisfied clause
            while True:
                ci = rng.randrange(len(clauses))
                if not clause_sat[ci]:
                    break
            
            clause = clauses[ci]
            
            # Evaluate flip candidates
            best_gain = -1
            best_var = -1
            flips = []
            
            for lit in clause:
                var = abs(lit) - 1
                current_val = assignment[var]
                desired_val = (lit > 0)
                
                if current_val == desired_val:
                    # Flipping this would make it worse (currently satisfies)
                    continue
                
                # Compute gain: how many clauses become satisfied minus how many become unsatisfied
                gain = 1  # This clause becomes satisfied
                for (ci2, lit2) in var_to_clauses[var]:
                    if ci2 == ci:
                        continue
                    if clause_sat[ci2]:
                        # This clause is currently satisfied; check if this literal is the only one satisfying it
                        only = True
                        for lit3 in clauses[ci2]:
                            if lit3 == lit2:
                                continue
                            var3 = abs(lit3) - 1
                            if assignment[var3] == (lit3 > 0):
                                only = False
                                break
                        if only:
                            gain -= 1
                
                flips.append((var, gain))
            
            if not flips:
                # All literals in the clause are already set to satisfy it, but clause is unsat?
                # This shouldn't happen, but handle gracefully
                var = abs(clause[rng.randrange(len(clause))]) - 1
                assignment[var] = not assignment[var]
                continue
            
            # Choose flip: with probability p, pick random; else pick best
            if rng.random() < p:
                var, gain = flips[rng.randrange(len(flips))]
            else:
                # Pick the flip with highest gain
                max_gain = max(g for _, g in flips)
                best_flips = [(v, g) for v, g in flips if g == max_gain]
                var, gain = best_flips[rng.randrange(len(best_flips))]
            
            # Apply flip
            old_val = assignment[var]
            assignment[var] = not assignment[var]
            
            # Update clause satisfaction
            for (ci2, lit2) in var_to_clauses[var]:
                clause = clauses[ci2]
                was_sat = clause_sat[ci2]
                # Recompute satisfaction for this clause
                sat = False
                for lit3 in clause:
                    var3 = abs(lit3) - 1
                    if assignment[var3] == (lit3 > 0):
                        sat = True
                        break
                clause_sat[ci2] = sat
                if was_sat and not sat:
                    unsat_count += 1
                elif not was_sat and sat:
                    unsat_count -= 1
        
        return unsat_count == 0
    
    def local_search_with_restarts(num_restarts, steps_per_restart):
        """Run multiple restarts of WalkSAT."""
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                return None
            
            rng = random.Random(restart * 1000003 + 42)
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Vary the random-walk probability
            p = 0.3 + 0.4 * (restart % 4) / 3.0
            
            if walk_sat(assignment, steps_per_restart, rng, p=p):
                return assignment
        
        return None
    
    # Strategy 1: Quick check with a few random assignments
    for attempt in range(5):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(attempt * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        if is_satisfied(assignment):
            return assignment
    
    # Strategy 2: WalkSAT with adaptive parameters and restarts
    # For planted SAT, the clause-to-variable ratio is typically near the threshold
    # which means solutions are findable but not trivially.
    
    # Use a budget-based approach
    base_steps = max(1000, n * 20)
    
    result = local_search_with_restarts(
        num_restarts=100,
        steps_per_restart=base_steps
    )
    if result is not None:
        return result
    
    # Strategy 3: More aggressive search with different parameters
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(trial * 9973 + 777)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Use higher random-walk probability for more exploration
        p = 0.6 + 0.3 * (trial % 3) / 2.0
        
        if walk_sat(assignment, base_steps * 2, rng, p=p):
            return assignment
    
    # If all else fails, return the last assignment (may not be satisfying)
    return assignment
# EVOLVE-BLOCK-END
