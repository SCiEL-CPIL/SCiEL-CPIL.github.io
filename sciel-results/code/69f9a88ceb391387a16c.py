import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s limit, keep 5s safety margin
    start_time = time.time()
    deadline = start_time + 295.0
    
    if m == 0:
        return [True] * n

    # Precompute var_to_clauses for efficient updates
    # var_to_clauses[var] = list of (clause_idx, literal_sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_to_clauses[var].append((i, lit))

    def is_satisfied(assignment):
        """Check if the full assignment satisfies all clauses."""
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

    def run_walksat(assignment, rng, p_walk, max_steps):
        """
        Efficient WalkSAT with best-gain heuristic.
        Returns True if solved, False otherwise.
        Modifies assignment in place.
        """
        # Initialize clause satisfaction status
        clause_sat = [False] * m
        unsat_count = 0
        
        for i, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            clause_sat[i] = sat
            if not sat:
                unsat_count += 1
        
        if unsat_count == 0:
            return True

        for step in range(max_steps):
            if time.time() > deadline - 1.0:
                return False
            
            if unsat_count == 0:
                return True
            
            # Pick a random unsatisfied clause
            # Optimization: Instead of scanning all clauses to find unsat ones,
            # we can just pick a random index and check. If it's sat, try again.
            # With high density, this is fast. If unsat_count is very low, 
            # a list of unsat indices would be better, but maintaining it is costly.
            # For 3-SAT with ~4.26 ratio, random sampling is usually fine.
            ci = rng.randrange(m)
            while clause_sat[ci]:
                ci = rng.randrange(m)
            
            clause = clauses[ci]
            
            # Find the best variable to flip in this clause
            # We want to minimize the number of other clauses that become unsatisfied
            best_gain = -1
            best_var = -1
            best_flips = []
            
            for lit in clause:
                var = abs(lit) - 1
                current_val = assignment[var]
                desired_val = (lit > 0)
                
                # If this literal is already set to satisfy the clause, 
                # flipping it would make this clause worse (or keep it unsat if others exist, 
                # but we picked an unsat clause, so if one is sat, the clause is sat. 
                # Wait, if clause is unsat, NO literal in it is satisfied.
                # So current_val != desired_val for all lits in clause.
                # Thus, flipping any of them will satisfy this clause.
                
                # Calculate gain: 
                # +1 for satisfying this clause (ci)
                # -1 for each other clause (ci2) containing var that is currently satisfied 
                # and will become unsatisfied.
                
                gain = 1
                for (ci2, lit2) in var_to_clauses[var]:
                    if ci2 == ci:
                        continue
                    if clause_sat[ci2]:
                        # Check if this literal is the ONLY one satisfying ci2
                        # If yes, flipping var will unsatisfy ci2.
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
                
                best_flips.append((var, gain))
                
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            
            # Choose flip: with probability p_walk, pick random from clause; 
            # else pick best gain.
            if rng.random() < p_walk:
                var, _ = best_flips[rng.randrange(len(best_flips))]
            else:
                # Pick best. If multiple have same best gain, pick one randomly among them
                # to avoid deterministic loops.
                max_gain = best_gain
                candidates = [v for v, g in best_flips if g == max_gain]
                var = candidates[rng.randrange(len(candidates))]
            
            # Apply flip
            assignment[var] = not assignment[var]
            
            # Update clause satisfaction for all clauses containing var
            for (ci2, lit2) in var_to_clauses[var]:
                was_sat = clause_sat[ci2]
                # Recompute satisfaction for ci2
                sat = False
                for lit3 in clauses[ci2]:
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

    # Main search loop
    # Strategy: Multiple restarts with different seeds and p_walk parameters.
    # For planted SAT, p_walk around 0.3-0.5 is typically good.
    
    base_steps = max(500, n * 10)
    
    restart_id = 0
    while time.time() < deadline - 5.0:
        # Create a new random assignment
        seed = restart_id * 1000003 + 42
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check
        if is_satisfied(assignment):
            return assignment
        
        # Vary p_walk slightly to explore different regions of the search space
        # p_walk = 0.3 + 0.4 * (restart_id % 5) / 4.0
        p_walk = 0.4 + 0.1 * (restart_id % 3) # 0.4, 0.5, 0.6
        
        # Run WalkSAT
        # Use a step budget that allows many restarts
        steps = base_steps * 2 
        
        if run_walksat(assignment, rng, p_walk, steps):
            return assignment
        
        restart_id += 1
        
        # If we are running out of time, do one final long run?
        # No, restarts are usually better for SAT.
        
        # Break if we've done many restarts and time is low
        if time.time() > deadline - 10.0:
             # One last desperate attempt with a different p
             rng = random.Random(restart_id * 7919 + 13)
             assignment = [rng.random() < 0.5 for _ in range(n)]
             if is_satisfied(assignment):
                 return assignment
             if run_walksat(assignment, rng, 0.5, 10000):
                 return assignment
             break

    # If we exhausted time without finding a solution, return the last assignment
    # (It will be scored as incorrect, but we must return something)
    return assignment

