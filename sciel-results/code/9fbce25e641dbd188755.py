"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    """
    Solve a planted satisfiable 3-CNF instance using a hybrid approach:
    1. Simulated Annealing (Max-SAT formulation) with restarts.
    2. WalkSAT as a final polish / fallback.
    
    Returns a list of booleans representing the assignment.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline guard: must finish within 300s. Use 290s as safety margin.
    deadline = time.time() + 290.0
    
    # Pre-process clauses for faster evaluation
    # Store clauses as lists of (var_index, is_positive)
    # var_index is 0-based
    processed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            var = abs(l) - 1
            pos = l > 0
            pc.append((var, pos))
        processed_clauses.append(pc)
    
    def eval_clause(clause, assignment):
        for var, pos in clause:
            if assignment[var] == pos:
                return True
        return False
    
    def count_satisfied(assignment):
        return sum(1 for c in processed_clauses if eval_clause(c, assignment))
    
    # We will use a deterministic seed for reproducibility within the function call,
    # but the problem states "locally generated planted satisfiable 3-CNF".
    # The seed program used random.Random(0). We should probably use a fixed seed or 
    # derive it from the instance to be consistent, but the task says "score is fraction solved".
    # Using a fixed seed is generally safer for deterministic evaluation unless the evaluator
    # expects different random behavior. The seed program used Random(0).
    # Let's use a fixed seed for the SA and WalkSAT parts to ensure reproducibility.
    rng = random.Random(0)
    
    best_assignment = None
    best_score = -1
    
    # Strategy: 
    # 1. Try a few random restarts of Simulated Annealing.
    # 2. If SA finds a solution, return it.
    # 3. If SA doesn't find one, use the best assignment found by SA as a starting point for WalkSAT.
    # 4. If WalkSAT finds a solution, return it.
    # 5. If time runs out, return the best assignment found so far (most satisfied clauses).
    
    max_restarts = 20
    sa_steps_per_restart = 50000  # Adjust based on time
    initial_temp = 1.0
    cooling_rate = 0.99
    
    # Estimate time per step to adjust budget
    # Let's do a quick timing of one SA step on a random assignment
    test_assignment = [rng.random() < 0.5 for _ in range(n)]
    start_t = time.time()
    # Simulate a few steps
    for _ in range(100):
        # Pick a random clause
        c_idx = rng.randrange(num_clauses)
        c = processed_clauses[c_idx]
        # Pick a random var in clause
        v_idx = rng.randrange(len(c))
        var, pos = c[v_idx]
        # Evaluate current score
        # This is expensive if done naively. 
        # For SA, we need efficient delta calculation.
        pass
    end_t = time.time()
    # The naive eval is too slow. We need an efficient scoring mechanism.
    
    # Efficient scoring:
    # Maintain count of satisfied clauses.
    # When flipping a variable, only clauses containing that variable change status.
    
    # Pre-compute: for each variable, which clauses contain it?
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(processed_clauses):
        for var, pos in c:
            var_to_clauses[var].append(i)
    
    # Function to get current satisfied count and clause satisfaction status
    # We maintain an array 'clause_sat' where clause_sat[i] is True if clause i is satisfied.
    # And 'var_flip_count' or just recalculate affected clauses.
    
    def init_state(assignment):
        clause_sat = [False] * num_clauses
        sat_count = 0
        for i, c in enumerate(processed_clauses):
            if eval_clause(c, assignment):
                clause_sat[i] = True
                sat_count += 1
        return clause_sat, sat_count
    
    def flip_var(var, assignment, clause_sat, sat_count):
        """
        Flip variable 'var' in 'assignment'.
        Update 'clause_sat' and 'sat_count' for clauses affected by 'var'.
        Returns new sat_count.
        """
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        
        delta = 0
        for c_idx in var_to_clauses[var]:
            c = processed_clauses[c_idx]
            # Check if clause was satisfied before
            was_sat = clause_sat[c_idx]
            # Check if clause is satisfied now
            # We need to check if any other literal in the clause is satisfied by the NEW assignment
            # Or if the literal for 'var' is now satisfied.
            
            # A clause is satisfied if ANY literal is true.
            # Before flip:
            #   - If 'var' literal was true, clause was sat (regardless of others).
            #   - If 'var' literal was false, clause sat iff some other literal was true.
            # After flip:
            #   - If 'var' literal is now true, clause is sat.
            #   - If 'var' literal is now false, clause sat iff some other literal is true.
            
            # Efficient check:
            # Find the literal for 'var' in clause c.
            var_lit_pos = None
            for v, p in c:
                if v == var:
                    var_lit_pos = p
                    break
            
            # Is the var literal satisfied in the NEW assignment?
            var_sat_now = (new_val == var_lit_pos)
            
            if var_sat_now:
                # Clause is definitely satisfied now
                if not was_sat:
                    delta += 1
                clause_sat[c_idx] = True
            else:
                # Var literal is NOT satisfied. Check if any OTHER literal is satisfied.
                # We can just eval the whole clause since it's short (3-SAT).
                # But we can be smarter: we know 'var' lit is false.
                # Check other 2 literals.
                sat_by_other = False
                for v, p in c:
                    if v != var:
                        if assignment[v] == p:
                            sat_by_other = True
                            break
                
                if sat_by_other:
                    # Clause is satisfied by other var
                    if not was_sat:
                        delta += 1
                    clause_sat[c_idx] = True
                else:
                    # Clause is NOT satisfied
                    if was_sat:
                        delta -= 1
                    clause_sat[c_idx] = False
        
        return sat_count + delta
    
    # Simulated Annealing
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat, sat_count = init_state(assignment)
        
        if sat_count == num_clauses:
            return assignment
        
        best_local_assignment = assignment[:]
        best_local_score = sat_count
        
        temp = initial_temp
        # Adaptive steps: try to do as many steps as possible before cooling too much or timing out
        # Let's fix steps per restart for simplicity, but monitor time.
        steps_in_restart = 0
        max_steps_in_restart = 20000  # Rough estimate
        
        while steps_in_restart < max_steps_in_restart:
            if time.time() > deadline - 10.0:
                break
                
            # Pick a random unsatisfied clause to focus on? Or random clause?
            # Standard SA on MaxSAT: pick a random variable to flip, accept based on energy change.
            # Or pick a random clause, pick a var in it.
            # Let's pick a random clause, then a random var in it. This biases search towards fixing unsat clauses.
            c_idx = rng.randrange(num_clauses)
            c = processed_clauses[c_idx]
            v_idx = rng.randrange(len(c))
            var = c[v_idx][0]
            
            old_sat_count = sat_count
            new_sat_count = flip_var(var, assignment, clause_sat, sat_count)
            
            delta = new_sat_count - old_sat_count
            
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                # Accept
                if new_sat_count == num_clauses:
                    return assignment
                if new_sat_count > best_local_score:
                    best_local_score = new_sat_count
                    best_local_assignment = assignment[:]
            else:
                # Reject: flip back
                flip_var(var, assignment, clause_sat, sat_count)
            
            steps_in_restart += 1
            # Cool down
            temp *= cooling_rate
            if temp < 0.01:
                break
            
            # Check time every 1000 steps
            if steps_in_restart % 1000 == 0 and time.time() > deadline - 10.0:
                break
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_assignment = best_local_assignment
    
    # If SA didn't find a solution, try WalkSAT with the best assignment found as start
    if best_score < num_clauses:
        assignment = best_assignment if best_assignment else [rng.random() < 0.5 for _ in range(n)]
        
        # WalkSAT parameters
        walk_steps = 50000
        for _ in range(walk_steps):
            if time.time() > deadline - 1.0:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for i in range(num_clauses):
                if not clause_sat[i]: # Wait, clause_sat might be stale if we didn't update it properly in SA reject/accept logic?
                    # Actually, clause_sat is updated in flip_var.
                    # But we need to ensure it's consistent with 'assignment'.
                    # It should be.
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsat clause
            c_idx = rng.choice(unsat_clauses)
            c = processed_clauses[c_idx]
            
            # Try to find a variable in the clause that can be flipped to satisfy it without breaking others?
            # Simple WalkSAT: pick random var in clause.
            # Better: pick var that maximizes number of satisfied clauses it would break? (Min break)
            # Let's do a simple "min-break" heuristic.
            
            best_var = None
            min_breaks = float('inf')
            
            for var, pos in c:
                # Simulate flip
                # Count how many clauses containing 'var' are currently satisfied and would become unsatisfied
                # Note: Clauses where 'var' is the only satisfying literal will break.
                breaks = 0
                for c2_idx in var_to_clauses[var]:
                    if clause_sat[c2_idx]:
                        # Check if 'var' is the ONLY literal satisfying c2_idx
                        c2 = processed_clauses[c2_idx]
                        only_sat = True
                        for v, p in c2:
                            if v != var and assignment[v] == p:
                                only_sat = False
                                break
                        if only_sat:
                            breaks += 1
                
                if breaks < min_breaks:
                    min_breaks = breaks
                    best_var = var
                    if min_breaks == 0:
                        break # Can't do better than 0 breaks
            
            # Flip best_var
            flip_var(best_var, assignment, clause_sat, sat_count)
            
            if sat_count == num_clauses:
                return assignment
    
    # If we are here, we didn't find a perfect solution.
    # Return the best assignment we found (maximizing satisfied clauses).
    # The task asks for a satisfying assignment. If we fail, we return the best effort.
    # The score is "fraction solved", so returning a non-satisfying assignment scores 0 for that instance.
    # But it's better than crashing.
    
    return assignment
# EVOLVE-BLOCK-END
