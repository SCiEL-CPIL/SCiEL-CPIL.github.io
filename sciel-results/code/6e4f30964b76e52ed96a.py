"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 330s limit, use 325s to be safe
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Preprocess clauses for faster access
    # Store as list of lists of (var_index, is_positive)
    # var_index is 0-based
    processed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pc.append((var_idx, is_pos))
        processed_clauses.append(pc)
    
    num_clauses = len(processed_clauses)
    
    # Helper to check satisfaction of a clause given an assignment
    # assignment is a list of booleans
    def is_satisfied(clause, assignment):
        for var_idx, is_pos in clause:
            if assignment[var_idx] == is_pos:
                return True
        return False

    # Helper to count unsatisfied clauses
    def count_unsat(assignment):
        return sum(0 if is_satisfied(c, assignment) else 1 for c in processed_clauses)

    # WalkSAT with Restarts
    # Parameters tuned for 50 vars
    max_steps_per_restart = 5000
    max_restarts = 50
    p_flip_random = 0.25  # Probability of flipping a random literal in the clause instead of the best one
    
    best_assignment = [False] * n
    best_unsat_count = num_clauses
    
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        # Check time
        if time.time() > deadline:
            break
            
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        for step in range(max_steps_per_restart):
            # Check time every 100 steps
            if step % 100 == 0 and time.time() > deadline:
                return best_assignment
            
            # Find an unsatisfied clause
            # For efficiency, we could maintain a list of unsat clauses, 
            # but for 50 vars and reasonable clause counts, scanning is okay.
            # However, scanning all clauses every step is O(C). 
            # Let's optimize by picking a random clause and checking if it's unsat.
            # If it is, we use it. If not, we pick another.
            # Actually, finding *any* unsat clause quickly is important.
            # Let's just pick a random clause index and check.
            # If the formula is hard, many clauses are unsat.
            # If it's easy, few are unsat.
            
            # A better way: maintain a list of unsat clauses? 
            # Given n=50, C is likely ~100-200. Scanning is fast in Python if we use local vars.
            
            unsat_clause = None
            # Try to find an unsat clause. 
            # We can just iterate through clauses until we find one.
            # To avoid bias, we can start at a random offset.
            start_idx = rng.randrange(num_clauses)
            for i in range(num_clauses):
                idx = (start_idx + i) % num_clauses
                c = processed_clauses[idx]
                if not is_satisfied(c, assignment):
                    unsat_clause = c
                    break
            
            if unsat_clause is None:
                # All clauses satisfied
                return assignment

            # Choose a literal to flip
            # Calculate the "gain" for flipping each variable in the clause
            # Gain = (number of clauses that become satisfied) - (number of clauses that become unsatisfied)
            # Simpler heuristic: Count how many *other* clauses contain this variable with the same polarity (would break) vs opposite polarity (would satisfy).
            # Let's use the standard WalkSAT heuristic:
            # 1. Pick a random literal from the clause.
            # 2. With probability p, flip it.
            # 3. With probability 1-p, pick the literal that maximizes the number of satisfied clauses (minimizes new unsat).
            
            # Let's implement the "min-conflicts" variant:
            # For each literal in the unsat clause, calculate the cost of flipping it.
            # Cost = number of other clauses that are currently satisfied but would become unsatisfied if we flip this variable.
            
            best_var = None
            best_cost = float('inf')
            
            for var_idx, is_pos in unsat_clause:
                # If we flip assignment[var_idx], it will change from current_val to not current_val
                # The clause we are in will become satisfied.
                # We need to check other clauses.
                cost = 0
                for c_idx, c in enumerate(processed_clauses):
                    if c is unsat_clause:
                        continue
                    # Check if this clause contains var_idx
                    # Optimization: Pre-compute variable to clause mapping?
                    # For n=50, we can just scan.
                    for v, p in c:
                        if v == var_idx:
                            # This clause contains the variable.
                            # Current state of this clause:
                            # If it is currently satisfied, will it remain satisfied after flip?
                            # It remains satisfied if it has another literal that is true.
                            # It becomes unsatisfied if it was satisfied ONLY by this literal.
                            
                            # Check if clause c is currently satisfied
                            c_satisfied = False
                            only_this_literal = True
                            for v2, p2 in c:
                                if assignment[v2] == p2:
                                    c_satisfied = True
                                    if v2 != var_idx:
                                        only_this_literal = False
                                    break
                            
                            if c_satisfied and only_this_literal:
                                cost += 1
                            break # Found the variable in this clause, no need to check others in this clause
                
                if cost < best_cost:
                    best_cost = cost
                    best_var = var_idx
            
            # Decide which variable to flip
            if rng.random() < p_flip_random:
                # Flip a random variable in the clause
                var_idx, _ = rng.choice(unsat_clause)
            else:
                var_idx = best_var
            
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsat count
            # Instead of recounting everything, we can update incrementally, 
            # but for simplicity and given small N, recounting is safer.
            # However, recounting O(C) every step is O(C*Steps).
            # Let's try to be a bit smarter or just accept the cost for N=50.
            current_unsat = count_unsat(assignment)
            
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

    # If we run out of time or restarts, return the best assignment found
    return best_assignment

# EVOLVE-BLOCK-END
