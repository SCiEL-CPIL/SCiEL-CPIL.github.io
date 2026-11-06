# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 330s limit, use 325s to be safe
    deadline = time.time() + 325.0
    
    # Preprocess: convert clauses to arrays of 0-indexed variable and sign
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
    
    # Build variable-to-clause mapping for efficient updates
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append((i, j))
    
    def evaluate_assignment(assignment):
        """
        assignment: numpy array of booleans (length n)
        Returns:
            unsat_clauses: numpy array of indices of unsatisfied clauses
            clause_sat_counts: numpy array of length m, count of true literals per clause
        """
        assignment_flat = assignment[clause_vars]
        sign_bool = clause_signs > 0
        satisfied = assignment_flat == sign_bool
        clause_sat_counts = satisfied.sum(axis=1)
        unsat_clauses = np.where(clause_sat_counts == 0)[0]
        return unsat_clauses, clause_sat_counts
    
    rng = random.Random(12345)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    num_restarts = 0
    max_restarts = 200
    
    # Parameters
    base_step_budget = 10000
    max_step_budget = 50000
    
    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1
        
        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        unsat_clauses, clause_sat_counts = evaluate_assignment(assignment)
        current_unsat = len(unsat_clauses)
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
        
        if current_unsat == 0:
            return assignment.tolist()
        
        # Adaptive step budget
        step_budget = min(base_step_budget + num_restarts * 200, max_step_budget)
        
        # WalkSAT loop with incremental updates
        for step in range(step_budget):
            if time.time() > deadline - 3:
                break
            
            if current_unsat == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_clauses[rng.randrange(len(unsat_clauses))]
            
            # Choose a literal from the clause to flip
            # Heuristic: prefer variables that appear in more unsatisfied clauses
            # For simplicity, pick randomly
            lit_pos = rng.randrange(3)
            var_idx = clause_vars[clause_idx, lit_pos]
            
            # Flip the variable
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val
            
            # Update clause_sat_counts for all clauses containing var_idx
            delta = 1 if new_val else -1
            # Actually, we need to check if the literal is satisfied or not
            # If new_val makes the literal true, sat_count increases
            # If new_val makes the literal false, sat_count decreases
            
            # For each clause containing var_idx:
            for (ci, cj) in var_to_clauses[var_idx]:
                # Check if the literal at (ci, cj) is satisfied by new_val
                sign = clause_signs[ci, cj]
                # Literal is satisfied if assignment[var] == (sign > 0)
                # new_val is the new assignment
                # If sign > 0, literal is satisfied when assignment is True
                # If sign < 0, literal is satisfied when assignment is False
                if sign > 0:
                    new_sat = new_val
                else:
                    new_sat = not new_val
                
                old_sat = not new_sat  # Since we flipped, old satisfaction is opposite
                # Wait, that's not right. old_sat depends on old_val
                # Let's compute properly:
                if sign > 0:
                    old_sat = old_val
                else:
                    old_sat = not old_val
                
                if new_sat and not old_sat:
                    clause_sat_counts[ci] += 1
                elif old_sat and not new_sat:
                    clause_sat_counts[ci] -= 1
            
            # Update unsat_clauses
            # Rebuild unsat_clauses from clause_sat_counts
            # This is O(m) which is expensive, so do it less frequently
            if step % 5 == 0 or step == step_budget - 1:
                unsat_clauses = np.where(clause_sat_counts == 0)[0]
                current_unsat = len(unsat_clauses)
                
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment.copy()
        
        if best_unsat_count == 0:
            return best_assignment.tolist()
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return [False] * n
# EVOLVE-BLOCK-END
