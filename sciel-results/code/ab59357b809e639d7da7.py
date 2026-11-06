    def flip_variable(v, assignment, clause_unsat_count, unsat_pos_count, unsat_neg_count, unsat_clause_indices):
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        # We will update the list in place or create a new one.
        # Let's create a set of unsat indices for O(1) removal/addition?
        # Or just filter the list.
        
        # To be safe and simple, let's just return a new list.
        # But we need to include ALL unsat clauses, not just the affected ones.
        
        # Strategy:
        # 1. Update counts for affected clauses.
        # 2. Rebuild the unsat_clause_indices list by scanning all clauses?
        #    This is O(N_clauses).
        #    Given N_clauses ~ 600, this is negligible compared to the heuristic calculation.
        
        # Let's do that. It's simpler and less bug-prone.
        
        # First, update the counts and clause_unsat_count for affected clauses.
        
        # Affected clauses are in var_pos[v] and var_neg[v]
        affected = set(var_pos[v]) | set(var_neg[v])
        
        for i in affected:
            old_count = clause_unsat_count[i]
            clause = clauses[i]
            
            # Calculate new count
            # ... (logic from before) ...
            
            # Update unsat_pos/neg counts for vars in clause if status changed
            # ... (logic from before) ...
            
            clause_unsat_count[i] = new_count
            
        # Now rebuild unsat_clause_indices
        new_unsat_indices = [i for i in range(num_clauses) if clause_unsat_count[i] > 0]
        
        return new_unsat_indices
