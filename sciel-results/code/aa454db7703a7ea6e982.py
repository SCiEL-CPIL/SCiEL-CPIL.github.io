"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert to 0-based signed integers
    # e.g., literal 1 -> 0, -1 -> -1 (using 0 as false, 1 as true? No, standard DPLL uses indices)
    # Let's use 0..n-1 for variables.
    # Literal l: if l > 0, var = l-1, pos = True. If l < 0, var = -l-1, pos = False.
    # We will store clauses as lists of (var, is_pos)
    
    processed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            var = abs(l) - 1
            pos = l > 0
            pc.append((var, pos))
        processed_clauses.append(pc)
    
    # DPLL Solver
    assignment = [None] * n  # None, True, False
    
    def unit_propagation(clauses_list):
        """
        Applies unit propagation.
        Returns (new_clauses, assignment_updated, sat_or_unsat)
        sat_or_unsat: True if SAT (all clauses satisfied/empty), False if UNSAT (empty clause), None if continue
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses_list:
                # Filter out satisfied literals
                unsat_literals = []
                satisfied = False
                for var, pos in clause:
                    if assignment[var] is not None:
                        if assignment[var] == pos:
                            satisfied = True
                            break
                    else:
                        unsat_literals.append((var, pos))
                
                if satisfied:
                    continue # Clause is satisfied, drop it
                
                if not unsat_literals:
                    return [], assignment, False # UNSAT: empty clause
                
                if len(unsat_literals) == 1:
                    # Unit clause
                    var, pos = unsat_literals[0]
                    if assignment[var] is not None:
                        if assignment[var] != pos:
                            return [], assignment, False # Conflict
                    else:
                        assignment[var] = pos
                        changed = True
                    # The clause becomes satisfied, so we don't add it to new_clauses
                    # Wait, if we assign it, the clause is satisfied.
                    # So we don't carry it over.
                else:
                    new_clauses.append(unsat_literals)
            
            clauses_list = new_clauses
            
            if not clauses_list:
                return [], assignment, True # SAT: no clauses left
        
        return clauses_list, assignment, None

    def pure_literal_elimination(clauses_list):
        """
        Identifies pure literals and assigns them.
        Returns (new_clauses, assignment_updated, sat_or_unsat)
        """
        if not clauses_list:
            return [], assignment, True
        
        # Count occurrences of each literal
        # var: (count_pos, count_neg)
        counts = {i: [0, 0] for i in range(n)}
        
        for clause in clauses_list:
            seen_vars = set()
            for var, pos in clause:
                if var in seen_vars: continue
                seen_vars.add(var)
                if pos:
                    counts[var][0] += 1
                else:
                    counts[var][1] += 1
        
        changed = True
        while changed:
            changed = False
            for var in range(n):
                if assignment[var] is not None:
                    continue
                pos_count, neg_count = counts[var]
                if pos_count > 0 and neg_count == 0:
                    # Pure positive
                    assignment[var] = True
                    changed = True
                elif neg_count > 0 and pos_count == 0:
                    # Pure negative
                    assignment[var] = False
                    changed = True
            
            if changed:
                # Re-filter clauses
                new_clauses = []
                for clause in clauses_list:
                    unsat_literals = []
                    satisfied = False
                    for var, pos in clause:
                        if assignment[var] is not None:
                            if assignment[var] == pos:
                                satisfied = True
                                break
                        else:
                            unsat_literals.append((var, pos))
                    if not satisfied:
                        if not unsat_literals:
                            return [], assignment, False
                        new_clauses.append(unsat_literals)
                clauses_list = new_clauses
                if not clauses_list:
                    return [], assignment, True
                # Recount? Or just return and let unit prop handle the rest?
                # Usually pure literal elimination is done once, then unit prop.
                # Let's just do one pass of pure literals then return to main loop.
                break
        
        return clauses_list, assignment, None

    def select_variable(clauses_list):
        """
        Select a variable to branch on.
        Heuristic: Most Frequent Variable in remaining clauses.
        """
        if not clauses_list:
            return -1
        
        freq = {i: 0 for i in range(n)}
        for clause in clauses_list:
            seen = set()
            for var, _ in clause:
                if var not in seen:
                    seen.add(var)
                    freq[var] += 1
        
        max_freq = -1
        best_var = -1
        for var in range(n):
            if assignment[var] is None and freq[var] > max_freq:
                max_freq = freq[var]
                best_var = var
        
        return best_var

    def dpll(clauses_list):
        """
        Recursive DPLL.
        Returns True if SAT, False if UNSAT.
        """
        # 1. Unit Propagation
        clauses_list, assignment, result = unit_propagation(clauses_list)
        if result is not None:
            return result
        
        # 2. Pure Literal Elimination
        clauses_list, assignment, result = pure_literal_elimination(clauses_list)
        if result is not None:
            return result
        
        # 3. Check if all clauses satisfied
        if not clauses_list:
            return True
        
        # 4. Select Variable
        var = select_variable(clauses_list)
        if var == -1:
            # No unassigned variables? Should have been caught by unit prop or pure lit?
            # If there are clauses but no unassigned vars, it means we have a contradiction?
            # Actually, if there are clauses, they must contain unassigned vars.
            # Unless we have a clause like (x) and x is assigned False? That would be UNSAT in unit prop.
            # So this shouldn't happen if unit prop is correct.
            return False

        # 5. Branch
        # Try True
        assignment[var] = True
        # Deep copy clauses? No, we filter them in unit_prop.
        # But unit_prop modifies clauses_list in place? No, it returns new list.
        # However, we need to branch on the *current* state of clauses.
        # The clauses_list passed to unit_prop is the one from the caller.
        # We need to branch on the clauses *after* unit prop and pure lit?
        # Standard DPLL:
        # 1. Simplify (Unit/Pure)
        # 2. If SAT/UNSAT return
        # 3. Branch
        # The clauses_list here is the simplified one.
        
        # We need to pass the simplified clauses to the recursive calls.
        # But wait, unit_propagation and pure_literal_elimination might have modified `clauses_list` variable in the local scope, but they return the new list.
        # We assigned the return value to `clauses_list`. So `clauses_list` is the simplified version.
        
        if dpll(clauses_list):
            return True
        
        # Backtrack
        assignment[var] = False
        # Re-simplify for the False branch?
        # The clauses_list is the same set of clauses (unsatisfied ones).
        # But the assignment is different.
        # We should pass the *same* simplified clauses_list?
        # No, the simplification depends on the assignment.
        # But the clauses that were dropped were satisfied by the *previous* assignment (True).
        # If we flip to False, those clauses might become unsatisfied again?
        # No, if a clause was satisfied by var=True, and we set var=False, it might no longer be satisfied.
        # So we CANNOT just reuse the simplified clauses_list from the True branch for the False branch.
        # We must branch on the clauses *before* the simplification that depended on the branch choice?
        # Or, we must re-simplify from the original set of clauses at this node?
        
        # Standard implementation:
        # def dpll(clauses):
        #   simplified_clauses = simplify(clauses)
        #   if sat(simplified_clauses): return True
        #   if unsat(simplified_clauses): return False
        #   var = select(simplified_clauses)
        #   if dpll(simplified_clauses with var=True): return True
        #   if dpll(simplified_clauses with var=False): return False
        #   return False
        
        # My current code:
        # clauses_list, assignment, result = unit_propagation(clauses_list)
        # ...
        # if dpll(clauses_list): # This passes the simplified clauses.
        #   ...
        # assignment[var] = False
        # if dpll(clauses_list): # This ALSO passes the simplified clauses.
        
        # This is WRONG.
        # If I set var=True, I drop clauses satisfied by var=True.
        # If I set var=False, those clauses might not be satisfied.
        # I need to pass the clauses *before* the unit propagation that was triggered by the branch?
        # No, unit propagation is triggered by the *current* assignment.
        
        # Correct approach:
        # 1. Simplify with current assignment.
        # 2. If SAT/UNSAT, return.
        # 3. Branch on var.
        # 4. Set var=True. Recurse on the *simplified* clauses?
        #    Wait, if I set var=True, I should simplify *again*?
        #    The simplified clauses from step 1 did NOT include var=True assignment.
        #    So I should pass the simplified clauses to the recursive call, and the recursive call will do unit propagation again with the new assignment.
        #    BUT, the simplified clauses from step 1 might have dropped clauses that are now relevant?
        #    No, step 1 dropped clauses satisfied by *previous* assignments.
        #    The new assignment var=True might satisfy MORE clauses.
        #    It will NOT unsatisfy any clause that was already satisfied.
        #    So passing the simplified clauses is safe?
        #    Yes, because any clause dropped in step 1 is satisfied by the *current* assignment (which includes var=True in the next step).
        #    So they remain satisfied.
        
        # So:
        # assignment[var] = True
        if dpll(clauses_list):
            return True
        
        assignment[var] = False
        # The clauses_list is still the one from step 1.
        # Is it safe to pass it again?
        # Yes, because the clauses in `clauses_list` are the ones NOT satisfied by the assignment *before* setting var.
        # Now var is False.
        # The clauses in `clauses_list` might contain var with polarity True (which is now False) or False (which is now True).
        # The recursive call will handle the unit propagation with var=False.
        
        if dpll(clauses_list):
            return True
        
        assignment[var] = None
        return False

    # Start DPLL
    # Initial clauses
    sat = dpll(processed_clauses)
    
    if sat:
        # Fill in any unassigned variables (should be none if DPLL is complete, but just in case)
        final_assignment = []
        for i in range(n):
            if assignment[i] is None:
                final_assignment.append(False) # Arbitrary, but shouldn't happen
            else:
                final_assignment.append(assignment[i])
        return final_assignment
    else:
        # Should not happen for planted SAT instances, but return a default
        return [False] * n
# EVOLVE-BLOCK-END
