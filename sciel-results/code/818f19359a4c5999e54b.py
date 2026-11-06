"""
Bit-vector accelerated WalkSAT solver for 3-SAT (50 variables).
Uses bitmask representation for O(1) satisfied-clause counting and
greedy flip selection to maximize net gain in satisfied clauses.
"""
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s limit, leave 15s margin for safety
    deadline = time.time() + 285.0
    
    # Precompute bitmasks for each variable's positive and negative occurrences
    # pos_mask[var] has bit i set if clause i contains literal (var+1) [positive]
    # neg_mask[var] has bit i set if clause i contains literal -(var+1) [negative]
    pos_mask = [0] * n
    neg_mask = [0] * n
    
    # Also store which variables are in each clause for quick iteration
    clause_vars = []  # list of lists of variable indices (0-based)
    
    for i, clause in enumerate(clauses):
        bit = 1 << i
        vars_in_clause = []
        for lit in clause:
            var = abs(lit) - 1
            if lit > 0:
                pos_mask[var] |= bit
            else:
                neg_mask[var] |= bit
            if var not in vars_in_clause:
                vars_in_clause.append(var)
        clause_vars.append(vars_in_clause)
    
    # For each variable, precompute the list of clauses it appears in
    # This helps with the greedy selection
    var_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            if i not in var_clauses[var]:
                var_clauses[var].append(i)
    
    def count_satisfied(assignment):
        """Count number of satisfied clauses using bitmask operations."""
        sat_mask = 0
        for var in range(n):
            if assignment[var]:
                sat_mask |= pos_mask[var]
            else:
                sat_mask |= neg_mask[var]
        return bin(sat_mask).count('1')
    
    def get_unsat_clause_indices(assignment):
        """Get list of indices of unsatisfied clauses."""
        sat_mask = 0
        for var in range(n):
            if assignment[var]:
                sat_mask |= pos_mask[var]
            else:
                sat_mask |= neg_mask[var]
        unsat = []
        for i in range(m):
            if not (sat_mask & (1 << i)):
                unsat.append(i)
        return unsat
    
    def evaluate_flip(var, assignment, sat_mask):
        """
        Evaluate the net gain in satisfied clauses if we flip variable `var`.
        Returns (gain, new_sat_mask) where gain = new_satisfied - old_satisfied.
        """
        # Current contribution of this variable
        if assignment[var]:
            # Currently True, so positive literals are satisfied
            # Flipping to False: lose pos_mask[var] contributions, gain neg_mask[var] contributions
            # But we need to account for clauses that are satisfied by OTHER variables too
            old_contrib = pos_mask[var]
            new_contrib = neg_mask[var]
        else:
            # Currently False, so negative literals are satisfied
            old_contrib = neg_mask[var]
            new_contrib = pos_mask[var]
        
        # Clauses that become unsatisfied: those in old_contrib but NOT in new_contrib AND not satisfied by other vars
        # Clauses that become satisfied: those in new_contrib but NOT in old_contrib AND not satisfied by other vars
        
        # A clause i is currently satisfied if sat_mask has bit i set
        # After flip, clause i is satisfied if:
        #   - it was satisfied by another variable (i.e., (sat_mask & ~(old_contrib | new_contrib)) has bit i) 
        #     OR it is in new_contrib
        # Wait, let's think more carefully.
        
        # sat_mask = OR of all contributions
        # Let other_mask = sat_mask & ~old_contrib (clauses satisfied by variables other than `var`, 
        #   but this is not quite right because a clause might be satisfied by `var` AND another var)
        
        # Better approach:
        # A clause i is satisfied iff (sat_mask >> i) & 1 == 1
        # After flipping var, the new sat_mask' = sat_mask ^ (old_contrib & ~new_contrib) ^ (new_contrib & ~old_contrib)
        # No, that's not right either because of overlaps.
        
        # Let's use: new_sat_mask = (sat_mask & ~(old_contrib & ~new_contrib)) | (new_contrib & ~old_contrib)
        # Wait, that's still not right.
        
        # Actually: 
        # sat_mask = OR over all vars of their contribution
        # Let's define:
        #   contrib_before = old_contrib  (what `var` contributes currently)
        #   contrib_after = new_contrib   (what `var` would contribute after flip)
        #   
        # A clause i is satisfied before iff (sat_mask >> i) & 1
        # A clause i is satisfied after iff:
        #   - (sat_mask without var's contribution) has bit i set, OR
        #   - contrib_after has bit i set
        #
        # sat_mask_without_var = sat_mask & ~contrib_before
        # But this is only correct if no other variable also satisfies that clause in a way that's "masked"
        # Actually, sat_mask = (OR of all other vars' contributions) | contrib_before
        # So OR of all other vars' contributions = sat_mask & ~contrib_before ... NO!
        # If contrib_before has bit i, and another var also has bit i, then sat_mask has bit i,
        # and sat_mask & ~contrib_before will NOT have bit i, even though another var satisfies it.
        
        # Correct approach:
        # other_mask = 0
        # for v in range(n):
        #     if v != var:
        #         if assignment[v]:
        #             other_mask |= pos_mask[v]
        #         else:
        #             other_mask |= neg_mask[v]
        # This is O(n) per evaluation, which is too slow for greedy over 3 vars.
        
        # Alternative: precompute for each variable, the "exclusive" contribution?
        # Or: use the fact that we only flip one variable at a time.
        
        # Let's just compute it directly but efficiently:
        # new_sat_mask = (sat_mask & ~old_contrib) | new_contrib
        # This is correct because:
        # - sat_mask & ~old_contrib gives us clauses satisfied by variables other than `var` 
        #   (assuming no other variable's contribution is "hidden" behind `var`'s contribution)
        # - Wait, if clause i is satisfied by both `var` and another var, then:
        #   - sat_mask has bit i
        #   - old_contrib has bit i
        #   - sat_mask & ~old_contrib does NOT have bit i
        #   - But another var DOES satisfy clause i
        #   - So new_sat_mask would incorrectly say clause i is not satisfied by other vars
        #   - But if new_contrib also has bit i, then new_sat_mask gets bit i from new_contrib
        #   - If new_contrib does NOT have bit i, then new_sat_mask loses bit i, which is WRONG
        #     because another var still satisfies it.
        
        # So this approach is flawed. We need to properly compute other_mask.
        
        # For 50 variables, computing other_mask is O(n) = O(50) bitwise ORs.
        # Then for each of the 3 literals in the clause, we compute the gain.
        # Total per step: 3 * 50 = 150 bitwise ORs, which is fast.
        
        other_mask = 0
        for v in range(n):
            if v != var:
                if assignment[v]:
                    other_mask |= pos_mask[v]
                else:
                    other_mask |= neg_mask[v]
        
        new_sat_mask = other_mask | new_contrib
        old_count = bin(sat_mask).count('1')
        new_count = bin(new_sat_mask).count('1')
        return new_count - old_count, new_sat_mask
    
    # WalkSAT with restarts
    best_assignment = None
    best_unsat_count = m
    
    max_restarts = 100
    steps_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 7919 + 12345)  # Different seed for each restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial evaluation
        sat_mask = 0
        for var in range(n):
            if assignment[var]:
                sat_mask |= pos_mask[var]
            else:
                sat_mask |= neg_mask[var]
        
        unsat_count = m - bin(sat_mask).count('1')
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        if best_unsat_count == 0:
            return best_assignment
        
        # Walk phase
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Get unsatisfied clauses
            unsat_indices = []
            for i in range(m):
                if not (sat_mask & (1 << i)):
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return assignment
            
            current_unsat_count = len(unsat_indices)
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_indices)
            clause = clauses[clause_idx]
            
            # Greedy: try each literal in the clause, pick the one with max net gain
            p_random = 0.15  # Probability of random flip to escape local minima
            if rng.random() < p_random:
                lit = rng.choice(clause)
                var = abs(lit) - 1
            else:
                best_var = None
                best_gain = -1
                best_new_sat_mask = None
                
                for lit in clause:
                    var = abs(lit) - 1
                    gain, new_sat_mask = evaluate_flip(var, assignment, sat_mask)
                    if gain > best_gain or (gain == best_gain and best_var is None):
                        best_gain = gain
                        best_var = var
                        best_new_sat_mask = new_sat_mask
                
                var = best_var
                sat_mask = best_new_sat_mask
                continue  # Skip the flip code below
            
            # Perform the flip
            var = abs(lit) - 1
            assignment[var] = not assignment[var]
            
            # Update sat_mask
            if assignment[var]:
                sat_mask |= pos_mask[var]
            else:
                # Need to recompute properly
                # Remove old contribution and add new
                # This is tricky with bitmasks. Let's just recompute.
                pass
            
            # Recompute sat_mask for correctness
            sat_mask = 0
            for v in range(n):
                if assignment[v]:
                    sat_mask |= pos_mask[v]
                else:
                    sat_mask |= neg_mask[v]
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
