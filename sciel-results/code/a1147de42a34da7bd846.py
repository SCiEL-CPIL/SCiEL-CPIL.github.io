import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 330s limit, use 290s to be safe
    start_time = time.time()
    deadline = start_time + 290.0
    
    m = len(clauses)
    
    # Preprocessing
    # var_clauses[v] = list of clause indices containing variable v
    var_clauses = [[] for _ in range(n)]
    
    # clause_vars[c] = list of (var_idx, sign) for clause c
    # sign: +1 for positive, -1 for negative
    clause_vars = []
    
    for i, clause in enumerate(clauses):
        vars_in_clause = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_in_clause.append((v, s))
            var_clauses[v].append(i)
        clause_vars.append(vars_in_clause)
        
    # State arrays
    # assignment: 1 for True, 0 for False
    # clause_unsat_count[c]: number of unsatisfied literals in clause c
    # var_unsat_count[v]: number of unsatisfied clauses containing variable v
    
    best_assignment = None
    best_unsat_total = float('inf')
    
    # Parameters
    # WalkSAT probability: 0.2 is a common good value
    p_walk = 0.2 
    # Max steps per restart
    max_steps_per_restart = 5000
    # Restart interval (in steps) if no improvement
    stall_limit = 100
    
    rng = random.Random(42)
    
    # We will try multiple restarts until deadline
    while time.time() < deadline:
        # Random initialization
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)
        
        # Initialize clause_unsat_count
        clause_unsat_count = np.zeros(m, dtype=np.int32)
        var_unsat_count = np.zeros(n, dtype=np.int32)
        
        # Calculate initial unsat counts
        for c in range(m):
            unsat_in_clause = 0
            for v, s in clause_vars[c]:
                # Literal is satisfied if assignment[v] == (s == 1)
                # i.e., if s==1 and assignment[v]==1, or s==-1 and assignment[v]==0
                # Unsatisfied if (s == 1 and assignment[v] == 0) or (s == -1 and assignment[v] == 1)
                # This is equivalent to: assignment[v] != (s == 1)
                if assignment[v] != (s == 1):
                    unsat_in_clause += 1
            clause_unsat_count[c] = unsat_in_clause
            
            if unsat_in_clause > 0:
                for v, s in clause_vars[c]:
                    var_unsat_count[v] += 1
        
        total_unsat = np.sum(clause_unsat_count > 0) # Number of unsatisfied clauses
        
        if total_unsat == 0:
            return [bool(x) for x in assignment]
            
        if total_unsat < best_unsat_total:
            best_unsat_total = total_unsat
            best_assignment = assignment.copy()
            
        steps_since_improvement = 0
        min_unsat_in_restart = total_unsat
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
                
            if total_unsat == 0:
                return [bool(x) for x in assignment]
                
            if total_unsat < best_unsat_total:
                best_unsat_total = total_unsat
                best_assignment = assignment.copy()
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            if steps_since_improvement > stall_limit:
                break # Restart
                
            # Select an unsatisfied clause
            unsat_clauses = np.where(clause_unsat_count > 0)[0]
            if len(unsat_clauses) == 0:
                return [bool(x) for x in assignment]
                
            c_idx = rng.choice(unsat_clauses)
            
            # Select variable to flip
            # Get variables in this clause
            vars_in_c = clause_vars[c_idx]
            
            if rng.random() < p_walk:
                # Random flip
                v, s = rng.choice(vars_in_c)
            else:
                # DLS: Pick variable with max var_unsat_count
                # Note: var_unsat_count is global. We pick the one in this clause with the highest count.
                max_count = -1
                best_v = -1
                best_s = 0
                for v, s in vars_in_c:
                    if var_unsat_count[v] > max_count:
                        max_count = var_unsat_count[v]
                        best_v = v
                        best_s = s
                v, s = best_v, best_s
                
            # Perform Flip
            old_val = assignment[v]
            new_val = 1 - old_val
            assignment[v] = new_val
            
            # Update states
            # We need to update clause_unsat_count for all clauses containing v
            # And var_unsat_count for all variables in those clauses
            
            # Optimization: We can just iterate over var_clauses[v]
            for c in var_clauses[v]:
                # Determine if the clause was satisfied before and after
                # Actually, we can just re-evaluate the unsat count for this specific clause?
                # No, that's O(3) which is fine, but we also need to update var_unsat_count for other vars in the clause.
                
                # Let's calculate the change in unsat literals for clause c due to flipping v
                # The literal of v in clause c has sign s_c (we need to find it)
                # Wait, clause_vars[c] has (v, s). We know v is in clause c.
                # We need to find the sign of v in clause c.
                
                # This is inefficient if we search. Let's precompute lit_sign_in_clause[c][v]?
                # Or just recompute the unsat count for clause c from scratch?
                # m is likely ~600 (4.2 * 150). Clause size 3.
                # Recalculating unsat count for one clause is 3 ops.
                # Updating var_unsat_count requires knowing if the clause changed from sat to unsat or vice versa.
                
                # Let's do it cleanly:
                # 1. Calculate old_unsat_c
                # 2. Flip v
                # 3. Calculate new_unsat_c
                # 4. If old_unsat_c == 0 and new_unsat_c > 0: Clause became unsat. Increment var_unsat_count for all vars in c.
                # 5. If old_unsat_c > 0 and new_unsat_c == 0: Clause became sat. Decrement var_unsat_count for all vars in c.
                # 6. If both > 0 or both 0: No change in var_unsat_count for other vars.
                #    BUT, v's var_unsat_count changes if the clause's status changes.
                #    Actually, var_unsat_count[v] is the count of unsat clauses containing v.
                #    So if clause c was unsat and becomes sat, var_unsat_count[v] decreases by 1.
                #    If clause c was sat and becomes unsat, var_unsat_count[v] increases by 1.
                #    The other vars in c also change their var_unsat_count by +/- 1.
                
                # To avoid searching for sign, let's just recompute unsat count for clause c.
                old_unsat = clause_unsat_count[c]
                
                # Recompute unsat count for clause c with new assignment
                new_unsat = 0
                for vv, ss in clause_vars[c]:
                    if assignment[vv] != (ss == 1):
                        new_unsat += 1
                clause_unsat_count[c] = new_unsat
                
                if old_unsat == 0 and new_unsat > 0:
                    # Clause became unsatisfied
                    # Increment var_unsat_count for all vars in this clause
                    for vv, ss in clause_vars[c]:
                        var_unsat_count[vv] += 1
                    total_unsat += 1
                elif old_unsat > 0 and new_unsat == 0:
                    # Clause became satisfied
                    # Decrement var_unsat_count for all vars in this clause
                    for vv, ss in clause_vars[c]:
                        var_unsat_count[vv] -= 1
                    total_unsat -= 1
                # else: status didn't change (0->0 or >0->>0), so var_unsat_count for other vars unchanged.
                # Note: v's var_unsat_count is updated in the loops above.
                
            # End update loop for var_clauses[v]
            
            if total_unsat < min_unsat_in_restart:
                min_unsat_in_restart = total_unsat
                
        # End step loop
        # If we didn't solve, we restart (loop continues)
        
    # Return best assignment found
    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    else:
        # Fallback: random assignment
        return [bool(x) for x in assignment]
