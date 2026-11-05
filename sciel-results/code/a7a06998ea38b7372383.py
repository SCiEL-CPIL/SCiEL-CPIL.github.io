import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-processing for efficiency
    # 1. Map variables to the indices of clauses they appear in
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l) - 1].append(i)
            
    # 2. For each clause, store the literals and a quick way to check satisfaction
    # We will maintain:
    # - assignment: list of booleans (0/1 or False/True)
    # - clause_sat: list of booleans, True if clause is satisfied
    # - unsat_count: integer count of unsatisfied clauses
    
    def is_lit_sat(lit, assign):
        # lit is 1-based signed, assign is 0-based boolean list
        var = abs(lit) - 1
        val = assign[var]
        return (lit > 0 and val) or (lit < 0 and not val)

    def get_clause_sat_status(clause, assign):
        return any(is_lit_sat(l, assign) for l in clause)

    # Initial random assignment
    rng = random.Random(42)
    
    # Try multiple restarts to ensure robustness
    max_restarts = 50
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_assign = None
    best_unsat = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize assignment
        # For planted SAT, the planted assignment is often biased. 
        # Standard WalkSAT uses 0.5. We can try slight biases or just random.
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause status
        clause_sat = [False] * len(clauses)
        unsat_count = 0
        for i, c in enumerate(clauses):
            if get_clause_sat_status(c, assignment):
                clause_sat[i] = True
            else:
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assignment[:]
            
        # WalkSAT Loop
        # Parameters:
        # - walk_prob: probability of taking a random step (walk) vs greedy step
        # - max_steps: limit steps per restart before restarting
        #   Adaptive: increase limit if we are making progress
        
        walk_prob = 0.3
        max_steps = 5000 + restart * 1000
        step = 0
        no_improve_steps = 0
        last_unsat = unsat_count
        
        while step < max_steps:
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            # Select an unsatisfied clause
            # Random selection is standard, but we can also pick one with more "false" literals
            # For simplicity and speed, random pick from unsat list is usually fine.
            # To avoid building a list every time, we can pick a random clause and check if unsat,
            # but that's inefficient if most are sat. 
            # Since n=150, m is likely ~420 (ratio 2.8) or similar. 
            # Building list of unsat indices is O(m). m is small.
            unsat_indices = [i for i in range(len(clauses)) if not clause_sat[i]]
            if not unsat_indices:
                return assignment
                
            c_idx = rng.choice(unsat_indices)
            c = clauses[c_idx]
            
            # Select variable to flip
            # Greedy: find var in c that minimizes unsat_count after flip
            # Walk: pick random var in c
            
            if rng.random() < walk_prob:
                # Walk step
                lit = rng.choice(c)
                var_idx = abs(lit) - 1
            else:
                # Greedy step
                best_gain = float('inf')
                best_var = None
                
                for lit in c:
                    var_idx = abs(lit) - 1
                    
                    # Calculate gain (change in unsat_count) if we flip var_idx
                    # Gain = new_unsat_count - old_unsat_count
                    # We want to minimize gain (most negative is best)
                    
                    # Current state of var_idx
                    current_val = assignment[var_idx]
                    new_val = not current_val
                    
                    gain = 0
                    # Iterate through clauses containing var_idx
                    for ci in var_to_clauses[var_idx]:
                        cc = clauses[ci]
                        is_sat_now = clause_sat[ci]
                        
                        # Check if flipping var_idx changes the satisfaction of cc
                        # Find the literal for var_idx in cc
                        lit_in_cc = None
                        for l in cc:
                            if abs(l) - 1 == var_idx:
                                lit_in_cc = l
                                break
                            
                        if lit_in_cc is None:
                            continue # Should not happen if data structure is correct
                        
                        # Is this literal satisfied by the NEW value?
                        new_lit_sat = (lit_in_cc > 0 and new_val) or (lit_in_cc < 0 and not new_val)
                        
                        if is_sat_now:
                            # Clause is currently satisfied
                            # It becomes unsatisfied ONLY IF:
                            # 1. The literal for var_idx was satisfied by the OLD value
                            # 2. The literal for var_idx is NOT satisfied by the NEW value
                            # 3. No OTHER literal in the clause is satisfied by the OLD value (i.e., var_idx was the ONLY satisfied lit)
                            
                            old_lit_sat = (lit_in_cc > 0 and current_val) or (lit_in_cc < 0 and not current_val)
                            
                            if old_lit_sat and not new_lit_sat:
                                # Check if any other lit in cc is satisfied
                                others_sat = False
                                for l in cc:
                                    if abs(l) - 1 != var_idx:
                                        if is_lit_sat(l, assignment):
                                            others_sat = True
                                            break
                                if not others_sat:
                                    # This clause will become unsatisfied
                                    gain += 1
                        else:
                            # Clause is currently unsatisfied
                            # It becomes satisfied ONLY IF:
                            # The literal for var_idx is satisfied by the NEW value
                            if new_lit_sat:
                                gain -= 1
                                
                    if gain < best_gain:
                        best_gain = gain
                        best_var = var_idx
                        
                if best_var is not None:
                    var_idx = best_var
                else:
                    # Fallback to random if something went wrong
                    lit = rng.choice(c)
                    var_idx = abs(lit) - 1
            
            # Perform the flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause statuses efficiently
            # Only clauses containing var_idx can change status
            for ci in var_to_clauses[var_idx]:
                cc = clauses[ci]
                old_sat = clause_sat[ci]
                new_sat = get_clause_sat_status(cc, assignment)
                
                if old_sat and not new_sat:
                    unsat_count += 1
                elif not old_sat and new_sat:
                    unsat_count -= 1
                    
                clause_sat[ci] = new_sat
                
            step += 1
            
            # Track progress
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve_steps = 0
                # Adjust walk probability: if we are improving, be more greedy?
                # Or keep it constant. Standard WalkSAT often uses 0.1-0.5.
            else:
                no_improve_steps += 1
                
            # Restart condition within the inner loop?
            # If stuck for too long, break out to next restart
            if no_improve_steps > 2000:
                break
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment[:]
                
        # End of inner loop for this restart
        
    return best_assign
