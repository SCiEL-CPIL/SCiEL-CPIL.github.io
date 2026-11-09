import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    # Deadline: 325 seconds to be safe against 330s kill
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Parameters
    max_steps_per_run = 10000
    noise = 0.2 # Probability to pick random literal if no improvement
    
    # Pre-process clauses for faster access
    # clause_sat_count[i] = number of literals in clause i that are currently satisfied
    clause_sat_count = [0] * len(clauses)
    
    # var_occurrences[v] = list of (clause_index, sign) where sign is +1 if var is positive in clause, -1 if negative
    # Actually, let's store the literal value or just the index and we can check sign from clause.
    # Better: var_occurrences[v] = list of clause indices
    var_occurrences = [[] for _ in range(n)]
    
    # Initialize
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Build index and initial counts
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_occurrences[v].append(i)
            # Check if this literal is satisfied in initial assignment
            if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                clause_sat_count[i] += 1
                
    # Helper to check if all satisfied
    def is_satisfied():
        for count in clause_sat_count:
            if count == 0:
                return False
        return True
        
    # Helper to get unsat clause indices
    def get_unsat_indices():
        return [i for i, count in enumerate(clause_sat_count) if count == 0]
        
    # Heuristic step
    # Returns True if solved, False otherwise
    def step(assignment, rng):
        unsat_indices = get_unsat_indices()
        if not unsat_indices:
            return True
        
        # Pick random unsat clause
        clause_idx = rng.choice(unsat_indices)
        clause = clauses[clause_idx]
        
        best_var_idx = None
        best_delta = float('inf')
        best_new_sat_counts = None
        
        # Evaluate flipping each variable in the clause
        for l in clause:
            v = abs(l) - 1
            # Calculate delta in clause_sat_count for all clauses containing v
            # If we flip v:
            # - For clauses where v is currently satisfied (literal matches assignment), count decreases by 1
            # - For clauses where v is currently unsatisfied (literal opposes assignment), count increases by 1
            
            delta_counts = {}
            # We need to know the current satisfaction status of v in each clause it appears in
            # Let's compute the net change in total unsat clauses or just track the specific clause counts
            
            # Optimization: We want to minimize the number of newly unsatisfied clauses.
            # Let's calculate the change in clause_sat_count for each clause containing v.
            
            current_sat = 0 # How many clauses containing v are currently satisfied by v?
            clauses_affected = var_occurrences[v]
            
            # Pre-calculate the change for each affected clause
            changes = {}
            for ci in clauses_affected:
                c = clauses[ci]
                # Determine if literal v in clause ci is currently satisfied
                # Find the literal in c that corresponds to v
                # Since we have var_occurrences, we know v is in c.
                # We can check the sign.
                lit_val = None
                for ll in c:
                    if abs(ll) - 1 == v:
                        lit_val = ll
                        break
                if lit_val is None:
                    continue # Should not happen
                
                is_satisfied_by_v = (lit_val > 0 and assignment[v]) or (lit_val < 0 and not assignment[v])
                
                if is_satisfied_by_v:
                    # Flipping v makes this clause lose one satisfied literal
                    changes[ci] = -1
                else:
                    # Flipping v makes this clause gain one satisfied literal
                    changes[ci] = 1
            
            # Now, calculate the "cost" of this flip.
            # We care about:
            # 1. The clause we picked (clause_idx) MUST become satisfied.
            #    Since it's unsat (count 0), and we are flipping a var in it,
            #    the literal for v in clause_idx must be currently unsatisfied (otherwise count would be >0? No, count is 0 means ALL literals unsatisfied).
            #    So for clause_idx, the change must be +1.
            
            # 2. We want to minimize the number of OTHER clauses that become unsatisfied (change from 1 to 0).
            #    A clause becomes unsatisfied if its count was 1 and we subtract 1.
            
            newly_unsat = 0
            newly_sat = 0
            
            for ci, delta in changes.items():
                old_count = clause_sat_count[ci]
                new_count = old_count + delta
                if old_count == 1 and new_count == 0:
                    newly_unsat += 1
                elif old_count == 0 and new_count == 1:
                    newly_sat += 1
            
            # We prioritize minimizing newly_unsat.
            # If ties, maybe maximize newly_sat? Or just random among best.
            
            if newly_unsat < best_delta:
                best_delta = newly_unsat
                best_var_idx = v
                # Store the changes to apply later
                best_changes = changes
                
            # Apply noise: with probability p, pick a random var from the clause instead
            # We can just do this after selection
        
        # Apply best flip
        if best_var_idx is not None:
            v = best_var_idx
            # Update clause_sat_count
            for ci, delta in best_changes.items():
                clause_sat_count[ci] += delta
            assignment[v] = not assignment[v]
        else:
            # Fallback: random flip from clause
            l = rng.choice(clause)
            v = abs(l) - 1
            # Update clause_sat_count
            for ci in var_occurrences[v]:
                c = clauses[ci]
                lit_val = None
                for ll in c:
                    if abs(ll) - 1 == v:
                        lit_val = ll
                        break
                if lit_val is None:
                    continue
                is_satisfied_by_v = (lit_val > 0 and assignment[v]) or (lit_val < 0 and not assignment[v])
                if is_satisfied_by_v:
                    clause_sat_count[ci] -= 1
                else:
                    clause_sat_count[ci] += 1
            assignment[v] = not assignment[v]
            
        # Noise: with probability noise, flip a random variable from the clause instead of the best one?
        # The standard WalkSAT often picks a random literal from the clause with prob p, else best.
        # My implementation above picks best. Let's add the noise logic properly.
        
        return False

    # Main Loop
    while time.time() < deadline:
        # Reset or continue? Standard WalkSAT restarts often.
        # Let's restart with a new random assignment if we hit max_steps.
        
        if not is_satisfied():
             # Initial check
             pass
             
        steps = 0
        while steps < max_steps_per_run:
            if time.time() >= deadline:
                return assignment
            
            if is_satisfied():
                return assignment
            
            step(assignment, rng)
            steps += 1
            
        # Restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        # Re-initialize clause_sat_count
        for i in range(len(clauses)):
            clause_sat_count[i] = 0
        for i, c in enumerate(clauses):
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    clause_sat_count[i] += 1

    return assignment
