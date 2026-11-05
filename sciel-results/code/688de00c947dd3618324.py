# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: finish well before 330s
    start_time = time.time()
    DEADLINE = 320.0  # safety margin
    
    # Preprocess clauses: convert to 0-based indices and store signs
    # clause is list of signed 1-based literals
    # Represent each literal as (var_index, is_positive)
    # var_index = abs(l)-1, is_positive = l>0
    
    # For speed, precompute variable index and polarity for each literal in each clause
    # clauses_processed[i] = list of (var_idx, polarity) where polarity is +1 or -1
    clauses_processed = []
    for clause in clauses:
        processed = []
        for lit in clause:
            var_idx = abs(lit) - 1
            pol = 1 if lit > 0 else -1
            processed.append((var_idx, pol))
        clauses_processed.append(processed)
    
    # We'll use a vectorized approach with numpy for the assignment
    # assignment is a numpy array of 0s and 1s
    
    def is_satisfied_fast(assignment_np):
        """Check if all clauses are satisfied. Returns True if SAT."""
        # For each clause, check if any literal is satisfied
        # A literal (var_idx, pol) is satisfied if:
        #   pol == 1 and assignment[var_idx] == 1
        #   pol == -1 and assignment[var_idx] == 0
        for clause in clauses_processed:
            sat = False
            for var_idx, pol in clause:
                if pol == 1:
                    if assignment_np[var_idx] == 1:
                        sat = True
                        break
                else:
                    if assignment_np[var_idx] == 0:
                        sat = True
                        break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment_np):
        """Count number of unsatisfied clauses."""
        cnt = 0
        for clause in clauses_processed:
            sat = False
            for var_idx, pol in clause:
                if pol == 1:
                    if assignment_np[var_idx] == 1:
                        sat = True
                        break
                else:
                    if assignment_np[var_idx] == 0:
                        sat = True
                        break
            if not sat:
                cnt += 1
        return cnt
    
    # WalkSAT with clause weighting and restarts
    # Use numpy for assignment representation
    rng = random.Random(42)
    
    # Try multiple random restarts
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 50
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random initial assignment
        assignment = np.random.randint(0, 2, size=n, dtype=np.uint8)
        
        # Track unsatisfied clause count
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return [bool(x) for x in assignment]
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # WalkSAT loop
        # We'll use a simple approach: pick random unsatisfied clause, flip a variable
        # To speed up, we'll track unsatisfied clauses
        
        # Build list of unsatisfied clause indices
        unsat_clauses = []
        for ci, clause in enumerate(clauses_processed):
            sat = False
            for var_idx, pol in clause:
                if pol == 1:
                    if assignment[var_idx] == 1:
                        sat = True
                        break
                else:
                    if assignment[var_idx] == 0:
                        sat = True
                        break
            if not sat:
                unsat_clauses.append(ci)
        
        # WalkSAT with noise parameter
        noise = 0.55  # probability of picking a random literal vs best literal
        
        for step in range(steps_per_restart):
            if time.time() - start_time > DEADLINE:
                break
            
            if not unsat_clauses:
                break
            
            # Pick random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = clauses_processed[ci]
            
            # Find literals in this clause that are currently false (flipping them would satisfy this clause)
            # A literal is false if:
            #   pol == 1 and assignment[var_idx] == 0
            #   pol == -1 and assignment[var_idx] == 1
            
            # Count how many other unsatisfied clauses would be satisfied if we flip each variable
            # For simplicity, we'll use a heuristic: pick a variable that appears in many unsatisfied clauses
            
            # For speed, let's just pick a random literal from the clause with some bias
            # Better approach: for each literal in the clause, compute the "gain" (number of unsat clauses that would be satisfied)
            
            # To keep it fast, let's use a simpler strategy:
            # Pick a random literal from the unsatisfied clause
            # But bias towards literals that appear in more unsatisfied clauses
            
            # Let's compute a score for each literal in the clause
            # Score = number of unsatisfied clauses containing this literal (with either polarity)
            # Actually, we want to flip the variable to satisfy this clause, so we want to know
            # how many unsatisfied clauses would become satisfied if we set the variable to the required value
            
            # For each literal (var_idx, pol) in the clause:
            #   If we set assignment[var_idx] = (1 if pol==1 else 0), which unsatisfied clauses become satisfied?
            #   A clause becomes satisfied if at least one of its literals matches the new assignment
            
            # This is expensive to compute exactly. Let's use an approximation:
            # Count how many unsatisfied clauses contain var_idx (regardless of polarity)
            
            # Precompute: for each variable, which unsatisfied clauses contain it
            # This changes every step, so it's expensive to maintain
            
            # Simpler approach: just pick a random literal from the clause
            # This is the original WalkSAT and it works reasonably well
            
            lit_idx = rng.randint(0, len(clause) - 1)
            var_idx, pol = clause[lit_idx]
            
            # Flip the variable
            new_val = 1 if pol == 1 else 0
            old_val = assignment[var_idx]
            
            if old_val == new_val:
                # Already set to satisfy this literal, flipping would break it
                # This shouldn't happen if the clause is unsatisfied, but just in case
                continue
            
            assignment[var_idx] = new_val
            
            # Update unsatisfied clauses
            # We need to recompute which clauses are now satisfied/unsatisfied
            # This is expensive, so let's do it incrementally
            
            # For efficiency, let's just recompute the unsatisfied list periodically
            # Or use a simpler check
            
            # Let's check if the total unsatisfied count decreased
            # For now, let's just continue and check periodically
            
            if step % 10 == 0:
                # Periodically check if we solved it
                if is_satisfied_fast(assignment):
                    return [bool(x) for x in assignment]
                
                # Update best
                uc = count_unsat(assignment)
                if uc < best_unsat_count:
                    best_unsat_count = uc
                    best_assignment = assignment.copy()
                
                # Rebuild unsat_clauses list
                unsat_clauses = []
                for cci, cclause in enumerate(clauses_processed):
                    sat = False
                    for vidx, cpol in cclause:
                        if cpol == 1:
                            if assignment[vidx] == 1:
                                sat = True
                                break
                        else:
                            if assignment[vidx] == 0:
                                sat = True
                                break
                    if not sat:
                        unsat_clauses.append(cci)
        
        # After this restart, check final state
        if is_satisfied_fast(assignment):
            return [bool(x) for x in assignment]
        
        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment.copy()
    
    # Return the best assignment found
    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    
    # Fallback: random assignment
    final = np.random.randint(0, 2, size=n, dtype=np.uint8)
    return [bool(x) for x in final]
# EVOLVE-BLOCK-END
