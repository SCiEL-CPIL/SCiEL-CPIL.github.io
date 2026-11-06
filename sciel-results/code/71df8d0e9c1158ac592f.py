# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Precompute variable to clause mapping for efficient gain calculation
    # Also precompute the literal sign for each occurrence to avoid searching in inner loop
    var_to_clauses = [[] for _ in range(n)]
    # clause_var_info[c_idx] will be a list of (var_idx, sign) where sign is 1 for +var, -1 for -var
    clause_var_info = []
    
    for c_idx, clause in enumerate(clauses):
        info = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            info.append((v, s))
            var_to_clauses[v].append(c_idx)
        clause_var_info.append(info)

    # Deadline: 325 seconds to be safe against the 330s kill
    deadline = time.time() + 325.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    noise = 0.5  # Probability of picking a random variable in the clause
    restart_threshold = 5000 # Restart if no improvement for this many steps
    
    while time.time() < deadline:
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial clause weights (number of satisfied literals)
        # weight > 0 means satisfied
        clause_weights = [0] * len(clauses)
        for i, clause_info in enumerate(clause_var_info):
            w = 0
            for v, s in clause_info:
                if assignment[v] == (s == 1):
                    w += 1
            clause_weights[i] = w
        
        total_unsat = sum(1 for w in clause_weights if w == 0)
        
        if total_unsat == 0:
            return assignment
        
        steps_since_improvement = 0
        min_unsat_seen = total_unsat
        
        for _ in range(100000): # Inner loop limit per restart
            if time.time() >= deadline:
                break
            
            if total_unsat == 0:
                return assignment
            
            # Track best solution
            if total_unsat < best_unsat_count:
                best_unsat_count = total_unsat
                best_assignment = assignment[:]
            
            if total_unsat < min_unsat_seen:
                min_unsat_seen = total_unsat
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            if steps_since_improvement > restart_threshold:
                break # Restart with new random assignment
            
            # Find an unsatisfied clause
            unsat_indices = [i for i, w in enumerate(clause_weights) if w == 0]
            if not unsat_indices:
                return assignment
                
            clause_idx = rng.choice(unsat_indices)
            clause_info = clause_var_info[clause_idx]
            
            # Heuristic: Pick variable that minimizes new unsat clauses
            # Or with probability 'noise', pick random variable in clause
            
            if rng.random() < noise:
                # Random walk step
                v, s = rng.choice(clause_info)
                var_idx = v
            else:
                # Greedy step: Find variable in clause that gives best gain
                best_gain = float('inf')
                best_var_idx = -1
                
                for v, s in clause_info:
                    var_idx = v
                    current_val = assignment[var_idx]
                    
                    # Calculate delta in total_unsat
                    delta = 0
                    for c_idx in var_to_clauses[var_idx]:
                        c_info = clause_var_info[c_idx]
                        old_w = clause_weights[c_idx]
                        
                        # Find the literal sign for var_idx in this clause c
                        # Since we know var_idx is in c, we can find its sign
                        # But we precomputed clause_var_info, so we need to find the sign for this specific var_idx
                        # Actually, we can just check the assignment change effect directly.
                        # If current_val is True (1), and sign is 1, lit is sat.
                        # If current_val is False (0), and sign is -1, lit is sat.
                        
                        # We need the sign of var_idx in clause c_idx.
                        # Let's find it.
                        c_sign = 0
                        for vv, ss in c_info:
                            if vv == var_idx:
                                c_sign = ss
                                break
                        if c_sign == 0:
                            continue # Should not happen

                        # Is the literal satisfied currently?
                        # If c_sign == 1, sat if assignment[var_idx] is True
                        # If c_sign == -1, sat if assignment[var_idx] is False
                        is_sat = (assignment[var_idx] == (c_sign == 1))
                        
                        if is_sat:
                            # Currently satisfied. Flipping makes it unsat.
                            # If old_w was 1, it becomes 0 (new unsat).
                            # If old_w > 1, it stays sat (w decreases by 1, but still >0).
                            if old_w == 1:
                                delta += 1 # Becomes unsat
                        else:
                            # Currently unsat. Flipping makes it sat.
                            # If old_w was 0, it becomes 1 (satisfied, so -1 unsat).
                            # If old_w > 0, it stays sat (w increases by 1).
                            if old_w == 0:
                                delta -= 1 # Becomes sat
                    
                    if delta < best_gain:
                        best_gain = delta
                        best_var_idx = var_idx
                        # Early exit if we found a perfect flip (satisfies clause, breaks none)
                        if best_gain == -1:
                            break
                            
                var_idx = best_var_idx
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause weights
            for c_idx in var_to_clauses[var_idx]:
                c_info = clause_var_info[c_idx]
                old_w = clause_weights[c_idx]
                
                # Find the sign of var_idx in this clause
                c_sign = 0
                for vv, ss in c_info:
                    if vv == var_idx:
                        c_sign = ss
                        break
                if c_sign == 0:
                    continue

                # Is the literal satisfied NOW (after flip)?
                is_sat = (assignment[var_idx] == (c_sign == 1))
                
                if is_sat:
                    clause_weights[c_idx] = old_w + 1
                else:
                    clause_weights[c_idx] = old_w - 1
                    
                # Update total_unsat
                if old_w == 0 and clause_weights[c_idx] > 0:
                    total_unsat -= 1
                elif old_w > 0 and clause_weights[c_idx] == 0:
                    total_unsat += 1

    return best_assignment
# EVOLVE-BLOCK-END
