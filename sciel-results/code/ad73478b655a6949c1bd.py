"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess: convert to 0-based, split positive/negative literals for speed
    # clause_pos[i] = list of variable indices (0-based) that must be True
    # clause_neg[i] = list of variable indices (0-based) that must be False
    clause_pos = []
    clause_neg = []
    for c in clauses:
        pos = []
        neg = []
        for l in c:
            if l > 0:
                pos.append(l - 1)
            else:
                neg.append(-l - 1)
        clause_pos.append(pos)
        clause_neg.append(neg)
        
    m = len(clauses)
    
    # Deadline guard: 330s limit, use 325s for safety
    start_time = time.time()
    deadline = start_time + 325.0
    
    # We will try multiple random restarts with WalkSAT-like heuristics
    # The key improvement over the seed is better variable selection heuristics
    # and adaptive parameters.
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Initial random assignment
    rng = random.Random(0)
    
    # Strategy: Use a hybrid approach
    # 1. Start with a random assignment
    # 2. Use WalkSAT with a "walk" probability that adapts
    # 3. Use a clause-weight based heuristic to pick which variable to flip
    
    # Precompute literal occurrence counts for each variable (for heuristic)
    # This helps in deciding which variable is more "critical"
    var_occurrence = [0] * n
    for i in range(m):
        for v in clause_pos[i]:
            var_occurrence[v] += 1
        for v in clause_neg[i]:
            var_occurrence[v] += 1
    
    # Normalize occurrence counts
    max_occ = max(var_occurrence) if var_occurrence else 1
    var_weight = [occ / max_occ for occ in var_occurrence]
    
    max_restarts = 100
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check
        unsat_count = 0
        for i in range(m):
            sat = False
            for v in clause_pos[i]:
                if assignment[v]:
                    sat = True
                    break
            if not sat:
                for v in clause_neg[i]:
                    if not assignment[v]:
                        sat = True
                        break
            if not sat:
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
        
        # WalkSAT with adaptive walk probability
        # Start with low walk prob, increase if stuck
        walk_prob = 0.1
        no_improve_count = 0
        prev_unsat = unsat_count
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
            
            # Find an unsatisfied clause
            # For speed, we pick a random clause and check if it's unsat
            # If not, try a few more
            clause_idx = None
            for _ in range(10):
                ci = rng.randrange(m)
                sat = False
                for v in clause_pos[ci]:
                    if assignment[v]:
                        sat = True
                        break
                if not sat:
                    for v in clause_neg[ci]:
                        if not assignment[v]:
                            sat = True
                            break
                if not sat:
                    clause_idx = ci
                    break
                    
            if clause_idx is None:
                # Fallback: scan from start
                for ci in range(m):
                    sat = False
                    for v in clause_pos[ci]:
                        if assignment[v]:
                            sat = True
                            break
                    if not sat:
                        for v in clause_neg[ci]:
                            if not assignment[v]:
                                sat = True
                                break
                    if not sat:
                        clause_idx = ci
                        break
            
            if clause_idx is None:
                break  # Should be satisfied
            
            # Pick a variable to flip from this clause
            # With probability walk_prob, pick randomly
            # Otherwise, pick the variable that improves satisfaction the most
            if rng.random() < walk_prob:
                # Random pick
                all_vars = clause_pos[clause_idx] + clause_neg[clause_idx]
                v = all_vars[rng.randrange(len(all_vars))]
            else:
                # Greedy: pick variable that maximizes improvement
                best_v = None
                best_score = -1
                
                # Variables in positive part: flipping to True might satisfy
                # Variables in negative part: flipping to False might satisfy
                candidates = clause_pos[clause_idx] + clause_neg[clause_idx]
                
                for v in candidates:
                    # Calculate how many clauses would be affected
                    # Simplified: just pick the one with highest weight
                    score = var_weight[v]
                    if score > best_score:
                        best_score = score
                        best_v = v
                
                if best_v is None:
                    best_v = candidates[0]
                v = best_v
            
            # Flip the variable
            assignment[v] = not assignment[v]
            
            # Update unsat count (approximate: just recalculate for speed in small cases)
            # For n=50, m is likely not too large, so full recalc is okay
            new_unsat = 0
            for i in range(m):
                sat = False
                for vv in clause_pos[i]:
                    if assignment[vv]:
                        sat = True
                        break
                if not sat:
                    for vv in clause_neg[i]:
                        if not assignment[vv]:
                            sat = True
                            break
                if not sat:
                    new_unsat += 1
                    
            unsat_count = new_unsat
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
            
            # Track improvement
            if unsat_count < prev_unsat:
                no_improve_count = 0
                walk_prob = max(0.05, walk_prob * 0.8)
            else:
                no_improve_count += 1
                if no_improve_count % 100 == 0:
                    walk_prob = min(0.5, walk_prob * 1.2)
            
            prev_unsat = unsat_count
            
            if unsat_count == 0:
                return assignment
        
        # End of restart steps
        if time.time() > deadline:
            break
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
