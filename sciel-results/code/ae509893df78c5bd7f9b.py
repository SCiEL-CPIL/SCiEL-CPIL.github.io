"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: finish well before the 300s wall-clock limit.
    deadline = time.time() + 290.0
    
    # Precompute clause indices and literal signs for speed.
    # Each clause is a list of signed 1-based literals.
    # Convert to 0-based indices with sign.
    cl_idx = []
    cl_sign = []
    for c in clauses:
        idx = [abs(l) - 1 for l in c]
        sign = [1 if l > 0 else 0 for l in c]  # 1 means positive, 0 means negative
        cl_idx.append(idx)
        cl_sign.append(sign)
    m = len(clauses)
    
    # Random seed per call? Use a fixed seed for reproducibility, but vary slightly.
    rng = random.Random(42)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    # WalkSAT with restarts and adaptive clause weighting.
    # We'll run multiple restarts, each with a flip budget.
    
    # Precompute for each variable, which clauses contain it.
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in cl_idx[i]:
            var_to_clauses[v].append(i)
    
    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            sat = False
            for j in range(3):
                v = cl_idx[i][j]
                s = cl_sign[i][j]
                if assignment[v] == s:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    # Try a few different random seeds / strategies
    for restart in range(200):
        if time.time() >= deadline - 5.0:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT phase
        # Adaptive: use a noise parameter that decreases over time
        noise = 0.45
        
        # Flip budget per restart
        flip_budget = 5000
        if restart > 50:
            flip_budget = 3000
        if restart > 100:
            flip_budget = 2000
        
        for step in range(flip_budget):
            if time.time() >= deadline - 5.0:
                break
            
            # Find an unsatisfied clause
            # Instead of scanning all, pick a random clause and check
            # For efficiency, maintain a list of unsat clauses
            # Simple approach: pick random clause, if sat, continue (wasteful)
            # Better: maintain unsat list
            
            # Let's use a simpler but effective approach:
            # Pick a random unsatisfied clause by sampling
            
            # Actually, let's maintain unsat clauses properly for the first few steps
            # then use sampling
            
            # For simplicity and speed, let's do:
            # Pick a random clause index, check if unsat
            # If sat, pick another (up to a few tries)
            # This is O(1) expected per step if unsat fraction is high
            
            unsat_clause_idx = -1
            for _ in range(10):
                ci = rng.randrange(m)
                sat = False
                for j in range(3):
                    v = cl_idx[ci][j]
                    s = cl_sign[ci][j]
                    if assignment[v] == s:
                        sat = True
                        break
                if not sat:
                    unsat_clause_idx = ci
                    break
            
            if unsat_clause_idx == -1:
                # All clauses are satisfied? Verify.
                unsat_count = count_unsat(assignment)
                if unsat_count == 0:
                    return assignment
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]
                # Couldn't find unsat clause by sampling, break out
                break
            
            ci = unsat_clause_idx
            idx = cl_idx[ci]
            sign = cl_sign[ci]
            
            # Score each literal in the clause: count how many other clauses would be satisfied
            # if we flip this variable
            best_var = -1
            best_score = -1
            best_gain = 0
            best_loss = 0
            
            for j in range(3):
                v = idx[j]
                s = sign[j]
                # Current value
                cur_val = assignment[v]
                # If we flip, new value is 1 - cur_val
                # This literal is currently unsatisfied (since clause is unsat)
                # After flip, this literal becomes satisfied: gain = 1 for this clause
                # But we need to count impact on other clauses
                
                # Count clauses that would become satisfied (currently unsat, this var is the only unsat literal)
                # and clauses that would become unsatisfied (currently sat, this var is the only sat literal)
                
                # For efficiency, just count the net change
                # gain: clauses where this var is the only unsat literal
                # loss: clauses where this var is the only sat literal
                
                # We need to check all clauses containing v
                # This is expensive. Let's use a simpler scoring:
                # Randomly pick among the 3 with some weighting
                
                # Actually, let's do a proper but bounded score:
                # For each var in clause, count how many clauses it appears in
                # and use a heuristic: prefer vars that appear in more clauses
                # (they have more impact)
                
                # Let's do a quick score: for each candidate var, 
                # check a few random other clauses containing it
                
                score = 0
                for ci2 in var_to_clauses[v]:
                    if ci2 == ci:
                        continue
                    # Check if flipping v would satisfy ci2
                    # i.e., is v the only unsat literal in ci2?
                    all_other_sat = True
                    for k in range(3):
                        if k == j:
                            continue
                        v2 = cl_idx[ci2][k]
                        s2 = cl_sign[ci2][k]
                        if assignment[v2] != s2:
                            all_other_sat = False
                            break
                    if all_other_sat:
                        # Flipping v would satisfy ci2
                        score += 1
                        continue
                    
                    # Check if flipping v would unsatisfy ci2
                    # i.e., is v the only sat literal in ci2?
                    all_other_unsat = True
                    for k in range(3):
                        if k == j:
                            continue
                        v2 = cl_idx[ci2][k]
                        s2 = cl_sign[ci2][k]
                        if assignment[v2] == s2:
                            all_other_unsat = False
                            break
                    if all_other_unsat:
                        # Flipping v would unsatisfy ci2
                        score -= 1
                
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_gain = score  # rough
        
            if best_var == -1:
                best_var = idx[0]
            
            # Decide whether to flip based on noise
            if rng.random() < noise:
                assignment[best_var] = not assignment[best_var]
            else:
                # Flip the best scoring var
                assignment[best_var] = not assignment[best_var]
            
            # Update unsat count incrementally? Too complex. 
            # Just check periodically.
            if step % 100 == 0:
                unsat_count = count_unsat(assignment)
                if unsat_count == 0:
                    return assignment
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]
        
        # End of restart
        if best_unsat_count == 0:
            return best_assignment
    
    # Return the best assignment found
    if best_assignment is not None:
        # Final check
        if count_unsat(best_assignment) == 0:
            return best_assignment
        return best_assignment
    
    # Fallback: all false
    return [False] * n

# EVOLVE-BLOCK-END
