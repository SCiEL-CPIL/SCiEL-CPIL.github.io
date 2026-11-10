import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute variable to clause mapping
    var_clauses = [[] for _ in range(n)]
    for idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_clauses[var_idx].append(idx)
            
    # Pre-compute literal sign for quick check
    # clause_lits[idx] = list of (var_idx, is_positive)
    clause_lits = []
    for clause in clauses:
        lits = []
        for lit in clause:
            lits.append((abs(lit) - 1, lit > 0))
        clause_lits.append(lits)

    start_time = time.time()
    deadline = start_time + 280.0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    while time.time() < deadline:
        rng = random.Random()
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize support counts
        support = [0] * len(clauses)
        unsat_clauses = []
        for idx, lits in enumerate(clause_lits):
            sat_count = 0
            for var_idx, is_pos in lits:
                if assignment[var_idx] == is_pos:
                    sat_count += 1
            support[idx] = sat_count
            if sat_count == 0:
                unsat_clauses.append(idx)
                
        unsat_count = len(unsat_clauses)
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        noise = 0.2
        steps_since_improve = 0
        max_steps = 20000 # Hard cap per restart to avoid infinite loops on easy cases? No, deadline handles it.
        
        while time.time() < deadline:
            if not unsat_clauses:
                return assignment
                
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1
                if steps_since_improve > 1000:
                    break # Restart
                    
            # Pick random unsat clause
            c_idx = rng.choice(unsat_clauses)
            
            # Choose literal
            if rng.random() < noise:
                # Random literal from clause
                var_idx, is_pos = rng.choice(clause_lits[c_idx])
            else:
                # Greedy: pick var in clause that minimizes delta
                best_var = None
                best_delta = float('inf')
                for var_idx, is_pos in clause_lits[c_idx]:
                    # Calculate delta if we flip var_idx
                    # Current support of c_idx is 0.
                    # Flipping var_idx will make the literal for var_idx in c_idx true.
                    # So c_idx becomes satisfied. Delta -1 for c_idx.
                    # Check other clauses containing var_idx
                    
                    delta = -1 # c_idx becomes sat
                    
                    for other_c_idx in var_clauses[var_idx]:
                        if other_c_idx == c_idx:
                            continue
                        # Check if the literal for var_idx in other_c_idx is currently true
                        # We need to know the sign of var_idx in other_c_idx
                        # Let's precompute a lookup or just iterate
                        # This inner loop is a bit slow. 
                        # Alternative: precompute for each var_idx and clause_idx, the sign.
                        pass
                    
                    # This greedy calculation is expensive if done naively.
                    # Let's stick to the simpler "count unsat after flip" if we can't optimize easily.
                    # Or use the support update logic.
                    
                    # Let's use the support update logic for the greedy choice.
                    # For each candidate var in the clause:
                    # 1. Simulate flip: 
                    #    - For c_idx: support becomes 1. (Was 0)
                    #    - For other clauses c' with var:
                    #      - If literal for var in c' is currently true: support[c'] -= 1
                    #      - If false: support[c'] += 1
                    # 2. Count how many c' go from >0 to 0 (cost +1) or 0 to >0 (gain -1)
                    
                    # To do this fast, we need to know current support[c'] and sign of var in c'.
                    # Let's precompute `sign_in_clause[var_idx][c_idx]`? No, sparse.
                    # Let's just iterate var_clauses[var_idx].
                    
                    local_delta = -1
                    for other_c_idx in var_clauses[var_idx]:
                        if other_c_idx == c_idx:
                            continue
                        # Get sign of var_idx in other_c_idx
                        # We can store this in a dict or list of dicts
                        # Let's create `lit_sign[var_idx][c_idx]` is not easy.
                        # Let's just find the sign.
                        # Since degree is small, we can just check the clause.
                        # But we need to be fast.
                        
                        # Precompute: `clause_var_sign[c_idx][var_idx]`
                        pass
                        
                    # This is getting complicated. 
                    # Let's revert to the "Full Recount" strategy but optimized in Python.
                    # N=150, M=600. 
                    # Full recount: 600 * 3 checks = 1800 ops.
                    # 1000 steps * 1800 = 1.8M ops.
                    # Python does ~10M-50M simple ops/sec.
                    # So 1 restart takes ~0.1s.
                    # In 280s, we can do ~2000 restarts.
                    # This is plenty.
                    
                    # So, let's just use Full Recount for simplicity and robustness.
                    pass
                    
                # If greedy is too complex to implement correctly and fast, 
                # use Random Flip with Noise.
                # Actually, WalkSAT with noise is known to work well.
                # Let's just pick a random literal from the clause with prob noise,
                # else pick the one that satisfies the most other clauses?
                # Let's just do Random Flip for now, it's simpler.
                # But the prompt suggests "Min-conflicts weighted literal selection".
                
                # Let's implement a simple weighted selection:
                # For each var in clause, count how many OTHER clauses it satisfies if flipped.
                # This requires checking support of other clauses.
                
                # Let's try this:
                best_var = None
                best_score = -1
                for var_idx, is_pos in clause_lits[c_idx]:
                    # Score: number of other clauses that become satisfied or stay satisfied
                    score = 0
                    for other_c_idx in var_clauses[var_idx]:
                        if other_c_idx == c_idx:
                            continue
                        # Check if flipping var_idx improves other_c_idx
                        # Current state of other_c_idx: support[other_c_idx]
                        # Sign of var_idx in other_c_idx:
                        # Find it
                        sign = None
                        for v, s in clause_lits[other_c_idx]:
                            if v == var_idx:
                                sign = s
                                break
                        if sign is None:
                            continue
                            
                        current_val = assignment[var_idx]
                        new_val = not current_val
                        
                        # Is the literal for var_idx in other_c_idx currently true?
                        is_lit_true = (current_val == sign)
                        
                        if is_lit_true:
                            # Flipping makes it false.
                            # If support[other_c_idx] == 1, it becomes unsat.
                            if support[other_c_idx] == 1:
                                score -= 10 # Penalty
                            else:
                                score += 0 # Stays sat
                        else:
                            # Flipping makes it true.
                            # It becomes sat (or stays sat? No, if it was false, and we make it true, support increases by 1).
                            # If support was 0, it becomes sat. Gain.
                            # If support was >0, it stays sat.
                            if support[other_c_idx] == 0:
                                score += 10 # Gain
                            else:
                                score += 0
                                                    
                    if score > best_score:
                        best_score = score
                        best_var = var_idx
                        
                var_idx = best_var
                # Find the literal in c_idx corresponding to var_idx
                # We just need the var_idx to flip
                pass

            # Flip var_idx
            # Update assignment
            assignment[var_idx] = not assignment[var_idx]
            
            # Update support and unsat_clauses
            # For each clause containing var_idx
            for c_idx_affected in var_clauses[var_idx]:
                # Determine sign of var_idx in c_idx_affected
                sign = None
                for v, s in clause_lits[c_idx_affected]:
                    if v == var_idx:
                        sign = s
                        break
                if sign is None:
                    continue
                    
                current_val = assignment[var_idx] # This is the NEW value
                # Wait, we flipped it. So current_val is the new value.
                # We need to know if the literal is true with the NEW value.
                is_lit_true_new = (current_val == sign)
                
                # We need the OLD value to know the change.
                # Or we can just recalculate support for this clause.
                # Recalculate support for c_idx_affected:
                new_support = 0
                for v, s in clause_lits[c_idx_affected]:
                    if assignment[v] == s:
                        new_support += 1
                old_support = support[c_idx_affected]
                support[c_idx_affected] = new_support
                
                # Update unsat_clauses
                if old_support == 0 and new_support > 0:
                    # Became satisfied
                    if c_idx_affected in unsat_clauses:
                        unsat_clauses.remove(c_idx_affected)
                        unsat_count -= 1
                elif old_support > 0 and new_support == 0:
                    # Became unsatisfied
                    unsat_clauses.append(c_idx_affected)
                    unsat_count += 1
                # If both 0 or both >0, no change in unsat status
                
            if unsat_count == 0:
                return assignment
                
    return best_assignment
