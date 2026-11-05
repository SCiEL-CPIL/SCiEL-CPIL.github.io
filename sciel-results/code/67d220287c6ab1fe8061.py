import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute clause literals as 0-indexed variables and signs
    # lit: (var_idx, is_positive)
    # Store as list of (var, sign) where sign is 1 for pos, -1 for neg
    parsed_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            pc.append((v, sign))
        parsed_clauses.append(pc)
    
    # Mapping variable to clauses it appears in
    var_to_clauses = [[] for _ in range(n)]
    for i, pc in enumerate(parsed_clauses):
        for v, _ in pc:
            var_to_clauses[v].append(i)

    rng = random.Random(42)
    deadline = time.time() + 290  # Safety margin for 300s limit
    
    # Weights for clauses and variables
    clause_weights = [1.0] * len(clauses)
    var_weights = [1.0] * n
    
    # Phase saving: remember last value that satisfied a clause
    phase = [False] * n
    
    best_assignment = None
    best_unsat = float('inf')
    
    # WalkSAT parameters
    base_noise = 0.05
    max_noise = 0.5
    noise_step = 0.02
    steps_per_restart = 10000
    max_restarts = 500
    
    def is_lit_true(v, sign, assignment):
        if sign == 1:
            return assignment[v]
        else:
            return not assignment[v]

    def is_clause_satisfied(clause, assignment):
        for v, sign in clause:
            if is_lit_true(v, sign, assignment):
                return True
        return False

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        # Use phase saving if available, else random
        assignment = [phase[i] if rng.random() < 0.5 else rng.random() < 0.5 for i in range(n)]
        
        # Calculate initial unsat
        unsat_indices = set()
        for i, c in enumerate(parsed_clauses):
            if not is_clause_satisfied(c, assignment):
                unsat_indices.add(i)
        
        cur_unsat = len(unsat_indices)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment
        
        noise = base_noise
        stall_count = 0
        last_unsat = cur_unsat
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if not unsat_indices:
                return assignment
            
            # Pick a random unsatisfied clause
            c_idx = rng.choice(list(unsat_indices))
            clause = parsed_clauses[c_idx]
            
            # Unit Clause Heuristic
            # If there is a clause with exactly 1 unsatisfied literal, 
            # strongly prefer flipping that variable.
            # We check the chosen clause first.
            unsat_lits_in_clause = [(v, s) for v, s in clause if not is_lit_true(v, s, assignment)]
            
            target_var = None
            if len(unsat_lits_in_clause) == 1:
                # Force flip this variable to satisfy this clause
                target_var = unsat_lits_in_clause[0][0]
            else:
                # Standard WalkSAT: pick variable from this clause
                # Score each variable in the clause
                best_score = float('inf')
                best_var = None
                best_weight = 0
                
                for v, s in unsat_lits_in_clause:
                    # Calculate gain/loss if we flip v
                    # Flipping v satisfies this clause (c_idx).
                    # Check other clauses containing v.
                    broken = 0
                    gained = 0
                    
                    for other_c_idx in var_to_clauses[v]:
                        if other_c_idx == c_idx:
                            continue
                        
                        other_clause = parsed_clauses[other_c_idx]
                        
                        # Check if other_clause is currently satisfied
                        # If it is satisfied, does it remain satisfied?
                        # If it is unsatisfied, does it become satisfied?
                        
                        # Count unsat literals in other_clause
                        # Note: We only care about the literal with var v
                        # Let's find the literal for v in other_clause
                        other_v_lit = None
                        for ov, os in other_clause:
                            if ov == v:
                                other_v_lit = (ov, os)
                                break
                            
                        if other_v_lit is None:
                            continue
                            
                        # Current state of other_clause
                        # We can approximate by checking if it's in unsat_indices
                        if other_c_idx in unsat_indices:
                            # It is unsat.
                            # Flipping v will make the literal for v true.
                            # Since the clause was unsat (all literals false), 
                            # making one true satisfies the clause.
                            # So we gain 1.
                            gained += 1
                        else:
                            # It is satisfied.
                            # We need to check if it relies solely on v.
                            # If it has another true literal, it stays sat.
                            # If v was the ONLY true literal, it becomes unsat (broken).
                            
                            # Check if any other literal in other_clause is true
                            has_other_true = False
                            for ov, os in other_clause:
                                if ov == v:
                                    continue
                                if is_lit_true(ov, os, assignment):
                                    has_other_true = True
                                    break
                                    
                            if not has_other_true:
                                # v was the only true literal
                                broken += 1
                            
                    # Score: minimize broken, maximize gained
                    # We want to flip if broken < gained or broken is small
                    # Standard WalkSAT minimizes broken.
                    # Here we use a weighted score.
                    score = broken - 0.5 * gained
                    
                    # Tie-breaker: clause weight and var weight
                    if score < best_score or (score == best_score and (clause_weights[c_idx] * var_weights[v]) > best_weight):
                        best_score = score
                        best_var = v
                        best_weight = clause_weights[c_idx] * var_weights[v]
                
                # Noise
                if rng.random() < noise:
                    # Pick random var from unsat literals in clause
                    v, s = rng.choice(unsat_lits_in_clause)
                    target_var = v
                else:
                    target_var = best_var
            
            if target_var is None:
                continue
            
            # Perform flip
            assignment[target_var] = not assignment[target_var]
            phase[target_var] = assignment[target_var]
            
            # Update unsat_indices
            # Only need to check clauses containing target_var
            for c_idx in var_to_clauses[target_var]:
                c = parsed_clauses[c_idx]
                sat = is_clause_satisfied(c, assignment)
                if sat:
                    if c_idx in unsat_indices:
                        unsat_indices.remove(c_idx)
                        # Update phase for this clause? 
                        # Phase saving usually updates when a clause becomes sat.
                        # We already updated phase[target_var].
                else:
                    if c_idx not in unsat_indices:
                        unsat_indices.add(c_idx)
            
            # Update weights
            # Increase weight of unsat clauses
            for c_idx in unsat_indices:
                clause_weights[c_idx] += 0.1
            # Increase weight of vars in unsat clauses
            for c_idx in unsat_indices:
                for v, s in parsed_clauses[c_idx]:
                    var_weights[v] += 0.1
            
            # Decay weights periodically to prevent overflow
            if step % 1000 == 0:
                for i in range(len(clauses)):
                    clause_weights[i] *= 0.9
                for i in range(n):
                    var_weights[i] *= 0.9
            
            # Stall detection
            if len(unsat_indices) < last_unsat:
                last_unsat = len(unsat_indices)
                stall_count = 0
                noise = max(base_noise, noise - noise_step)
            else:
                stall_count += 1
                if stall_count > 100:
                    noise = min(max_noise, noise + noise_step)
                
        # End local search loop
        if time.time() > deadline:
            break

    if best_assignment:
        return best_assignment
    return [False] * n
