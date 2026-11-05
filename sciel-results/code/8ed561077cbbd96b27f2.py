import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute clause literals as 0-indexed variables and signs
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
            
    # --- Pre-processing: Unit Propagation and Pure Literal Elimination ---
    # This can simplify the instance significantly.
    # We work on a copy of the clauses to avoid modifying the original if we need to backtrack (though we don't backtrack here, we just simplify).
    # Note: Pure literal elimination is sound for SAT.
    # Unit propagation is sound.
    
    # To keep it simple and fast, we'll do a single pass of simplification.
    # If this is too complex or risky, we can skip it, but it's a standard "solver" improvement.
    # Given the "Do NOT switch design family" constraint, pre-processing is generally accepted as part of the solver pipeline.
    
    # Let's stick to the core WalkSAT improvements first to be safe, as pre-processing might introduce bugs if not handled carefully with the mapping back.
    # However, for 50 variables, it's very fast.
    
    # Let's focus on the WalkSAT improvements.
    
    rng = random.Random(42)
    deadline = time.time() + 270
    
    # Weights for clauses and variables (unused in new heuristic, but kept for structure)
    clause_weights = [1.0] * len(clauses)
    var_weights = [1.0] * n
    
    # Phase saving
    phase = [False] * n
    
    best_assignment = None
    best_unsat = float('inf')
    
    # WalkSAT parameters
    base_noise = 0.1
    max_noise = 0.4
    noise_step = 0.02
    steps_per_restart = 15000
    max_restarts = 500
    polish_threshold = 10  # When unsat <= this, enter polish mode
    polish_steps = 50000
    
    def is_clause_satisfied_fast(clause, assignment):
        for v, sign in clause:
            if sign == 1:
                if assignment[v]:
                    return True
            else:
                if not assignment[v]:
                    return True
        return False
        
    def count_conflicts(v, s, assignment, unsat_indices):
        """
        Count how many satisfied clauses would be broken if we flip v to satisfy s.
        s is the sign of the literal in the clause we are trying to satisfy.
        We are flipping v.
        """
        broken = 0
        for c_idx in var_to_clauses[v]:
            c = parsed_clauses[c_idx]
            # Check if this clause is currently satisfied
            # If it is, check if v is the only literal satisfying it.
            # If so, flipping v breaks it.
            sat = False
            v_satisfies = False
            for ov, os in c:
                if os == 1:
                    if assignment[ov]:
                        sat = True
                        if ov == v:
                            v_satisfies = True
                else:
                    if not assignment[ov]:
                        sat = True
                        if ov == v:
                            v_satisfies = True
            if sat and v_satisfies:
                broken += 1
        return broken

    def count_gains(v, assignment, unsat_indices):
        """
        Count how many unsatisfied clauses would be satisfied if we flip v.
        """
        gained = 0
        for c_idx in var_to_clauses[v]:
            if c_idx in unsat_indices:
                # If v is flipped, does it satisfy this clause?
                # We only call this for v which is in an unsat clause.
                # If v is in an unsat clause, flipping it will satisfy it ONLY IF the literal for v in that clause is currently false.
                # Since the clause is unsat, all literals are false.
                # So flipping v will make the literal for v true.
                # Thus, it satisfies the clause.
                gained += 1
        return gained

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment with phase saving
        assignment = [phase[i] if rng.random() < 0.5 else rng.random() < 0.5 for i in range(n)]
        
        # Calculate initial unsat
        unsat_indices = set()
        for i, c in enumerate(parsed_clauses):
            if not is_clause_satisfied_fast(c, assignment):
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
        polish_mode = False
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if not unsat_indices:
                return assignment
            
            # Switch to polish mode if close to solution
            if not polish_mode and cur_unsat <= polish_threshold:
                polish_mode = True
                noise = 0.0 # Deterministic in polish
                # Extend steps for polish
                steps_per_restart = step + polish_steps
                last_unsat = cur_unsat
                stall_count = 0
            
            # Pick a random unsatisfied clause
            c_idx = rng.choice(list(unsat_indices))
            clause = parsed_clauses[c_idx]
            
            # Get unsatisfied literals in this clause
            unsat_lits_in_clause = []
            for v, s in clause:
                if s == 1:
                    if not assignment[v]:
                        unsat_lits_in_clause.append((v, s))
                else:
                    if assignment[v]:
                        unsat_lits_in_clause.append((v, s))
                        
            target_var = None
            
            if len(unsat_lits_in_clause) == 1:
                # Unit clause: force flip
                target_var = unsat_lits_in_clause[0][0]
            else:
                if polish_mode:
                    # MinConflicts strategy:
                    # Pick the variable that breaks the fewest satisfied clauses.
                    # If there's a tie, pick randomly.
                    min_broken = float('inf')
                    candidates = []
                    for v, s in unsat_lits_in_clause:
                        broken = count_conflicts(v, s, assignment, unsat_indices)
                        if broken < min_broken:
                            min_broken = broken
                            candidates = [v]
                        elif broken == min_broken:
                            candidates.append(v)
                    target_var = rng.choice(candidates)
                else:
                    # WalkSAT heuristic
                    best_score = float('inf')
                    best_var = None
                    
                    # Adaptive weight for gains
                    # As cur_unsat decreases, we value gains more to avoid breaking too much
                    gain_weight = 1.0 + 0.5 / (cur_unsat + 1.0)
                    
                    for v, s in unsat_lits_in_clause:
                        broken = count_conflicts(v, s, assignment, unsat_indices)
                        gained = count_gains(v, assignment, unsat_indices)
                        
                        score = broken - gain_weight * gained
                        
                        if score < best_score:
                            best_score = score
                            best_var = v
                            
                    # Noise-based random selection
                    if rng.random() < noise:
                        v, s = rng.choice(unsat_lits_in_clause)
                        target_var = v
                    else:
                        target_var = best_var
            
            if target_var is None:
                continue
            
            # Perform flip
            assignment[target_var] = not assignment[target_var]
            phase[target_var] = assignment[target_var]
            
            # Update unsat_indices for affected clauses
            for c_idx2 in var_to_clauses[target_var]:
                c = parsed_clauses[c_idx2]
                sat = is_clause_satisfied_fast(c, assignment)
                if sat:
                    if c_idx2 in unsat_indices:
                        unsat_indices.remove(c_idx2)
                else:
                    if c_idx2 not in unsat_indices:
                        unsat_indices.add(c_idx2)
            
            cur_unsat = len(unsat_indices)
            
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
            
            # Stall detection and noise adjustment
            if cur_unsat < last_unsat:
                last_unsat = cur_unsat
                stall_count = 0
                if not polish_mode:
                    noise = max(base_noise, noise - noise_step)
            else:
                stall_count += 1
                if stall_count > 50 and not polish_mode:
                    noise = min(max_noise, noise + noise_step)
        
        # End local search loop
        if time.time() > deadline:
            break
    
    if best_assignment is not None:
        return best_assignment
    return [False] * n
