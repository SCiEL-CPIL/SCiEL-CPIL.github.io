"""Improved planted 3-SAT solver: weighted WalkSAT with min-conflicts, restarts, SA escapes, and deadline guard."""
import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 300s wall-clock limit, use 295s as safety margin
    deadline = time.time() + 295.0
    
    # Precompute: for each variable, which clauses contain it (positive or negative)
    # var_clauses[i] = list of (clause_idx, sign) where sign=1 if var i positive in clause
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            vi = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[vi].append((ci, sign))
    
    # Helper to check if a clause is satisfied
    def clause_satisfied(clause, assignment):
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False
    
    # Count unsatisfied clauses
    def count_unsat(assignment):
        cnt = 0
        for c in clauses:
            if not clause_satisfied(c, assignment):
                cnt += 1
        return cnt
    
    # Weighted clause selection + min-conflicts variable scoring
    # With simulated annealing-style escapes and restarts
    def walksat_once(assignment, rng, max_steps, temperature):
        """Run WalkSAT with weighted clause selection and min-conflicts variable scoring.
        Returns (assignment, solved) where solved is True if all clauses satisfied."""
        n = len(assignment)
        # Track unsatisfied clause count per variable for min-conflicts
        # For each variable, count how many unsatisfied clauses it appears in
        # and how many of those would be satisfied if we flipped it
        
        # Build initial state
        unsat_clauses = []
        for ci, c in enumerate(clauses):
            if not clause_satisfied(c, assignment):
                unsat_clauses.append(ci)
        
        if not unsat_clauses:
            return assignment, True
        
        # For efficiency, track per-variable: 
        #   num_unsat_containing[var] = number of unsat clauses containing var
        #   num_unsat_satisfied_if_flip[var] = among unsat clauses containing var, 
        #       how many have var as the only false literal (so flipping var satisfies them)
        # Actually, let's use a simpler approach: for each var, count unsat clauses it's in,
        # and count how many of those would become sat if we flip it.
        
        def compute_var_scores(assignment, unsat_clauses):
            """Compute for each variable: (num_unsat_in, num_would_satisfy_on_flip)"""
            num_unsat_in = [0] * n
            num_would_satisfy = [0] * n
            for ci in unsat_clauses:
                c = clauses[ci]
                # For this unsat clause, all literals are false
                # If we flip a variable in this clause, that literal becomes true, satisfying the clause
                # But we need to check: does flipping var satisfy this clause?
                # Since clause is unsat, all literals are false. Flipping any var in the clause
                # makes that literal true, thus satisfying the clause.
                for lit in c:
                    vi = abs(lit) - 1
                    num_unsat_in[vi] += 1
                    num_would_satisfy[vi] += 1
            return num_unsat_in, num_would_satisfy
        
        num_unsat_in, num_would_satisfy = compute_var_scores(assignment, unsat_clauses)
        
        # Weighted clause selection: prefer clauses with fewer satisfied literals
        # (but all unsat clauses have 0 satisfied literals in 3-SAT... so all equal weight)
        # Instead, use clause frequency as weight or just uniform over unsat
        # For planted instances, let's use a slight bias toward clauses that are "harder"
        # We'll use uniform selection from unsat clauses but with min-conflicts for variable
        
        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False
            
            if not unsat_clauses:
                return assignment, True
            
            # Weighted clause selection: pick an unsat clause
            # Use probability proportional to 1 (uniform) or weighted by clause size
            ci = rng.choice(unsat_clauses)
            clause = clauses[ci]
            
            # Score each variable in the clause
            # Min-conflicts: minimize number of unsat clauses that variable is in
            # After flipping, the variable's contribution changes
            best_var = None
            best_score = None
            best_delta = None
            
            candidates = []
            for lit in clause:
                vi = abs(lit) - 1
                # If we flip vi:
                #   - The current clause ci will be satisfied (since all lits are false, flipping makes one true)
                #   - Other clauses containing vi may change status
                # Score: net change in unsat count
                #   -1 for clause ci (becomes sat)
                #   +1 for each other unsat clause containing vi where vi was the only false lit? No...
                #   Actually, for other unsat clauses containing vi: flipping vi makes that lit true, 
                #   so those clauses also become satisfied. So we get -1 for each.
                #   For currently satisfied clauses containing vi: flipping vi might make them unsat
                #   if vi was the only true literal in that clause.
                
                # Simplified min-conflicts: 
                #   delta = -(num_would_satisfy[vi] - 1) - (num_sat_losing[vi])
                #   where num_would_satisfy[vi] includes clause ci
                #   num_sat_losing[vi] = number of currently-sat clauses where vi is the only true lit
                
                # Let's just use: score = num_unsat_in[vi] (minimize this)
                # With a tie-breaker: prefer flipping if it satisfies more clauses
                # This is the standard min-conflicts heuristic
                
                # For SA escape: with probability temperature, pick a random variable
                if temperature > 0 and rng.random() < temperature:
                    score = rng.random() * 10  # random score
                else:
                    # Min-conflicts: minimize unsat clauses the variable is in
                    # After flip, clause ci becomes sat, so effective unsat_in decreases by 1
                    # But other clauses may also change
                    # Simple heuristic: use num_unsat_in[vi] as score (lower is better)
                    score = num_unsat_in[vi]
                
                candidates.append((score, vi, lit))
            
            # Pick the variable with minimum score (min-conflicts)
            candidates.sort(key=lambda x: x[0])
            _, vi, lit = candidates[0]
            
            # Flip the variable
            old_val = assignment[vi]
            assignment[vi] = not old_val
            
            # Update unsat_clauses and var scores
            # Determine which clauses' status changed
            # Clauses containing vi that were unsat: they may become sat
            # Clauses containing vi that were sat: they may become unsat (if vi was only true lit)
            
            # Remove ci from unsat_clauses (it's now satisfied)
            unsat_clauses.remove(ci)
            
            # Check other clauses containing vi
            changed_clauses = []
            for (oci, osign) in var_clauses[vi]:
                if oci == ci:
                    continue
                # Check if this clause's status changed
                was_unsat = False
                for uci in unsat_clauses:
                    if uci == oci:
                        was_unsat = True
                        break
                
                is_now_satisfied = clause_satisfied(clauses[oci], assignment)
                
                if was_unsat and is_now_satisfied:
                    # Clause became satisfied
                    unsat_clauses.remove(oci)
                    changed_clauses.append((oci, 'became_sat'))
                elif (not was_unsat) and (not is_now_satisfied):
                    # Clause became unsatisfied
                    unsat_clauses.append(oci)
                    changed_clauses.append((oci, 'became_unsat'))
            
            # Recompute var scores (simplified: recompute from scratch for correctness)
            # This is O(m) per step, which for 150 vars and ~600 clauses is manageable
            # But for speed, let's do incremental updates
            # For simplicity and correctness, recompute
            num_unsat_in, num_would_satisfy = compute_var_scores(assignment, unsat_clauses)
        
        # Final check
        if not unsat_clauses:
            return assignment, True
        return assignment, False
    
    # Main loop with restarts
    best_assignment = None
    best_unsat = float('inf')
    
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initial assignment
        rng = random.Random(0 + restart * 12345)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial unsat count
        unsat_cnt = count_unsat(assignment)
        if unsat_cnt == 0:
            return assignment
        
        if unsat_cnt < best_unsat:
            best_unsat = unsat_cnt
            best_assignment = list(assignment)
        
        # Run WalkSAT with increasing step budget
        # Use adaptive step count
        base_steps = 5000
        max_steps = base_steps + restart * 1000
        
        # Temperature for SA escapes: starts higher, decreases
        temperature = 0.1 * math.exp(-restart / 50.0)
        
        result_assignment, solved = walksat_once(assignment, rng, max_steps, temperature)
        
        if solved:
            return result_assignment
        
        # Track best
        final_unsat = count_unsat(result_assignment)
        if final_unsat < best_unsat:
            best_unsat = final_unsat
            best_assignment = list(result_assignment)
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
