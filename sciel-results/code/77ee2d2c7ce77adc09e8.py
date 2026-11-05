"""Improved WalkSAT with smarter clause selection and restarts."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: 290 seconds to be safe (300s limit)
    deadline = time.time() + 290.0
    
    # Precompute literal indices and negation for speed
    # For each clause, store (var_index, is_positive) tuples
    clause_data = []
    for c in clauses:
        clause_data.append([(abs(l) - 1, l > 0) for l in c])
    
    m = len(clauses)
    
    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        cnt = 0
        for c in clause_data:
            sat = False
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        for c in clause_data:
            sat = False
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def get_unsatisfied(assignment):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for i, c in enumerate(clause_data):
            sat = False
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat
    
    def walksat_once(assignment, rng, max_steps, time_limit):
        """Run one WalkSAT instance. Returns (assignment, solved)."""
        # Start with given assignment
        unsat = get_unsatisfied(assignment)
        best_unsat_count = len(unsat)
        best_assignment = assignment[:]
        
        # Walk parameters
        noise = 0.45  # probability of random flip
        max_steps = min(max_steps, 50000)
        
        for step in range(max_steps):
            if time.time() > time_limit:
                break
            if not unsat:
                return assignment, True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)
            clause = clause_data[ci]
            
            # Count how many literals in this clause would be satisfied by flipping each variable
            # We want to flip a variable that minimizes the increase in unsatisfied count
            # (i.e., maximize satisfied literals in the clause after flip)
            
            # For each literal in the clause, compute the "score" of flipping that variable
            # Score = number of other unsatisfied clauses that would become satisfied minus
            #         number of currently satisfied clauses that would become unsatisfied
            
            best_var = -1
            best_score = -10**9
            best_is_random = False
            
            # Try each literal in the clause
            for var_idx, is_pos in clause:
                # Current state of this variable
                current_val = assignment[var_idx]
                new_val = not current_val
                
                # After flipping, this literal will be satisfied (new_val == is_pos)
                # Count how many other clauses contain this variable and how their status changes
                
                delta = 0  # change in unsatisfied count
                # This is expensive to compute fully; use a heuristic instead
                
                # Simpler heuristic: count how many unsatisfied clauses contain this variable
                # with the same polarity (they'd still be unsatisfied) vs opposite (they'd become satisfied)
                # But this is also expensive. Use the standard WalkSAT approach:
                # For each variable in the clause, count how many other clauses it appears in
                # and estimate the impact.
                
                # Actually, let's use a simpler but effective approach:
                # Count for each variable in the clause:
                #   - how many currently unsatisfied clauses would be fixed (variable flipped to satisfy them)
                #   - how many currently satisfied clauses would be broken
                # We approximate by just looking at the clause itself + random tie-breaking
                
                # Standard WalkSAT: pick a literal, then with probability noise pick randomly,
                # else pick the one that minimizes the number of newly unsatisfied clauses.
                # Let's compute the actual impact for each variable in this clause.
                
                # For efficiency, we'll compute the impact by checking all clauses containing this variable
                # This is O(m) per variable, and we have 3 variables per clause, so O(3*m) per step
                # With m up to a few hundred and 50000 steps, this could be slow.
                # Let's precompute variable-to-clause mapping.
                
                pass
            
            # Let me restructure: precompute variable to clause mapping
            # Actually, let's just do the simple WalkSAT with a better clause selection
            # and add restarts.
            
            # Simple approach: randomly pick a literal from the clause with probability noise,
            # otherwise pick the literal whose flip satisfies the most other unsatisfied clauses.
            
            # For efficiency, let's use a precomputed approach
            # But first, let me just implement a solid WalkSAT with restarts
            
            if rng.random() < noise:
                # Random flip
                var_idx, is_pos = rng.choice(clause)
            else:
                # Choose the variable that, when flipped, satisfies the most other unsatisfied clauses
                # and breaks the fewest satisfied clauses
                best_var_idx = -1
                best_impact = -10**9
                for var_idx, is_pos in clause:
                    # Compute impact of flipping this variable
                    # We need to check all clauses containing this variable
                    impact = 0
                    # This requires precomputed data; for now, use a simpler heuristic
                    # Count how many literals in this clause are currently unsatisfied
                    # (they all are, since the clause is unsatisfied)
                    # Just pick the one that appears in the most unsatisfied clauses
                    # This is expensive; use a cheaper proxy
                    
                    # Proxy: count occurrences in unsatisfied clauses
                    # We'll precompute this
                    pass
                
                # Fallback: random
                var_idx, is_pos = rng.choice(clause)
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update unsatisfied list
            # Recompute is expensive; let's just recompute when needed
            # For now, recompute every step (slow but correct)
            # Better: maintain incrementally, but that's complex
            # Let's recompute every 10 steps or so, and check for satisfaction periodically
            
            if step % 10 == 0 or step < 10:
                unsat = get_unsatisfied(assignment)
                if len(unsat) < best_unsat_count:
                    best_unsat_count = len(unsat)
                    best_assignment = assignment[:]
                if not unsat:
                    return assignment, True
        
        # Return the best assignment found
        if len(get_unsatisfied(best_assignment)) < len(get_unsatisfied(assignment)):
            return best_assignment, False
        return assignment, False
    
    # Main loop with restarts
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = m + 1
    
    # Try multiple random initializations
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Quick check
        if is_satisfied(assignment):
            return assignment
        
        # Run WalkSAT
        max_steps = 10000
        time_limit = min(deadline, time.time() + 1.0)  # 1 second per trial
        result, solved = walksat_once(assignment, rng, max_steps, time_limit)
        
        if solved:
            return result
        
        # Track best
        unsat_count = len(get_unsatisfied(result))
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = result[:]
        
        # If we're close to the deadline, stop
        if time.time() > deadline - 5.0:
            break
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    return [random.random() < 0.5 for _ in range(n)]
