import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    rng = random.Random()
    
    # Pre-process
    # clauses: list of lists of signed literals (1-based)
    # Convert to 0-based indices and boolean polarity
    processed_clauses = []
    var_to_clauses = [[] for _ in range(n)]
    
    for c_idx, c in enumerate(clauses):
        pc = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pc.append((var_idx, is_pos))
            var_to_clauses[var_idx].append(c_idx)
        processed_clauses.append(pc)
        
    m = len(processed_clauses)
    
    def init_counts(assignment):
        """Initialize clause_sat_count and unsat_clauses."""
        clause_sat_count = [0] * m
        for c_idx, c in enumerate(processed_clauses):
            count = 0
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    count += 1
            clause_sat_count[c_idx] = count
        unsat_clauses = [i for i in range(m) if clause_sat_count[i] == 0]
        return clause_sat_count, unsat_clauses

    best_assignment = None
    best_unsat_count = m + 1
    
    max_restarts = 5000
    restart_limit = 20000 # Steps per restart
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize assignment
        if best_assignment is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Phase saving with perturbation
            assignment = list(best_assignment)
            # Perturb 5% of variables
            for i in range(n):
                if rng.random() < 0.05:
                    assignment[i] = rng.random() < 0.5
                    
        clause_sat_count, unsat_clauses = init_counts(assignment)
        current_unsat = len(unsat_clauses)
        
        if current_unsat == 0:
            return assignment
            
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = list(assignment)
            
        for step in range(restart_limit):
            if time.time() > deadline:
                break
                
            if not unsat_clauses:
                return assignment
                
            # Pick random unsatisfied clause
            c_idx = rng.choice(unsat_clauses)
            clause = processed_clauses[c_idx]
            
            # Pick literal to flip
            # Heuristic: Min-conflicts (approximate)
            # Pick literal that is in the most unsatisfied clauses?
            # Or just random?
            # Random WalkSAT is surprisingly effective.
            # Let's use a mix: 50% random, 50% min-conflicts (variable in most unsat clauses)
            
            if rng.random() < 0.5:
                var_idx, is_pos = rng.choice(clause)
            else:
                # Find variable in this clause that appears in the most unsat clauses
                max_count = -1
                best_var = -1
                for var_idx, is_pos in clause:
                    # Count how many unsat clauses contain this variable
                    # This is O(U * 3) which is expensive if U is large.
                    # Optimization: Pre-compute? No, U changes.
                    # Alternative: Just pick random. The min-conflicts overhead might be too high.
                    # Let's stick to random for speed, or a very cheap heuristic.
                    # Cheap heuristic: Flip the variable that is true in the fewest other unsat clauses?
                    # Actually, just random is standard for WalkSAT.
                    best_var = var_idx
                    break # Just pick first? No, let's pick random to avoid bias.
                var_idx, is_pos = rng.choice(clause)
                
            # Flip variable
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val
            
            # Update clause counts
            for c in var_to_clauses[var_idx]:
                # Determine if var_idx was satisfying c
                # We need to know the polarity in this clause
                # Find the entry in processed_clauses[c] for var_idx
                # Since clauses are small (3), linear search is fine
                was_satisfying = False
                is_satisfying_now = False
                for v, p in processed_clauses[c]:
                    if v == var_idx:
                        was_satisfying = (old_val == p)
                        is_satisfying_now = (new_val == p)
                        break
                        
                if was_satisfying and not is_satisfying_now:
                    # Clause becomes less satisfied
                    clause_sat_count[c] -= 1
                    if clause_sat_count[c] == 0:
                        unsat_clauses.append(c)
                elif not was_satisfying and is_satisfying_now:
                    # Clause becomes more satisfied
                    clause_sat_count[c] += 1
                    if clause_sat_count[c] == 1:
                        # It was unsat, now sat. Remove from unsat_clauses.
                        # List removal is O(N). U is usually small.
                        # To avoid O(N) removal, we can use a set or just filter later.
                        # For now, try to remove.
                        try:
                            unsat_clauses.remove(c)
                        except ValueError:
                            pass # Should not happen if logic is correct
                else:
                    # No change in satisfaction status for this clause
                    pass
                    
            # Update best
            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = list(assignment)
                
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
