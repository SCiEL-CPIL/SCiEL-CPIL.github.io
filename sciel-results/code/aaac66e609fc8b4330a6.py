"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 325.0  # Leave margin for overhead (evaluator kills at 330s)
    
    m = len(clauses)
    if m == 0:
        return [True] * n
        
    # Preprocess clauses into numpy arrays for fast evaluation
    # clause_data: shape (m, 3), values are +/-1
    # clause_vars: shape (m, 3), values are var indices (0-based)
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Precompute sign_bool for faster satisfaction check
    sign_bool = (clause_data == 1)  # (m, 3) bool array
    
    # Build variable-to-clauses mapping for incremental updates
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_clauses[v].append(i)
    
    def compute_satisfaction(assignment):
        """Compute boolean array of satisfied clauses from scratch."""
        vals = assignment[clause_vars]  # (m, 3) bools
        literal_true = (vals == sign_bool)  # (m, 3)
        return np.any(literal_true, axis=1)  # (m,) bools
    
    def walk_sat(restart_assignment, max_flips, seed):
        """Run WalkSAT with incremental clause tracking."""
        rng = random.Random(seed)
        assignment = restart_assignment.copy()
        
        # Initialize satisfaction state
        sat = compute_satisfaction(assignment)
        unsat_count = int(np.sum(~sat))
        
        if unsat_count == 0:
            return assignment.tolist()
        
        # Track which clauses are unsatisfied for quick access
        unsat_set = set(np.where(~sat)[0].tolist())
        
        for step in range(max_flips):
            if time.time() - start_time > timeout - 2:
                return None
            
            if unsat_count == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            clause_idx = list(unsat_set)[rng.randint(0, len(unsat_set) - 1)]
            
            # Choose variable to flip
            # 15% greedy: pick var that appears in most unsat clauses within this clause
            # 85% random: pick random var from clause
            if rng.random() < 0.15:
                # Count how many unsat clauses each var in this clause appears in
                clause_v = clause_vars[clause_idx]
                best_var = clause_v[0]
                best_score = -1
                for v in clause_v:
                    # Count unsat clauses containing this variable
                    score = 0
                    for ci in var_clauses[v]:
                        if ci in unsat_set:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_var = v
                var_to_flip = best_var
            else:
                var_to_flip = clause_vars[clause_idx, rng.randint(0, 2)]
            
            # Flip the variable
            old_val = assignment[var_to_flip]
            new_val = not old_val
            assignment[var_to_flip] = new_val
            
            # Incrementally update satisfaction for affected clauses
            for ci in var_clauses[var_to_flip]:
                was_sat = sat[ci]
                # Recompute this clause
                v_vals = assignment[clause_vars[ci]]
                is_now_sat = bool(np.any(v_vals == sign_bool[ci]))
                sat[ci] = is_now_sat
                
                if was_sat and not is_now_sat:
                    unsat_count += 1
                    unsat_set.add(ci)
                elif not was_sat and is_now_sat:
                    unsat_count -= 1
                    unsat_set.discard(ci)
        
        # Final check
        if unsat_count == 0:
            return assignment.tolist()
        
        return None
    
    # Main search loop with restarts
    best_result = None
    rng = random.Random(0)
    restart_count = 0
    
    while time.time() - start_time < timeout - 10:
        # Create a random initial assignment
        initial = np.array([rng.random() < 0.5 for _ in range(n)])
        
        # Run WalkSAT with flip budget
        max_flips = 10000
        result = walk_sat(initial, max_flips, seed=restart_count * 12345 + 67890)
        
        if result is not None:
            best_result = result
            break
        
        restart_count += 1
    
    if best_result is not None:
        return best_result
    
    # If we didn't find a solution, return a random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
