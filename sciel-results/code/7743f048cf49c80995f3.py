"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    timeout = 315.0  # Leave margin for overhead
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays
    # clause_sgn: shape (m, 3), values are +1 or -1
    # clause_var: shape (m, 3), values are var indices (0-based)
    clause_sgn = np.empty((m, 3), dtype=np.int8)
    clause_var = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_sgn[i, j] = 1 if lit > 0 else -1
            clause_var[i, j] = abs(lit) - 1
    
    # Build variable-to-clause incidence
    # For each variable, list of (clause_idx, position)
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_var[i, j]
            var_clauses[v].append((i, j))
    
    # Precompute for each variable: array of clause indices and signs
    var_clause_idx = [np.array([x[0] for x in lst], dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in var_clauses]
    var_clause_pos = [np.array([x[1] for x in lst], dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in var_clauses]
    
    # Precompute literal frequency for heuristic initial assignment
    # For each variable, count how often it appears positively vs negatively
    pos_count = np.zeros(n, dtype=np.int32)
    neg_count = np.zeros(n, dtype=np.int32)
    for i in range(m):
        for j in range(3):
            v = clause_var[i, j]
            if clause_sgn[i, j] == 1:
                pos_count[v] += 1
            else:
                neg_count[v] += 1
    
    def evaluate(assignment):
        """Return unsatisfied clause indices."""
        vals = clause_sgn * (2.0 * assignment[clause_var] - 1.0)
        sat = np.any(vals > 0, axis=1)
        return np.where(~sat)[0]
    
    def make_heuristic_initial(rng):
        """Create initial assignment using literal frequency heuristic with noise."""
        # Set variable to True if it appears more positively, False if more negatively
        # Add some randomness to break ties and explore
        assignment = np.zeros(n, dtype=np.float64)
        for v in range(n):
            p = pos_count[v]
            q = neg_count[v]
            if p == q:
                assignment[v] = 1.0 if rng.rand() < 0.5 else 0.0
            elif p > q:
                # Bias toward True but with some probability of False
                prob_true = 0.7 + 0.2 * (p - q) / (p + q)
                assignment[v] = 1.0 if rng.rand() < prob_true else 0.0
            else:
                # Bias toward False but with some probability of True
                prob_true = 0.3 - 0.2 * (q - p) / (p + q)
                assignment[v] = 1.0 if rng.rand() < max(0.05, prob_true) else 0.0
        return assignment
    
    # WalkSAT with restarts and adaptive parameters
    rng = np.random.RandomState(42)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    max_restarts = 500
    base_steps = 5000
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Use heuristic initial assignment for first few restarts, then random
        if restart < 10:
            assignment = make_heuristic_initial(rng)
        else:
            assignment = (rng.rand(n) < 0.5).astype(np.float64)
        
        unsat = evaluate(assignment)
        unsat_count = len(unsat)
        
        if unsat_count == 0:
            return assignment.astype(bool).tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # Adaptive step budget
        steps = base_steps + restart * 500
        
        # WalkSAT loop
        for step in range(steps):
            if time.time() - start_time > timeout:
                break
            
            if len(unsat) == 0:
                return assignment.astype(bool).tolist()
            
            if len(unsat) < best_unsat_count:
                best_unsat_count = len(unsat)
                best_assignment = assignment.copy()
            
            # Choose flip
            # Use score-based selection with some randomness
            clause_idx = unsat[rng.randint(len(unsat))]
            
            # Try all 3 variables in the clause, pick best
            best_v = -1
            best_delta = float('-inf')
            candidates = []
            
            for j in range(3):
                v = clause_var[clause_idx, j]
                cidx = var_clause_idx[v]
                cpos = var_clause_pos[v]
                
                if len(cidx) == 0:
                    continue
                
                signs = clause_sgn[cidx, cpos]
                other_vars = clause_var[cidx]
                
                all_vals = clause_sgn[cidx] * (2.0 * assignment[other_vars] - 1.0)
                currently_sat = np.any(all_vals > 0, axis=1)
                
                new_all_vals = all_vals.copy()
                rows = np.arange(len(cidx))
                new_all_vals[rows, cpos] *= -1
                new_sat = np.any(new_all_vals > 0, axis=1)
                
                delta = np.sum(new_sat) - np.sum(currently_sat)
                
                if delta > best_delta:
                    best_delta = delta
                    best_v = v
                    candidates = [v]
                elif delta == best_delta:
                    candidates.append(v)
            
            if best_v == -1:
                v = clause_var[clause_idx, rng.randint(3)]
            elif len(candidates) > 1:
                v = candidates[rng.randint(len(candidates))]
            else:
                v = best_v
            
            assignment[v] = 1.0 - assignment[v]
            
            # Update unsat clauses efficiently using vectorized approach
            cidx = var_clause_idx[v]
            if len(cidx) == 0:
                unsat = evaluate(assignment)
            else:
                # Recompute only affected clauses
                all_vals = clause_sgn[cidx] * (2.0 * assignment[clause_var[cidx]] - 1.0)
                sat = np.any(all_vals > 0, axis=1)
                
                # Vectorized update: 
                # unsat_new = (unsat & ~in_cidx) | (cidx & ~sat)
                # Use numpy operations to avoid Python loop
                unsat_set = set(unsat.tolist())
                for k in range(len(cidx)):
                    ci = int(cidx[k])
                    if sat[k]:
                        unsat_set.discard(ci)
                    else:
                        unsat_set.add(ci)
                unsat = np.array(list(unsat_set), dtype=np.int32)
        
        # Check after walk
        if len(unsat) == 0:
            return assignment.astype(bool).tolist()
        if len(unsat) < best_unsat_count:
            best_unsat_count = len(unsat)
            best_assignment = assignment.copy()
    
    # Return best found
    if best_assignment is not None:
        return best_assignment.astype(bool).tolist()
    else:
        return (rng.rand(n) < 0.5).tolist()
# EVOLVE-BLOCK-END
