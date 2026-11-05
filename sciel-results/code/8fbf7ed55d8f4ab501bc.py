"""Improved planted 3-SAT solver with WalkSAT + clause weighting + restarts + deadline guard."""
import random
import time

# EVOLVE-BLOCK-START

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: use 280s out of 330s to be safe
    deadline = time.time() + 280.0
    
    # Precompute clause literal indices and signs for speed
    clause_idx = [[abs(l) - 1 for l in c] for c in clauses]
    clause_sign = [[(l > 0) for l in c] for c in clauses]
    
    # Track clause weights for weighted WalkSAT
    weights = [1.0] * m
    
    # Precompute variable-to-clause index map
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l) - 1].append(i)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Multiple restarts with increasing step budgets
    restart_budgets = [2000, 5000, 10000, 20000, 40000, 80000, 150000]
    rng_seed = 42
    
    for restart in range(len(restart_budgets)):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(rng_seed + restart * 7919)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause weights for this restart
        w = [1.0] * m
        
        step_budget = restart_budgets[restart]
        
        for step in range(step_budget):
            if time.time() > deadline - 2.0:
                break
            
            # Find unsatisfied clauses (using weighted selection)
            unsat_indices = []
            unsat_weights = []
            for ci in range(m):
                sat = False
                for j in range(3):
                    if assignment[clause_idx[ci][j]] == clause_sign[ci][j]:
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(ci)
                    unsat_weights.append(w[ci])
            
            if not unsat_indices:
                return assignment
            
            # Track best
            cur_unsat = len(unsat_indices)
            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_assignment = assignment[:]
            
            # Select clause weighted
            total_w = sum(unsat_weights)
            if total_w <= 0:
                total_w = 1.0
                unsat_weights = [1.0] * len(unsat_indices)
            
            r = rng.random() * total_w
            cum = 0.0
            chosen_ci = unsat_indices[0]
            for idx, wt in enumerate(unsat_weights):
                cum += wt
                if r <= cum:
                    chosen_ci = unsat_indices[idx]
                    break
            
            # Count how many literals in chosen clause are satisfied
            ci = chosen_ci
            sat_count = 0
            for j in range(3):
                if assignment[clause_idx[ci][j]] == clause_sign[ci][j]:
                    sat_count += 1
            
            # Weighted flip: prefer flipping unsatisfied literals (make clause closer to sat)
            # With probability p, flip an unsatisfied literal; otherwise flip a satisfied one
            # Classic WalkSAT: flip random literal, but we bias
            p = 0.75  # probability to flip an unsatisfied literal
            if sat_count < 3:
                # There are unsatisfied literals
                unsat_lits = []
                sat_lits = []
                for j in range(3):
                    if assignment[clause_idx[ci][j]] == clause_sign[ci][j]:
                        sat_lits.append(j)
                    else:
                        unsat_lits.append(j)
                
                if unsat_lits and (rng.random() < p or not sat_lits):
                    j = rng.choice(unsat_lits)
                else:
                    j = rng.choice(sat_lits) if sat_lits else rng.choice(unsat_lits)
            else:
                # All satisfied (shouldn't happen for unsat clause, but safety)
                j = rng.randint(0, 2)
            
            var = clause_idx[ci][j]
            assignment[var] = not assignment[var]
            
            # Update weights: increase weight of clauses that became unsatisfied due to this flip
            for ci2 in var_to_clauses[var]:
                # Check if clause ci2 is now unsatisfied
                sat2 = False
                for jj in range(3):
                    if assignment[clause_idx[ci2][jj]] == clause_sign[ci2][jj]:
                        sat2 = True
                        break
                if not sat2:
                    w[ci2] *= 1.05  # small increase
                else:
                    w[ci2] = max(1.0, w[ci2] * 0.98)  # decay
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n

# EVOLVE-BLOCK-END
