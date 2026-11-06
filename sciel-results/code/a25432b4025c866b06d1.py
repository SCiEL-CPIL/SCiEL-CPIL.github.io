# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n
    
    rng = random.Random(42)
    
    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_ = [abs(l) - 1 for l in clause]
        signs = [l > 0 for l in clause]
        clause_vars.append(vars_)
        clause_signs.append(signs)
    
    # Build variable-to-clause index for fast lookup
    var_to_clauses = [[] for _ in range(n)]
    for i, vars_ in enumerate(clause_vars):
        for v in vars_:
            var_to_clauses[v].append(i)
    
    # Step 1: Gaussian Elimination on linear relaxation
    # Treat each clause (l1 OR l2 OR l3) as a soft constraint.
    # For planted SAT, we can try to find an assignment that satisfies the "parity" 
    # interpretation or use unit propagation style reasoning.
    
    # Better: Use a weighted approach. Initialize with 0.5, then do belief propagation
    # or use the fact that planted instances have a unique solution.
    
    # Strategy: 
    # 1. Try to deduce variable values using 1-in-3 SAT style or unit propagation
    # 2. If that fails, use random init + targeted repair
    
    # Let's try a hybrid: 
    # - First, check if any variable appears in many clauses with consistent polarity
    # - Use that to seed the assignment
    # - Then do targeted repair
    
    # Initialize assignment with a heuristic based on clause structure
    # For each variable, count how often it appears positive vs negative
    pos_count = np.zeros(n)
    neg_count = np.zeros(n)
    for clause in clauses:
        for l in clause:
            if l > 0:
                pos_count[l - 1] += 1
            else:
                neg_count[-l - 1] += 1
    
    # Heuristic: set variable to True if it appears more positively, else False
    # But this is too naive. Let's use a different approach.
    
    # Use random initialization but with a bias based on occurrence
    assignment = np.zeros(n, dtype=bool)
    for i in range(n):
        total = pos_count[i] + neg_count[i]
        if total > 0:
            # Slight bias towards the more common polarity
            p_true = 0.5 + 0.1 * (pos_count[i] - neg_count[i]) / total
            assignment[i] = rng.random() < p_true
        else:
            assignment[i] = rng.random() < 0.5
    
    # Vectorized satisfaction checking
    max_clause_len = max(len(c) for c in clauses)
    clause_matrix = np.full((num_clauses, max_clause_len), -1, dtype=int)
    clause_sign_matrix = np.zeros((num_clauses, max_clause_len), dtype=bool)
    clause_lengths = np.zeros(num_clauses, dtype=int)
    
    for i, (vars_, signs) in enumerate(zip(clause_vars, clause_signs)):
        length = len(vars_)
        clause_lengths[i] = length
        clause_matrix[i, :length] = vars_
        clause_sign_matrix[i, :length] = signs
    
    valid_mask = np.arange(max_clause_len)[None, :] < clause_lengths[:, None]
    
    def get_unsatisfied_indices():
        var_indices = clause_matrix.copy()
        var_indices[~valid_mask] = 0
        var_values = assignment[var_indices]
        lit_satisfied = var_values == clause_sign_matrix
        lit_satisfied[~valid_mask] = True
        clause_satisfied = lit_satisfied.any(axis=1)
        return np.where(~clause_satisfied)[0]
    
    # Targeted repair phase
    # Only flip variables that are in unsatisfied clauses
    max_iterations = 100000
    best_unsat_count = float('inf')
    best_assignment = assignment.copy()
    stuck_counter = 0
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        unsat_indices = get_unsatisfied_indices()
        unsat_count = len(unsat_indices)
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, reset to best
        if no_improve_count > 5000:
            assignment = best_assignment.copy()
            no_improve_count = 0
        
        # Pick a random unsatisfied clause
        clause_idx = rng.choice(unsat_indices.tolist())
        clause = clause_vars[clause_idx]
        
        # Find variables in this clause that appear in other unsatisfied clauses
        # Prioritize flipping variables that affect many unsatisfied clauses
        var_scores = {}
        for v in clause:
            # Count how many unsatisfied clauses contain this variable
            score = 0
            for ci in var_to_clauses[v]:
                if ci in set(unsat_indices.tolist()):
                    score += 1
            var_scores[v] = score
        
        # With some probability, pick the highest-scoring variable
        # With some probability, pick randomly (to escape local minima)
        if rng.random() < 0.3:
            var_idx = rng.choice(clause)
        else:
            # Pick variable with highest score, break ties randomly
            max_score = max(var_scores.values())
            candidates = [v for v in clause if var_scores[v] == max_score]
            var_idx = rng.choice(candidates)
        
        assignment[var_idx] = not assignment[var_idx]
    
    # Return best assignment found
    return best_assignment.tolist()
# EVOLVE-BLOCK-END
