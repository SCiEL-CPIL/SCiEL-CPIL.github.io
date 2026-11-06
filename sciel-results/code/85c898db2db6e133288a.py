# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    deadline = time.time() + 280  # 300s limit, 20s safety margin
    
    # Preprocess: convert to 0-based, build incidence structures
    # For each clause, store (var_idx, sign) where sign is +1 or -1
    # We'll use a vectorized approach with NumPy
    
    # Convert clauses to arrays
    # Each clause: list of 3 literals
    # literal l: var = abs(l)-1, sign = 1 if l>0 else -1
    
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
    
    # Build variable-to-clause incidence
    var_to_clauses = [[] for _ in range(n)]
    var_to_positions = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append(i)
            var_to_positions[v].append(j)
    
    # Convert to arrays for faster access
    var_to_clauses_arr = [np.array(c, dtype=np.int32) for c in var_to_clauses]
    var_to_positions_arr = [np.array(p, dtype=np.int8) for p in var_to_positions]
    
    def count_satisfied(assignment):
        """Count satisfied clauses given assignment (numpy array of 0/1)."""
        # assignment[v] is 1 if var v is true, 0 if false
        # For literal with sign s: satisfied if assignment[var] == (s==1)
        # i.e., assignment[var] * s == 1, or (assignment[var] if s=1, 1-assignment[var] if s=-1)
        
        # Build a matrix: for each clause and each literal position, 
        # the literal value given the assignment
        # lit_val[i,j] = assignment[clause_vars[i,j]] if clause_signs[i,j]==1 else 1-assignment[clause_vars[i,j]]
        
        # Vectorized: 
        # lit_vals = np.where(clause_signs == 1, assignment[clause_vars], 1 - assignment[clause_vars])
        # This is a fancy index: assignment[clause_vars] gives (m,3) array
        
        a = assignment[clause_vars]  # (m, 3) - value of each variable in each clause
        lit_vals = np.where(clause_signs == 1, a, 1 - a)  # (m, 3)
        # Clause is satisfied if any literal is true (1)
        clause_sat = np.any(lit_vals == 1, axis=1)  # (m,)
        return np.sum(clause_sat), lit_vals
    
    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        a = assignment[clause_vars]
        lit_vals = np.where(clause_signs == 1, a, 1 - a)
        clause_sat = np.any(lit_vals == 1, axis=1)
        return np.all(clause_sat)
    
    # Strategy: Multi-restart WalkSAT with vectorized evaluation
    # Plus a spectral/glow approach for the planted structure
    
    rng = random.Random(42)
    
    best_assignment = None
    best_score = -1
    
    # Try a few different initializations and search strategies
    
    # Strategy 1: Random restart WalkSAT with adaptive noise
    max_restarts = 50
    steps_per_restart = 3000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        if restart == 0:
            # Planted solution is likely biased; try balanced
            assignment = np.random.RandomState(restart).rand(n) < 0.5
            assignment = assignment.astype(np.int8)
        else:
            # Perturb from best or random
            if best_assignment is not None and rng.random() < 0.5:
                assignment = best_assignment.copy()
                # Flip a few random variables
                n_flips = rng.randint(3, 10)
                flip_vars = rng.sample(range(n), n_flips)
                assignment[flip_vars] = 1 - assignment[flip_vars]
            else:
                assignment = (np.random.RandomState(restart).rand(n) < 0.5).astype(np.int8)
        
        # WalkSAT with vectorized clause evaluation
        noise = 0.55  # probability of choosing random literal vs. most-affected
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return assignment.tolist() if best_assignment is None else best_assignment.tolist()
            
            # Count unsatisfied clauses
            a = assignment[clause_vars]
            lit_vals = np.where(clause_signs == 1, a, 1 - a)
            clause_sat = np.any(lit_vals == 1, axis=1)
            unsat_count = m - np.sum(clause_sat)
            
            if unsat_count == 0:
                return assignment.tolist()
            
            # If this is better than best, update
            if best_score < m - unsat_count:
                best_score = m - unsat_count
                best_assignment = assignment.copy()
            
            # Find unsatisfied clauses
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # For this clause, find which literal to flip
            # Compute for each of the 3 literals: how many other unsatisfied clauses would become satisfied if we flip this variable
            # Actually, in WalkSAT: pick a literal from the clause, flip its variable
            # With noise: with prob (1-noise), pick the literal that affects the most other clauses
            
            lit_idx = rng.randint(0, 2)
            
            # Better: compute impact for each literal in the clause
            impacts = []
            for j in range(3):
                var = clause_vars[clause_idx, j]
                # If we flip var, how many currently-unsatisfied clauses (other than this one) become satisfied?
                # A clause c is affected by var if var appears in c
                # After flipping, check if clause c becomes satisfied
                # This is expensive; approximate by counting clauses where this literal is the only false one
                # Simpler: just use random selection with some bias
                
                # Count how many unsatisfied clauses contain this variable
                vcl = var_to_clauses_arr[var]
                # Check which of these are currently unsatisfied
                if len(vcl) > 0:
                    va = assignment[vcl]  # values of this var in those clauses... no, this is wrong
                    # vcl are clause indices. For each such clause, check if it's unsatisfied
                    # and if flipping var would satisfy it
                    # This is getting complex. Let's use a simpler heuristic.
                    impacts.append(len(vcl))
                else:
                    impacts.append(0)
            
            # Choose literal: with prob (1-noise), pick the one with max impact; else random
            if rng.random() < (1 - noise):
                lit_idx = int(np.argmax(impacts))
            
            var_to_flip = clause_vars[clause_idx, lit_idx]
            assignment[var_to_flip] = 1 - assignment[var_to_flip]
    
    # Strategy 2: If WalkSAT didn't find a solution, try a spectral approach
    if time.time() < deadline:
        # Build the constraint matrix for a spectral/glow approach
        # For each clause (l1 OR l2 OR l3), we want to maximize the number satisfied
        # This can be formulated as: for each clause, max(lit1, lit2, lit3)
        # Approximate with: lit1 + lit2 + lit3 - (lit1*lit2 + lit1*lit3 + lit2*lit3) + lit1*lit2*lit3
        # But this is complex. Instead, use a simpler approach:
        # Minimize sum of (1 - max(lit_i))^2 which is the number of unsatisfied clauses
        # Or use a continuous relaxation
        
        # Let's try a simple gradient-based approach on the continuous relaxation
        # x in [0,1]^n, minimize sum over clauses of prod(1 - lit_val)
        # where lit_val = x[var] if sign=+1, (1-x[var]) if sign=-1
        
        # Actually, let's try a different approach: use the structure of planted SAT
        # The planted solution is a specific assignment. Let's try to find it.
        
        # Try a few random assignments and local search
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            x = np.random.RandomState(1000 + attempt).rand(n)
            
            # Gradient descent on the continuous relaxation
            for gd_step in range(500):
                if time.time() > deadline:
                    break
                
                # Compute gradient
                # For each clause, the contribution to the loss is prod(1 - lit_val_i)
                # d/dx_v of prod(1 - lit_val_i) for v in clause:
                # = -sign_v * prod_{j != pos_v}(1 - lit_val_j) * (1 - lit_val_v) / (1 - lit_val_v)
                # Wait, let me redo this.
                
                # lit_val for literal (v, s) is: x[v] if s=+1, (1-x[v]) if s=-1
                # 1 - lit_val = 1 - x[v] if s=+1, x[v] if s=-1
                # Let's define: for literal (v, s), let w = x[v] if s=+1, 1-x[v] if s=-1
                # Then 1 - w = 1 - x[v] if s=+1, x[v] if s=-1
                # The clause is unsatisfied when all w_i = 0, i.e., all (1-w_i) = 1
                # Loss for clause = prod(1 - w_i)
                
                # d/dx_v [prod_i (1 - w_i)] where w depends on x_v
                # If v appears in the clause with sign s:
                # w_v = x_v if s=+1, 1-x_v if s=-1
                # 1 - w_v = 1 - x_v if s=+1, x_v if s=-1
                # d(1-w_v)/dx_v = -1 if s=+1, +1 if s=-1 = -s
                # So d/dx_v [prod_i (1-w_i)] = -s * prod_{i != v} (1 - w_i)
                
                # Total loss = sum over clauses of prod_i (1 - w_i)
                # Gradient = sum over clauses containing v of -s_v * prod_{i in clause, i != v} (1 - w_i)
                
                # Compute (1 - w) for all literals
                a = x[clause_vars]  # (m, 3)
                one_minus_w = np.where(clause_signs == 1, 1 - a, a)  # (m, 3)
                
                # For each clause, compute prod of (1-w) for all 3 literals
                # And for each literal position, the product excluding that position
                # prod_all = one_minus_w[:,0] * one_minus_w[:,1] * one_minus_w[:,2]
                # prod_excl_0 = one_minus_w[:,1] * one_minus_w[:,2]
                # prod_excl_1 = one_minus_w[:,0] * one_minus_w[:,2]
                # prod_excl_2 = one_minus_w[:,0] * one_minus_w[:,1]
                
                prod_excl = np.ones((m, 3), dtype=np.float64)
                prod_excl[:, 0] = one_minus_w[:, 1] * one_minus_w[:, 2]
                prod_excl[:, 1] = one_minus_w[:, 0] * one_minus_w[:, 2]
                prod_excl[:, 2] = one_minus_w[:, 0] * one_minus_w[:, 1]
                
                # Gradient contribution: for each clause and each literal position,
                # d/dx_v = -sign * prod_excl
                # Accumulate per variable
                grad = np.zeros(n, dtype=np.float64)
                # For each clause i and position j:
                # var = clause_vars[i,j], sign = clause_signs[i,j]
                # grad[var] += -sign * prod_excl[i,j]
                
                # Vectorized: use np.add.at
                grad_contrib = -clause_signs * prod_excl  # (m, 3)
                vars_flat = clause_vars.ravel()  # (3m,)
                grad_contrib_flat = grad_contrib.ravel()  # (3m,)
                np.add.at(grad, vars_flat, grad_contrib_flat)
                
                # Update
                lr = 0.1 / (1 + gd_step * 0.01)
                x -= lr * grad
                x = np.clip(x, 0, 1)
                
                # Check if rounded assignment satisfies
                if gd_step % 50 == 0:
                    x_rounded = (x > 0.5).astype(np.int8)
                    a_r = x_rounded[clause_vars]
                    lit_vals_r = np.where(clause_signs == 1, a_r, 1 - a_r)
                    clause_sat_r = np.any(lit_vals_r == 1, axis=1)
                    if np.all(clause_sat_r):
                        return x_rounded.tolist()
            
            # After gradient descent, try the rounded solution
            x_rounded = (x > 0.5).astype(np.int8)
            a_r = x_rounded[clause_vars]
            lit_vals_r = np.where(clause_signs == 1, a_r, 1 - a_r)
            clause_sat_r = np.any(lit_vals_r == 1, axis=1)
            if np.all(clause_sat_r):
                return x_rounded.tolist()
    
    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return (np.random.RandomState(0).rand(n) < 0.5).tolist()

# EVOLVE-BLOCK-END
