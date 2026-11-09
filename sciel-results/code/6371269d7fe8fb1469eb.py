"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # safety margin before 330s kill
    rng = random.Random(0)
    
    # Pre-compute clause literal indices for speed
    clause_indices = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    num_clauses = len(clauses)
    
    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        cnt = 0
        for idxs in clause_indices:
            sat = False
            for var_idx, is_pos in idxs:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def is_satisfied(clause_idxs, assignment):
        for var_idx, is_pos in clause_idxs:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    # Multi-restart WalkSAT with clause weighting
    best_assignment = None
    best_unsat = num_clauses + 1
    num_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # WalkSAT loop
        steps_per_restart = 10000
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break
            
            # Find unsatisfied clauses and count false literals per variable
            unsat_clauses = []
            false_counts = [0] * n  # number of false literals in unsat clauses per var
            true_counts = [0] * n   # number of true literals in unsat clauses per var
            
            for ci, clause_idxs in enumerate(clause_indices):
                sat = False
                false_lits = []
                for var_idx, is_pos in clause_idxs:
                    if assignment[var_idx] == is_pos:
                        sat = True
                        break
                    else:
                        false_lits.append((var_idx, is_pos))
                if not sat:
                    unsat_clauses.append((ci, false_lits))
                    for var_idx, is_pos in false_lits:
                        false_counts[var_idx] += 1
                        if assignment[var_idx]:
                            true_counts[var_idx] += 1
            
            if not unsat_clauses:
                best_assignment = assignment[:]
                best_unsat = 0
                break
            
            # Track best
            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
            
            # Weighted clause selection: prefer clauses with fewer false literals
            # (closer to being satisfied)
            weights = [1.0 / (len(fl) ** 2) for _, fl in unsat_clauses]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0.0
            chosen = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = i
                    break
            
            _, false_lits = unsat_clauses[chosen]
            
            # Variable selection: weighted by (true_count / (true_count + false_count + 1))
            # Prefer flipping variables that would satisfy more other unsat clauses
            var_weights = []
            for var_idx, is_pos in false_lits:
                # If we flip this var, it becomes true for this clause
                # Benefit: it might satisfy other unsat clauses where it's false
                benefit = true_counts[var_idx]
                cost = false_counts[var_idx]
                # Soft weighting
                w = benefit / (benefit + cost + 1) + 0.1
                var_weights.append(w)
            
            total_vw = sum(var_weights)
            rv = rng.random() * total_vw
            cum = 0.0
            chosen_var_idx = false_lits[0][0]
            chosen_is_pos = false_lits[0][1]
            for j, vw in enumerate(var_weights):
                cum += vw
                if rv <= cum:
                    chosen_var_idx = false_lits[j][0]
                    chosen_is_pos = false_lits[j][1]
                    break
            
            # Flip the variable
            assignment[chosen_var_idx] = not assignment[chosen_var_idx]
        
        if best_unsat == 0:
            break
    
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
# EVOLVE-BLOCK-END
