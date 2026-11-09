"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Precompute clause structure for fast access
    # clauses is a list of lists of signed 1-based literals
    # Convert to 0-based indices and store sign separately for speed
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            var = abs(l) - 1
            pos = l > 0
            cd.append((var, pos))
        clause_data.append(cd)
    
    # Precompute variable-to-clause incidence for faster gain computation
    # var_clauses[var] = list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci, cd in enumerate(clause_data):
        for var, pos in cd:
            var_clauses[var].append((ci, pos))
    
    deadline = time.time() + 290  # Safety margin
    
    rng = random.Random(42)
    
    def is_clause_sat(clause_idx, assignment):
        for var, pos in clause_data[clause_idx]:
            if assignment[var] == pos:
                return True
        return False
    
    def count_satisfied(assignment):
        count = 0
        for i in range(num_clauses):
            if is_clause_sat(i, assignment):
                count += 1
        return count
    
    def walk_sat_trial(assignment, max_iters, deadline):
        """Run WalkSAT with score-based literal selection."""
        cur_count = count_satisfied(assignment)
        
        if cur_count == num_clauses:
            return assignment, cur_count
        
        no_improve = 0
        restart_threshold = 1000
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_indices = []
            for i in range(num_clauses):
                if not is_clause_sat(i, assignment):
                    unsat_indices.append(i)
            
            if not unsat_indices:
                return assignment, num_clauses
            
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat_indices)
            clause = clause_data[clause_idx]
            
            # Score each literal: how many clauses would become satisfied
            # if we flip this variable to make this literal true
            scores = []
            for var, pos in clause:
                # If we set assignment[var] = pos, which clauses gain satisfaction?
                # Only consider clauses that are currently unsatisfied
                # and contain var with the same polarity
                gain = 0
                for ci, p in var_clauses[var]:
                    if p == pos and not is_clause_sat(ci, assignment):
                        gain += 1
                scores.append(gain)
            
            # Pick literal with highest score, with some randomization
            max_score = max(scores)
            # Use a soft selection: prefer higher scores but allow some randomness
            # Weighted random selection
            total_weight = sum(s + 1 for s in scores)
            r = rng.random() * total_weight
            cumulative = 0
            chosen_idx = 0
            for i, s in enumerate(scores):
                cumulative += s + 1
                if r <= cumulative:
                    chosen_idx = i
                    break
            
            var, pos = clause[chosen_idx]
            assignment[var] = pos
            
            new_count = count_satisfied(assignment)
            if new_count > cur_count:
                cur_count = new_count
                no_improve = 0
                if cur_count == num_clauses:
                    return assignment, cur_count
            else:
                no_improve += 1
            
            if no_improve >= restart_threshold:
                break
        
        return assignment, cur_count
    
    best_assignment = [False] * n
    best_count = -1
    
    max_restarts = 200
    iters_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        assignment, cur_count = walk_sat_trial(assignment, iters_per_restart, deadline)
        
        if cur_count > best_count:
            best_count = cur_count
            best_assignment = assignment[:]
        
        if best_count == num_clauses:
            return best_assignment
    
    return best_assignment
# EVOLVE-BLOCK-END
