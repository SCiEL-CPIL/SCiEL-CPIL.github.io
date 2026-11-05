"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Precompute clause data for speed
    clause_data = []
    for c in clauses:
        clause_vars = []
        clause_sigs = []
        for l in c:
            var_idx = abs(l) - 1
            clause_vars.append(var_idx)
            clause_sigs.append(l > 0)
        clause_data.append((clause_vars, clause_sigs))
    
    # Precompute variable to clause indices mapping for cost calculation
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_idx = abs(l) - 1
            var_to_clauses[var_idx].append(i)
    
    num_clauses = len(clauses)
    
    # Deadline guard: 300s limit, leave 5s margin
    deadline = time.time() + 295.0
    
    def is_satisfied(c_idx, assignment):
        vars_, sigs_ = clause_data[c_idx]
        for v, s in zip(vars_, sigs_):
            if assignment[v] == s:
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                cnt += 1
        return cnt
    
    def compute_flip_cost_fast(var_idx, assignment, sat_counts):
        """Compute the change in unsatisfied clause count if we flip var_idx.
        Uses precomputed sat_counts for speed.
        Returns positive if flipping increases unsat count, negative if it decreases."""
        cost = 0
        for c_idx in var_to_clauses[var_idx]:
            vars_, sigs_ = clause_data[c_idx]
            sc = sat_counts[c_idx]
            
            # Find if var_idx is currently satisfied in this clause
            var_is_sat = False
            for v, s in zip(vars_, sigs_):
                if v == var_idx:
                    var_is_sat = (assignment[v] == s)
                    break
            
            if var_is_sat:
                # Currently satisfied. If we flip, it becomes unsatisfied.
                # If it was the only satisfied literal (sc == 1), clause becomes unsat: cost +1
                if sc == 1:
                    cost += 1
            else:
                # Currently unsatisfied. If we flip, it becomes satisfied.
                # If it was the only unsatisfied literal (3 - sc == 1), clause becomes sat: cost -1
                if sc == 2:
                    cost -= 1
        
        return cost
    
    def walksat_start(assignment, steps, p, rng):
        """Perform WalkSAT steps with cost-based flipping. Returns (solved, new_assignment)."""
        # Precompute initial sat counts
        sat_counts = [0] * num_clauses
        for i in range(num_clauses):
            vars_, sigs_ = clause_data[i]
            for v, s in zip(vars_, sigs_):
                if assignment[v] == s:
                    sat_counts[i] += 1
        
        unsat_count = sum(1 for sc in sat_counts if sc == 0)
        
        for step in range(steps):
            if time.time() > deadline:
                return False, assignment, unsat_count
            
            if unsat_count == 0:
                return True, assignment, 0
            
            # Find unsatisfied clauses (sat_count == 0)
            # Select clause with minimum unsatisfied literals (i.e., maximum sat_count among unsat, which is 0)
            # Actually for unsat clauses, sat_count is 0, so all have same "closeness"
            # Pick a random unsatisfied clause
            unsat_indices = [i for i in range(num_clauses) if sat_counts[i] == 0]
            best_c_idx = rng.choice(unsat_indices)
            
            vars_, sigs_ = clause_data[best_c_idx]
            
            # Compute cost for each literal in the selected clause
            costs = []
            for lit_pos in range(3):
                var_to_flip = vars_[lit_pos]
                c = compute_flip_cost_fast(var_to_flip, assignment, sat_counts)
                costs.append(c)
            
            # With probability p, pick the literal with minimum cost
            # With probability (1-p), pick randomly
            if rng.random() < p:
                min_cost = min(costs)
                # Pick randomly among literals with minimum cost
                min_indices = [i for i, c in enumerate(costs) if c == min_cost]
                lit_idx = rng.choice(min_indices)
            else:
                lit_idx = rng.randrange(3)
            
            var_to_flip = vars_[lit_idx]
            
            # Update sat_counts incrementally
            for c_idx in var_to_clauses[var_to_flip]:
                vars_c, sigs_c = clause_data[c_idx]
                # Find the sign for this variable in this clause
                var_sign = None
                for v, s in zip(vars_c, sigs_c):
                    if v == var_to_flip:
                        var_sign = s
                        break
                
                old_val = assignment[var_to_flip]
                new_val = not old_val
                
                if old_val == var_sign:
                    # Was satisfied, now unsatisfied
                    sat_counts[c_idx] -= 1
                    if sat_counts[c_idx] == 0:
                        unsat_count += 1
                else:
                    # Was unsatisfied, now satisfied
                    sat_counts[c_idx] += 1
                    if sat_counts[c_idx] == 1:
                        unsat_count -= 1
            
            assignment[var_to_flip] = not assignment[var_to_flip]
        
        # Check if solved
        if unsat_count == 0:
            return True, assignment, 0
        return False, assignment, unsat_count
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random()
    
    # Initial random assignment
    current_assign = [rng.random() < 0.5 for _ in range(n)]
    
    # Try WalkSAT with various parameters
    max_restarts = 50
    steps_per_restart = 1000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # New random assignment for each restart
        if restart > 0:
            current_assign = [rng.random() < 0.5 for _ in range(n)]
        
        # Use p=0.5 for balanced exploration/exploitation
        solved, current_assign, unsat_count = walksat_start(current_assign, steps_per_restart, 0.5, rng)
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = current_assign[:]
        
        if solved:
            return current_assign
    
    # If we ran out of time or restarts, try one final long run with best assignment
    if time.time() < deadline - 1.0 and best_assignment:
        solved, current_assign, unsat_count = walksat_start(best_assignment, 5000, 0.5, rng)
        if solved:
            return current_assign
        if unsat_count < best_unsat_count:
            best_assignment = current_assign[:]
    
    return best_assignment
# EVOLVE-BLOCK-END
