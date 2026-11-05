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
    
    def compute_flip_cost(var_idx, assignment):
        """Compute the change in unsatisfied clause count if we flip var_idx.
        Returns positive if flipping increases unsat count, negative if it decreases."""
        cost = 0
        for c_idx in var_to_clauses[var_idx]:
            vars_, sigs_ = clause_data[c_idx]
            # Find the sign required for this variable in this clause
            # var_idx should appear in vars_
            # Count satisfied and unsatisfied literals in this clause
            sat_count = 0
            unsat_count = 0
            for v, s in zip(vars_, sigs_):
                if assignment[v] == s:
                    sat_count += 1
                else:
                    unsat_count += 1
            
            # Check if var_idx is currently satisfied in this clause
            # Find the position of var_idx in vars_
            var_is_sat = False
            for v, s in zip(vars_, sigs_):
                if v == var_idx:
                    var_is_sat = (assignment[v] == s)
                    break
            
            if var_is_sat:
                # Currently satisfied. If we flip, it becomes unsatisfied.
                # If it was the only satisfied literal (sat_count == 1), clause becomes unsat: cost +1
                if sat_count == 1:
                    cost += 1
            else:
                # Currently unsatisfied. If we flip, it becomes satisfied.
                # If it was the only unsatisfied literal (unsat_count == 1), clause becomes sat: cost -1
                if unsat_count == 1:
                    cost -= 1
        
        return cost
    
    def walksat_start(assignment, steps, p):
        """Perform WalkSAT steps with cost-based flipping. Returns (solved, new_assignment)."""
        for _ in range(steps):
            if time.time() > deadline:
                return False, assignment
            
            # Find all unsatisfied clauses
            unsat_clauses = []
            for i in range(num_clauses):
                if not is_satisfied(i, assignment):
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return True, assignment
            
            # Select clause with minimum unsatisfied literals (closest to satisfied)
            min_unsat_lits = 3
            best_c_idx = unsat_clauses[0]
            for c_idx in unsat_clauses:
                vars_, sigs_ = clause_data[c_idx]
                u = 0
                for v, s in zip(vars_, sigs_):
                    if assignment[v] != s:
                        u += 1
                if u < min_unsat_lits:
                    min_unsat_lits = u
                    best_c_idx = c_idx
                    if min_unsat_lits == 1:
                        break
            
            vars_, sigs_ = clause_data[best_c_idx]
            
            # Compute cost for each literal in the selected clause
            costs = []
            for lit_pos in range(3):
                var_to_flip = vars_[lit_pos]
                c = compute_flip_cost(var_to_flip, assignment)
                costs.append(c)
            
            # With probability p, pick the literal with minimum cost
            # With probability (1-p), pick randomly
            if random.random() < p:
                min_cost = min(costs)
                # Pick randomly among literals with minimum cost
                min_indices = [i for i, c in enumerate(costs) if c == min_cost]
                lit_idx = random.choice(min_indices)
            else:
                lit_idx = random.randrange(3)
            
            var_to_flip = vars_[lit_idx]
            assignment[var_to_flip] = not assignment[var_to_flip]
        
        # Check if solved
        for i in range(num_clauses):
            if not is_satisfied(i, assignment):
                return False, assignment
        return True, assignment
    
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
        solved, current_assign = walksat_start(current_assign, steps_per_restart, 0.5)
        
        unsat_count = count_unsat(current_assign)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = current_assign[:]
        
        if solved:
            return current_assign
    
    # If we ran out of time or restarts, try one final long run with best assignment
    if time.time() < deadline - 1.0 and best_assignment:
        solved, current_assign = walksat_start(best_assignment, 5000, 0.5)
        if solved:
            return current_assign
        unsat_count = count_unsat(current_assign)
        if unsat_count < best_unsat_count:
            best_assignment = current_assign[:]
    
    return best_assignment
# EVOLVE-BLOCK-END
