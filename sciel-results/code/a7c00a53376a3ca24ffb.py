"""Simulated Annealing based 3-SAT solver."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Random initial assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Pre-compute clause literals for faster access
    # clauses is a list of lists of signed integers (1-based)
    
    # Helper to check if a clause is satisfied
    def is_satisfied(clause, assignment):
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False
    
    # Count unsatisfied clauses
    def count_unsat(assignment, clauses):
        count = 0
        for clause in clauses:
            if not is_satisfied(clause, assignment):
                count += 1
        return count
    
    # Initial energy
    current_energy = count_unsat(assignment, clauses)
    
    if current_energy == 0:
        return assignment
    
    # Simulated Annealing parameters
    T_start = 10.0
    T_end = 0.001
    cooling_rate = 0.999
    
    # Number of iterations
    max_iterations = 50000
    
    T = T_start
    
    for iteration in range(max_iterations):
        if current_energy == 0:
            break
        
        # Choose a random variable to flip
        var = rng.randint(0, n - 1)
        
        # Calculate delta energy
        # We need to see how many clauses contain this variable and how flipping affects them
        # For efficiency, we can pre-compute which clauses contain each variable,
        # but for simplicity and since n=150, we can just recompute affected clauses.
        
        old_value = assignment[var]
        new_value = not old_value
        
        delta = 0
        for clause in clauses:
            # Check if clause contains this variable
            contains_var = False
            for lit in clause:
                if abs(lit) - 1 == var:
                    contains_var = True
                    break
            
            if not contains_var:
                continue
            
            # Check if clause was satisfied before
            was_satisfied = False
            for lit in clause:
                idx = abs(lit) - 1
                if idx == var:
                    # Use old value
                    if old_value == (lit > 0):
                        was_satisfied = True
                        break
                else:
                    if assignment[idx] == (lit > 0):
                        was_satisfied = True
                        break
            
            # Check if clause is satisfied after
            is_satisfied_now = False
            for lit in clause:
                idx = abs(lit) - 1
                if idx == var:
                    # Use new value
                    if new_value == (lit > 0):
                        is_satisfied_now = True
                        break
                else:
                    if assignment[idx] == (lit > 0):
                        is_satisfied_now = True
                        break
            
            if was_satisfied and not is_satisfied_now:
                delta += 1
            elif not was_satisfied and is_satisfied_now:
                delta -= 1
        
        # Accept or reject
        if delta <= 0:
            # Always accept improvements or neutral moves
            assignment[var] = new_value
            current_energy += delta
        else:
            # Accept with probability exp(-delta / T)
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    assignment[var] = new_value
                    current_energy += delta
        
        # Cool down
        T *= cooling_rate
    
    return assignment
# EVOLVE-BLOCK-END
