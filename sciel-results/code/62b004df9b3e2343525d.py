"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time


def solve(instance):
    # Simulated annealing for Max-SAT (planted 3-SAT)
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Track deadline
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s safety margin
    
    # Precompute literal indices and signs for speed
    clause_info = []
    for clause in clauses:
        info = [(abs(l) - 1, l > 0) for l in clause]
        clause_info.append(info)
    
    def count_satisfied(assignment):
        count = 0
        for clause in clause_info:
            for var_idx, is_pos in clause:
                if assignment[var_idx] == is_pos:
                    count += 1
                    break
        return count
    
    # Initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    current_score = count_satisfied(assignment)
    best_assignment = assignment[:]
    best_score = current_score
    
    # Simulated annealing parameters
    initial_temp = 5.0
    cooling_rate = 0.9995
    max_steps = 1000000
    
    temp = initial_temp
    step = 0
    
    while step < max_steps:
        if time.time() > deadline:
            break
        
        # Pick a random variable to flip
        var_idx = rng.randint(0, n - 1)
        old_val = assignment[var_idx]
        assignment[var_idx] = not old_val
        
        new_score = count_satisfied(assignment)
        delta = new_score - current_score
        
        if delta >= 0 or rng.random() < math.exp(delta / temp):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_assignment = assignment[:]
                if best_score == len(clauses):
                    break
        else:
            assignment[var_idx] = old_val
        
        temp *= cooling_rate
        step += 1
    
    return best_assignment
# EVOLVE-BLOCK-END
