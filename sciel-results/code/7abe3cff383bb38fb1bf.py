"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    # The evaluator kills at 330s. We target 325s to be safe.
    deadline = time.time() + 325.0
    
    # Pre-compute literal indices for speed
    # Store clauses as lists of (var_index, required_value)
    # required_value is True if the literal is positive, False if negative.
    # assignment[var_index] == required_value means the literal is satisfied.
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Helper to count unsatisfied clauses
    # This is the bottleneck. We optimize by breaking early if possible, 
    # but we need the count for best-assignment tracking.
    def count_unsat(assignment):
        count = 0
        for fc in fast_clauses:
            sat = False
            # Unroll or simple loop is fine for size 3
            if assignment[fc[0][0]] == fc[0][1]:
                sat = True
            elif assignment[fc[1][0]] == fc[1][1]:
                sat = True
            elif assignment[fc[2][0]] == fc[2][1]:
                sat = True
            
            if not sat:
                count += 1
        return count

    # Helper to find unsatisfied clauses (returns list of indices)
    def find_unsat(assignment):
        unsat_indices = []
        for i, fc in enumerate(fast_clauses):
            if not (assignment[fc[0][0]] == fc[0][1] or 
                    assignment[fc[1][0]] == fc[1][1] or 
                    assignment[fc[2][0]] == fc[2][1]):
                unsat_indices.append(i)
        return unsat_indices

    best_assignment = None
    best_unsat_count = float('inf')
    
    # Strategy: Multiple WalkSAT restarts.
    # We adapt the number of steps per restart based on remaining time.
    # Initial estimate: We can do roughly 1000-5000 steps/second in Python with this overhead.
    # Let's aim for a base of 2000 steps per restart, scaling with time.
    
    while time.time() < deadline:
        # Estimate steps we can afford for this restart
        # If we have > 100s left, we can afford a long run (e.g., 5000 steps)
        # If we have < 10s left, we do a short run (e.g., 500 steps)
        time_remaining = deadline - time.time()
        
        # Heuristic step budget: 
        # Assume ~1ms per step (conservative for Python with list lookups).
        # So 1000 steps = 1s.
        # We want to leave some time for the next restart or just finish.
        # Let's cap steps at 10000 to avoid one restart eating all time if it's stuck.
        # But if it's easy, it will break early.
        max_steps = int(min(10000, max(500, time_remaining * 800)))
        
        # Initialize random assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Initial check
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        # WalkSAT Loop
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Find unsat clauses
            unsat_indices = find_unsat(assignment)
            
            if not unsat_indices:
                return assignment
            
            # Update best
            if len(unsat_indices) < best_unsat_count:
                best_unsat_count = len(unsat_indices)
                best_assignment = assignment[:]
                
            # Select a random unsatisfied clause
            clause_idx = random.choice(unsat_indices)
            clause = fast_clauses[clause_idx]
            
            # Select a random literal from the clause to flip
            # The inspiration used pure random. We stick to that for reliability.
            var_idx, _ = random.choice(clause)
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
        # End of restart loop
        
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: return a random assignment
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
