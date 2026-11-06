import time
import random
import sys

def solve(instance):
    """
    Solves a 3-SAT instance using a robust Local Search algorithm 
    with dynamic variable weighting and restarts.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 300s limit, use 290s for safety
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Pre-process clauses for faster access
    # Store as list of lists of (var_index, is_positive)
    processed_clauses = []
    var_to_clauses = [[] for _ in range(n)]
    
    for c in clauses:
        pc = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            pc.append((var_idx, is_pos))
            var_to_clauses[var_idx].append(len(processed_clauses))
        processed_clauses.append(pc)
        
    m = len(processed_clauses)
    
    # State
    assignment = [False] * n
    # sat_count[i] is the number of literals in clause i that are satisfied
    sat_count = [0] * m
    # clause_satisfied[i] is True if sat_count[i] >= 1
    clause_satisfied = [False] * m
    # weights for variable selection
    weights = [0.0] * n
    
    def is_satisfied(clause_idx):
        return clause_satisfied[clause_idx]
    
    def update_clause(clause_idx, new_val):
        """Update sat_count and clause_satisfied for a specific clause given a new variable value."""
        # This is called when a variable changes. We need to re-evaluate the clause.
        # Since n is small, we can just re-evaluate the whole clause.
        c = processed_clauses[clause_idx]
        cnt = 0
        for var_idx, is_pos in c:
            if assignment[var_idx] == is_pos:
                cnt += 1
        sat_count[clause_idx] = cnt
        clause_satisfied[clause_idx] = (cnt > 0)
        return cnt

    def initial_setup():
        """Initialize assignment and state."""
        nonlocal assignment, sat_count, clause_satisfied, weights
        rng = random.Random(0) # Fixed seed for reproducibility in initial setup? 
        # Actually, for restarts, we want different seeds or just random.
        # Let's use a global rng or re-seed.
        assignment = [random.random() < 0.5 for _ in range(n)]
        sat_count = [0] * m
        clause_satisfied = [False] * m
        weights = [0.0] * n
        
        for i in range(m):
            c = processed_clauses[i]
            cnt = 0
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_count[i] = cnt
            clause_satisfied[i] = (cnt > 0)

    def get_unsat_count():
        return sum(1 for s in clause_satisfied if not s)

    def flip_variable(var_idx):
        """Flip a variable and update all clauses containing it."""
        assignment[var_idx] = not assignment[var_idx]
        for clause_idx in var_to_clauses[var_idx]:
            # Re-evaluate this clause
            c = processed_clauses[clause_idx]
            cnt = 0
            for v, is_pos in c:
                if assignment[v] == is_pos:
                    cnt += 1
            sat_count[clause_idx] = cnt
            clause_satisfied[clause_idx] = (cnt > 0)

    # Main Search Loop
    initial_setup()
    current_unsat = get_unsat_count()
    best_assignment = assignment.copy()
    best_unsat = current_unsat
    
    if best_unsat == 0:
        return best_assignment

    restart_threshold = 2000 # Steps without improvement before restart
    steps_since_improvement = 0
    max_restarts = 50
    restarts = 0
    
    # Decay factor for weights
    decay = 0.95
    
    while time.time() < deadline:
        if current_unsat == 0:
            return assignment
        
        # If we found a better assignment, save it
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
        
        # Restart logic
        if steps_since_improvement > restart_threshold:
            if restarts >= max_restarts:
                # Try a more aggressive restart: randomize a few more variables
                for _ in range(5):
                    v = random.randint(0, n-1)
                    flip_variable(v)
            else:
                # Full restart
                initial_setup()
                current_unsat = get_unsat_count()
                restarts += 1
                steps_since_improvement = 0
                continue
        
        # Select variable to flip
        # Strategy: Pick variable with highest weight from unsatisfied clauses
        # First, update weights based on unsatisfied clauses
        # Zero out weights or decay? Let's decay and add.
        for i in range(n):
            weights[i] *= decay
        
        # Add to weights for variables in unsatisfied clauses
        for i in range(m):
            if not clause_satisfied[i]:
                for var_idx, is_pos in processed_clauses[i]:
                    weights[var_idx] += 1.0
        
        # Find max weight variable
        # Tie-breaking: random among max weights
        max_w = max(weights)
        candidates = [i for i in range(n) if weights[i] == max_w]
        
        # If all weights are 0 (shouldn't happen if unsat > 0), pick random
        if not candidates:
            var_to_flip = random.randint(0, n-1)
        else:
            var_to_flip = random.choice(candidates)
        
        # Flip the variable
        old_unsat = current_unsat
        flip_variable(var_to_flip)
        current_unsat = get_unsat_count()
        
        # If we improved, we reset steps_since_improvement in the next loop iteration check
        # But we can also check here for early exit
        if current_unsat == 0:
            return assignment

    # If deadline reached, return the best assignment found
    return best_assignment
