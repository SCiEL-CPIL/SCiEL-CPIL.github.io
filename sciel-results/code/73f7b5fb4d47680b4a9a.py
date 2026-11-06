"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-process: build clause list as list of tuples of signed literals (1-based)
    # Convert to 0-based indices for internal use
    clauses_0based = [[l - 1 if l > 0 else -l - 1 for l in clause] for clause in clauses]
    
    # Build variable-to-clause index
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses_0based):
        for lit in clause:
            var_to_clauses[abs(lit)].append(ci)
    
    # Phase 1: Vectorized parity-based initialization
    # Treat each clause's parity (XOR of literals) as a soft constraint
    # We want to find an assignment that satisfies as many parity constraints as possible
    # This is equivalent to solving a system of linear equations over GF(2)
    
    # Build the parity constraint matrix
    # Each row corresponds to a clause: XOR of the variables in the clause = 1 (for odd number of negations... actually let's think differently)
    # For a clause (l1 OR l2 OR l3), the parity constraint would be: x_l1 XOR x_l2 XOR x_l3 = 1
    # where x_li is the value of variable |li| if li > 0, else NOT value
    # Actually, let's use a simpler heuristic: initialize based on variable frequency and sign bias
    
    # Count positive and negative occurrences
    pos_count = np.zeros(n, dtype=np.int64)
    neg_count = np.zeros(n, dtype=np.int64)
    for clause in clauses_0based:
        for lit in clause:
            if lit >= 0:
                pos_count[lit] += 1
            else:
                neg_count[-lit - 1] += 1
    
    # Initialize assignment based on sign bias
    # If a variable appears more positively, set it to True; otherwise False
    # For balanced variables, use random
    rng = random.Random(42)
    assignment = np.zeros(n, dtype=bool)
    for i in range(n):
        if pos_count[i] > neg_count[i]:
            assignment[i] = True
        elif neg_count[i] > pos_count[i]:
            assignment[i] = False
        else:
            assignment[i] = rng.random() < 0.5
    
    # Phase 2: Vectorized conflict detection and targeted repair
    # Use numpy for efficient satisfaction checking
    
    def is_satisfied_vectorized(assignment):
        """Check which clauses are unsatisfied. Returns boolean array of clause satisfaction."""
        # For each clause, check if any literal is satisfied
        # A literal l is satisfied if assignment[|l|-1] == (l > 0)
        # Convert clauses to numpy array for vectorized ops
        # This is tricky with variable-length clauses, so we'll use a loop but with optimized inner check
        
        sat = np.ones(len(clauses_0based), dtype=bool)
        for ci, clause in enumerate(clauses_0based):
            # Check if clause is satisfied
            for lit in clause:
                var_idx = abs(lit)
                expected = lit > 0
                if assignment[var_idx] == expected:
                    break
            else:
                sat[ci] = False
        return sat
    
    # Initial check
    sat_mask = is_satisfied_vectorized(assignment)
    
    # Targeted repair: only flip variables involved in unsatisfied clauses
    max_iterations = 50000
    for iteration in range(max_iterations):
        unsat_indices = np.where(~sat_mask)[0]
        if len(unsat_indices) == 0:
            break
        
        # Get all variables involved in unsatisfied clauses
        involved_vars = set()
        for ci in unsat_indices:
            for lit in clauses_0based[ci]:
                involved_vars.add(abs(lit))
        
        if not involved_vars:
            break
        
        # Choose a variable to flip: prioritize variables in most unsatisfied clauses
        # Count how many unsatisfied clauses each involved variable is in
        var_unsat_count = {}
        for ci in unsat_indices:
            for lit in clauses_0based[ci]:
                v = abs(lit)
                var_unsat_count[v] = var_unsat_count.get(v, 0) + 1
        
        # Pick the variable with highest count (break ties randomly)
        max_count = max(var_unsat_count.values())
        candidates = [v for v, c in var_unsat_count.items() if c == max_count]
        var_to_flip = rng.choice(candidates)
        
        # Flip the variable
        assignment[var_to_flip] = not assignment[var_to_flip]
        
        # Update satisfaction status only for clauses involving this variable
        for ci in var_to_clauses[var_to_flip]:
            clause = clauses_0based[ci]
            # Check if clause is now satisfied
            clause_sat = False
            for lit in clause:
                var_idx = abs(lit)
                expected = lit > 0
                if assignment[var_idx] == expected:
                    clause_sat = True
                    break
            sat_mask[ci] = clause_sat
    
    # Convert numpy boolean array to list of booleans
    return [bool(a) for a in assignment]
# EVOLVE-BLOCK-END
