"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def solve(instance):
    """
    Solve a planted 3-SAT instance using a hybrid approach:
    1. DPLL with unit propagation and pure literal elimination
    2. Random restarts with Luby sequence
    3. Fallback to WalkSAT-style local search if DPLL struggles
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 280.0  # Safety margin before 300s kill
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    # Represent assignment as list of -1, 0, 1 (false, unassigned, true)
    # Clauses as list of lists of signed ints
    
    def make_copy(clauses, assignment):
        return [list(c) for c in clauses], list(assignment)
    
    def unit_propagate(clauses, assignment):
        """
        Perform unit propagation. Returns (new_clauses, new_assignment, success).
        If a clause becomes empty, success is False (contradiction).
        """
        changed = True
        while changed and check_time():
            changed = False
            new_clauses = []
            for clause in clauses:
                # Remove satisfied literals and check for unit
                new_clause = []
                satisfied = False
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment[var] == 1 and lit > 0:
                        satisfied = True
                        break
                    elif assignment[var] == -1 and lit < 0:
                        satisfied = True
                        break
                    elif assignment[var] == 0:
                        new_clause.append(lit)
                if satisfied:
                    continue
                if len(new_clause) == 0:
                    return None, None, False  # Contradiction
                if len(new_clause) == 1:
                    lit = new_clause[0]
                    var = abs(lit) - 1
                    assignment[var] = 1 if lit > 0 else -1
                    changed = True
                else:
                    new_clauses.append(new_clause)
            clauses = new_clauses
        return clauses, assignment, True
    
    def pure_literal_elimination(clauses, assignment):
        """
        Eliminate pure literals. Returns (new_clauses, new_assignment, changed).
        """
        if not check_time():
            return clauses, assignment, False
        
        # Build literal -> clause index map
        lit_clauses = defaultdict(list)
        for i, clause in enumerate(clauses):
            for lit in clause:
                lit_clauses[lit].append(i)
        
        changed = False
        new_clauses = []
        for i, clause in enumerate(clauses):
            skip = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] != 0:
                    continue
                # Check if both lit and -lit appear in clauses
                if -lit in lit_clauses:
                    # Not pure
                    break
                else:
                    # Pure literal, assign it
                    assignment[var] = 1 if lit > 0 else -1
                    changed = True
                    skip = True
                    break
            if not skip:
                new_clauses.append(clause)
        
        return new_clauses, assignment, changed
    
    def dpll(clauses, assignment, depth=0):
        """
        DPLL solver with unit propagation and pure literal elimination.
        Returns assignment if solution found, None otherwise.
        """
        if not check_time():
            return None
        
        # Check if all clauses satisfied
        if len(clauses) == 0:
            return assignment
        
        # Try unit propagation
        clauses, assignment, success = unit_propagate(clauses, assignment)
        if not success:
            return None
        if len(clauses) == 0:
            return assignment
        
        # Try pure literal elimination
        clauses, assignment, changed = pure_literal_elimination(clauses, assignment)
        if len(clauses) == 0:
            return assignment
        
        if not check_time():
            return None
        
        # Choose a variable to branch on
        # Use frequency heuristic: pick variable appearing in most clauses
        var_freq = defaultdict(int)
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == 0:
                    var_freq[var] += 1
        
        if not var_freq:
            return assignment
        
        # Pick the most frequent unassigned variable
        var = max(var_freq, key=lambda v: var_freq[v])
        
        # Try both assignments
        for val in [1, -1]:
            if not check_time():
                break
            new_assignment = list(assignment)
            new_assignment[var] = val
            new_clauses, new_assignment, success = unit_propagate(
                [list(c) for c in clauses], new_assignment
            )
            if success:
                result = dpll(new_clauses, new_assignment, depth + 1)
                if result is not None:
                    return result
        
        return None
    
    def luby_sequence():
        """Generate Luby restart sequence."""
        i = 0
        while True:
            k = 1
            while (1 << k) - 1 > i:
                k += 1
            j = 1
            while j <= i + 1 - (1 << (k - 1)):
                yield (1 << (k - j - 1)) if k - j - 1 >= 0 else 1
                j += 1
            i += 1
    
    def walksat_step(clauses, assignment, rng, max_steps=5000):
        """
        One round of WalkSAT. Returns (assignment, num_unsat).
        """
        def count_unsat(clauses, assignment):
            count = 0
            for clause in clauses:
                sat = False
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment[var] == 1 and lit > 0:
                        sat = True
                        break
                    elif assignment[var] == -1 and lit < 0:
                        sat = True
                        break
                if not sat:
                    count += 1
            return count
        
        def get_unsat_clauses(clauses, assignment):
            unsat = []
            for clause in clauses:
                sat = False
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment[var] == 1 and lit > 0:
                        sat = True
                        break
                    elif assignment[var] == -1 and lit < 0:
                        sat = True
                        break
                if not sat:
                    unsat.append(clause)
            return unsat
        
        num_unsat = count_unsat(clauses, assignment)
        
        for step in range(max_steps):
            if not check_time():
                break
            if num_unsat == 0:
                return assignment, 0
            
            unsat_clauses = get_unsat_clauses(clauses, assignment)
            if not unsat_clauses:
                return assignment, 0
            
            clause = rng.choice(unsat_clauses)
            
            # Choose literal with probability 0.55 to flip, otherwise random
            if rng.random() < 0.55:
                # Flip literal that minimizes increase in unsatisfied clauses
                best_lit = None
                best_score = float('inf')
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment[var] == 0:
                        continue
                    # Simulate flip
                    old_val = assignment[var]
                    assignment[var] = -old_val
                    new_unsat = count_unsat(clauses, assignment)
                    assignment[var] = old_val
                    if new_unsat < best_score:
                        best_score = new_unsat
                        best_lit = lit
                if best_lit is not None:
                    var = abs(best_lit) - 1
                    assignment[var] = -assignment[var]
                    num_unsat = count_unsat(clauses, assignment)
            else:
                lit = rng.choice(clause)
                var = abs(lit) - 1
                assignment[var] = -assignment[var]
                num_unsat = count_unsat(clauses, assignment)
        
        return assignment, num_unsat
    
    # Main solving strategy:
    # 1. Try DPLL with restarts using Luby sequence
    # 2. If that fails, try WalkSAT
    
    best_assignment = None
    best_unsat = float('inf')
    
    # Strategy 1: DPLL with restarts
    rng = random.Random(42)
    
    # First, try a straightforward DPLL
    initial_assignment = [0] * n
    result = dpll([list(c) for c in clauses], initial_assignment)
    if result is not None:
        # Convert to boolean list
        return [bool(x) for x in result]
    
    # Try DPLL with random variable ordering
    for restart in range(50):
        if not check_time():
            break
        
        initial_assignment = [0] * n
        # Randomly assign some variables to break symmetry
        for _ in range(min(n // 10, 10)):
            var = rng.randint(0, n - 1)
            if initial_assignment[var] == 0:
                initial_assignment[var] = rng.choice([1, -1])
        
        result = dpll([list(c) for c in clauses], initial_assignment)
        if result is not None:
            return [bool(x) for x in result]
    
    # Strategy 2: WalkSAT with random restarts
    for restart in range(100):
        if not check_time():
            break
        
        # Random initial assignment
        assignment = [rng.choice([1, -1]) for _ in range(n)]
        
        # Do WalkSAT
        assignment, num_unsat = walksat_step(clauses, assignment, rng, max_steps=20000)
        
        if num_unsat == 0:
            return [bool(x) for x in assignment]
        
        if num_unsat < best_unsat:
            best_unsat = num_unsat
            best_assignment = list(assignment)
    
    # If we have a best assignment, return it
    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    
    # Fallback: return random assignment
    return [bool(rng.random() < 0.5) for _ in range(n)]

# EVOLVE-BLOCK-END
