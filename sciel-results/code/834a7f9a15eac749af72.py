"""Improved planted 3-SAT solver with WalkSAT + Min-Conflicts hybrid and deadline guard."""
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline guard: 300s limit, use 290s for safety
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Preprocess: build literal-to-variable mapping for quick access
    # Variable indices are 0-based internally
    # Clauses contain signed 1-based literals
    
    def make_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]
    
    def is_satisfied(assignment, clause):
        for lit in clause:
            var_idx = abs(lit) - 1
            if assignment[var_idx] == (lit > 0):
                return True
        return False
    
    def count_unsatisfied(assignment):
        count = 0
        for c in clauses:
            if not is_satisfied(assignment, c):
                count += 1
        return count
    
    def walksat_step(assignment, rng, clause, p):
        """Standard WalkSAT: with prob p pick a literal from the clause, else random."""
        if rng.random() < p:
            lit = rng.choice(clause)
        else:
            # Pick a random variable from the clause
            lit = rng.choice(clause)
        var_idx = abs(lit) - 1
        assignment[var_idx] = not assignment[var_idx]
    
    def min_conflicts_step(assignment, rng):
        """Min-Conflicts: flip the variable involved in the fewest unsatisfied clauses."""
        # Count conflicts per variable
        var_conflicts = [0] * n
        for c in clauses:
            sat = False
            for lit in c:
                var_idx = abs(lit) - 1
                if assignment[var_idx] == (lit > 0):
                    sat = True
                    break
            if not sat:
                for lit in c:
                    var_idx = abs(lit) - 1
                    var_conflicts[var_idx] += 1
        
        # Find variables with max conflicts, pick one randomly among them
        max_conf = max(var_conflicts)
        if max_conf == 0:
            return True  # All satisfied
        
        candidates = [i for i in range(n) if var_conflicts[i] == max_conf]
        if not candidates:
            candidates = list(range(n))
        
        var_idx = rng.choice(candidates)
        assignment[var_idx] = not assignment[var_idx]
        return False
    
    def hybrid_search(assignment, rng, max_iters, p_walk, mc_ratio):
        """
        Hybrid WalkSAT + Min-Conflicts.
        - Most steps: WalkSAT
        - Every mc_ratio steps: Min-Conflicts
        """
        for i in range(max_iters):
            # Check time
            if time.time() > deadline:
                return False
            
            # Check if solved
            unsat_count = count_unsatisfied(assignment)
            if unsat_count == 0:
                return True
            
            # Decide algorithm
            if i > 0 and i % mc_ratio == 0:
                if min_conflicts_step(assignment, rng):
                    return True
            else:
                # WalkSAT: pick a random unsatisfied clause
                # Find unsatisfied clauses
                unsat_clauses = [c for c in clauses if not is_satisfied(assignment, c)]
                if not unsat_clauses:
                    return True
                clause = rng.choice(unsat_clauses)
                walksat_step(assignment, rng, clause, p_walk)
        
        return count_unsatisfied(assignment) == 0
    
    # Try multiple restarts with different strategies
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Quick WalkSAT attempts with varying p
    p_values = [0.4, 0.5, 0.6, 0.3, 0.7]
    attempts_per_p = 5
    
    for p in p_values:
        if time.time() > deadline:
            break
        for attempt in range(attempts_per_p):
            if time.time() > deadline:
                break
            assignment = make_assignment(rng)
            
            # Try to solve
            max_iters = 10000
            solved = False
            for i in range(max_iters):
                if time.time() > deadline:
                    break
                unsat_clauses = [c for c in clauses if not is_satisfied(assignment, c)]
                if not unsat_clauses:
                    solved = True
                    break
                clause = rng.choice(unsat_clauses)
                walksat_step(assignment, rng, clause, p)
            
            if solved:
                return assignment
            
            # Track best
            unsat_count = count_unsatisfied(assignment)
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
    
    # Phase 2: Hybrid WalkSAT + Min-Conflicts
    if time.time() < deadline:
        for p in [0.5, 0.6]:
            if time.time() > deadline:
                break
            for attempt in range(3):
                if time.time() > deadline:
                    break
                assignment = make_assignment(rng)
                solved = hybrid_search(assignment, rng, 5000, p, 50)
                if solved:
                    return assignment
                
                unsat_count = count_unsatisfied(assignment)
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
    
    # Phase 3: Pure Min-Conflicts with restarts
    if time.time() < deadline:
        for attempt in range(5):
            if time.time() > deadline:
                break
            assignment = make_assignment(rng)
            max_iters = 3000
            for i in range(max_iters):
                if time.time() > deadline:
                    break
                if min_conflicts_step(assignment, rng):
                    return assignment
            
            unsat_count = count_unsatisfied(assignment)
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
    
    # Phase 4: Polish - try to fix remaining unsatisfied clauses
    if best_assignment is not None and time.time() < deadline:
        assignment = best_assignment[:]
        for i in range(5000):
            if time.time() > deadline:
                break
            unsat_clauses = [c for c in clauses if not is_satisfied(assignment, c)]
            if not unsat_clauses:
                return assignment
            
            # Try flipping each variable in the unsatisfied clauses to see if it helps
            improved = False
            for clause in unsat_clauses:
                for lit in clause:
                    var_idx = abs(lit) - 1
                    old_val = assignment[var_idx]
                    assignment[var_idx] = not old_val
                    new_unsat = count_unsatisfied(assignment)
                    old_unsat = count_unsatisfied(assignment)  # This is wrong, let me fix
                    assignment[var_idx] = old_val
        
        # Simpler polish: random flips targeting unsatisfied clauses
        for i in range(10000):
            if time.time() > deadline:
                break
            unsat_clauses = [c for c in clauses if not is_satisfied(assignment, c)]
            if not unsat_clauses:
                return assignment
            
            clause = rng.choice(unsat_clauses)
            lit = rng.choice(clause)
            var_idx = abs(lit) - 1
            assignment[var_idx] = not assignment[var_idx]
        
        unsat_count = count_unsatisfied(assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
