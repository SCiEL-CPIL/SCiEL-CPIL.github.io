"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 290 seconds to be safe
    deadline = time.time() + 290
    
    # Parameters
    noise = 0.5  # Probability of random flip (WalkSAT standard)
    max_steps_per_restart = 10000
    restarts = 0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Pre-compute clause lengths if needed, but 3-SAT is fixed size mostly.
    # We'll just iterate.
    
    while time.time() < deadline:
        restarts += 1
        # Initialize random assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat count
        unsat_count = sum(0 if any(assignment[abs(l)-1] == (l > 0) for l in c) else 1 for c in clauses)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT loop
        steps = 0
        while steps < max_steps_per_restart and time.time() < deadline:
            # Find an unsatisfied clause
            # To avoid O(N) scan every step, we can maintain a list of unsat clauses,
            # but for N=50 and M~100-200, a scan is fast enough.
            # Let's use a simple scan for clarity and correctness first.
            unsat_clause = None
            for c in clauses:
                if not any(assignment[abs(l)-1] == (l > 0) for l in c):
                    unsat_clause = c
                    break
            
            if unsat_clause is None:
                # Satisfied
                return assignment
            
            # Heuristic: MinConfl
            # For each literal in the clause, calculate how many satisfied clauses
            # would become unsatisfied if we flip that variable.
            # We want to minimize this "damage".
            
            # Optimization: Pre-calculate clause satisfaction state?
            # Given small N, we can just compute on the fly.
            
            min_damage = float('inf')
            best_lit = unsat_clause[0]
            
            for lit in unsat_clause:
                var_idx = abs(lit) - 1
                # Simulate flip
                new_val = not assignment[var_idx]
                
                damage = 0
                for c in clauses:
                    # Check if clause c is currently satisfied
                    # And if it contains var_idx
                    # If it does, and the literal in c matches the current value,
                    # and no other literal in c matches the current value (so it's the only one keeping it satisfied),
                    # then flipping will unsatisfy it.
                    
                    # Actually, simpler:
                    # A clause is satisfied if at least one literal is true.
                    # If we flip var_idx, the truth value of any literal involving var_idx changes.
                    
                    # Let's just count satisfied clauses before and after? No, too slow.
                    
                    # Standard MinConfl:
                    # For each clause c that is currently satisfied:
                    #   If c contains literal l' with var_idx:
                    #     Check if c has any OTHER literal l'' that is true in current assignment.
                    #     If not, then c is "solely" satisfied by l'.
                    #     Flipping var_idx will make l' false, so c becomes unsatisfied.
                    #     Increment damage.
                    
                    # This is O(M * L) per literal, O(M * L^2) per step.
                    # M~200, L=3. 200*9 = 1800 ops per step. 10000 steps = 18M ops.
                    # Python is slow. 18M ops might take ~1-2 seconds.
                    # 290s / 2s = ~145 restarts. That's okay.
                    
                    # Let's optimize the damage calculation.
                    pass
                
                # Let's implement the damage calc properly
                damage = 0
                for c in clauses:
                    # Check if c is currently satisfied
                    sat = False
                    for l in c:
                        if assignment[abs(l)-1] == (l > 0):
                            sat = True
                            break
                    if not sat:
                        continue
                    
                    # c is satisfied. Does it contain var_idx?
                    contains_var = False
                    for l in c:
                        if abs(l) - 1 == var_idx:
                            contains_var = True
                            break
                    
                    if not contains_var:
                        continue
                    
                    # c contains var_idx and is satisfied.
                    # Is var_idx's literal the ONLY one satisfying c?
                    # If yes, flipping it unsatisfies c.
                    only_satisfier = True
                    for l in c:
                        if abs(l) - 1 == var_idx:
                            continue # Skip the literal we are flipping
                        if assignment[abs(l)-1] == (l > 0):
                            only_satisfier = False
                            break
                    
                    if only_satisfier:
                        damage += 1
                
                if damage < min_damage:
                    min_damage = damage
                    best_lit = lit
            
            # Decide flip
            if random.random() < noise:
                lit_to_flip = random.choice(unsat_clause)
            else:
                lit_to_flip = best_lit
            
            assignment[abs(lit_to_flip) - 1] = not assignment[abs(lit_to_flip) - 1]
            steps += 1
            
            # Update unsat_count?
            # We can update incrementally, but checking is O(M).
            # Let's just check if satisfied every step.
            if all(any(assignment[abs(l)-1] == (l > 0) for l in c) for c in clauses):
                return assignment
            
            # Update best
            # Calculating full unsat count is O(M).
            # We can do it every 10 steps or so to save time.
            if steps % 10 == 0:
                cur_unsat = sum(0 if any(assignment[abs(l)-1] == (l > 0) for l in c) else 1 for c in clauses)
                if cur_unsat < best_unsat_count:
                    best_unsat_count = cur_unsat
                    best_assignment = assignment[:]
        
        # End of restart steps
        # If we are running low on time, break outer loop
        if time.time() > deadline - 5:
            break
            
    return best_assignment
# EVOLVE-BLOCK-END
