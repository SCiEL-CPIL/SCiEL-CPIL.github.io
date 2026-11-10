"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 295  # Safety margin before 300s kill
    
    # 1. Preprocessing: Unit Propagation
    # We work on a simplified instance to speed up the search.
    # But we must map back to the original variables.
    
    # Let's try a lighter preprocessing: just identify unit clauses in the original and assign them.
    # If we assign variables, we can reduce the problem.
    
    # To keep it simple and robust within the time limit, let's do a "lookahead" initialization 
    # and a standard WalkSAT, but with the suggested "majority sign" bias.
    
    # Calculate majority sign bias for each variable
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            v = abs(lit) - 1
            if 0 <= v < n:
                if lit > 0:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
    
    # 2. WalkSAT with improved initialization and restarts
    rng = random.Random(42)
    
    # Precompute variable to clauses mapping for fast update
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            if 0 <= v < n:
                var_to_clauses[v].append((ci, lit))
    
    m = len(clauses)
    
    def is_satisfied(ci, assignment):
        for lit in clauses[ci]:
            v = abs(lit) - 1
            if 0 <= v < n and assignment[v] == (lit > 0):
                return True
        return False

    def count_unsat(assignment):
        return sum(1 for ci in range(m) if not is_satisfied(ci, assignment))

    def luby_sequence(seed=1):
        """Generate Luby restart sequence."""
        def luby(n):
            # Luby sequence: 1, 2, 1, 4, 1, 2, 1, 8, ...
            # Find the largest power of 2 less than or equal to n
            p = 1
            while p * 2 <= n:
                p *= 2
            if n == p:
                return p
            else:
                return luby(n - p)
        
        i = 0
        while True:
            i += 1
            yield luby(i)

    luby_iter = luby_sequence()
    
    best_assign = None
    best_unsat = float('inf')
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Get flip budget from Luby sequence
        # Scale the Luby sequence to be appropriate for N=50
        # Standard WalkSAT flips ~ 100*N to 1000*N
        base_flip_budget = 1000
        flip_budget = int(base_flip_budget * next(luby_iter))
        
        # Cap the flip budget to avoid extremely long runs in a single restart
        if flip_budget > 20000:
            flip_budget = 20000
            
        # Initialization: Majority Sign Bias
        assignment = [False] * n
        for v in range(n):
            total = pos_count[v] + neg_count[v]
            if total > 0:
                bias = pos_count[v] / total
                # Probability of True is bias
                assignment[v] = rng.random() < bias
            else:
                assignment[v] = rng.random() < 0.5
        
        # Track satisfaction state
        sat = [is_satisfied(ci, assignment) for ci in range(m)]
        unsat_indices = [ci for ci in range(m) if not sat[ci]]
        current_unsat_count = len(unsat_indices)
        
        if current_unsat_count == 0:
            return assignment
        
        # Adaptive Noise: Start with higher noise to explore, lower to exploit?
        # Or vary per restart.
        # Let's use a noise that decreases slightly within a restart?
        # Standard WalkSAT uses fixed noise per restart.
        # Let's try a noise that depends on restart count, oscillating.
        noise = 0.1 + 0.05 * math.sin(restart_count * 0.5)
        
        for step in range(flip_budget):
            if time.time() > deadline:
                break
            
            if not unsat_indices:
                return assignment
            
            # Pick a random unsatisfied clause
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = clauses[ci]
            
            # Calculate score for each literal in the clause
            best_score = -1e9
            best_lit = clause[0]
            candidates = []
            
            for lit in clause:
                v = abs(lit) - 1
                if v < 0 or v >= n:
                    continue
                
                gain = 0
                loss = 0
                
                for (uci, ulit) in var_to_clauses[v]:
                    if ulit == lit:
                        if not sat[uci]:
                            gain += 1
                    else:
                        if sat[uci]:
                            # Check if ulit is the only satisfying literal
                            # To be precise, we should check if flipping v would make uci unsat
                            # This requires checking if any other literal in uci is true.
                            # For speed, we approximate loss += 1 if sat[uci].
                            # This is the standard "min-unsat" WalkSAT approximation.
                            loss += 1
                
                score = gain - loss
                candidates.append((score, lit))
                
                if score > best_score:
                    best_score = score
                    best_lit = lit
            
            if not candidates:
                lit = rng.choice(clause)
            else:
                if rng.random() < noise:
                    _, lit = rng.choice(candidates)
                else:
                    max_score = best_score
                    tied = [c for c in candidates if c[0] == max_score]
                    _, lit = rng.choice(tied)
            
            # Flip variable
            v = abs(lit) - 1
            assignment[v] = not assignment[v]
            
            # Update satisfaction
            for (uci, ulit) in var_to_clauses[v]:
                if sat[uci]:
                    new_sat = is_satisfied(uci, assignment)
                    if not new_sat:
                        sat[uci] = False
                        unsat_indices.append(uci)
                        current_unsat_count += 1
                else:
                    new_sat = is_satisfied(uci, assignment)
                    if new_sat:
                        sat[uci] = True
                        try:
                            unsat_indices.remove(uci)
                        except ValueError:
                            pass
                        current_unsat_count -= 1
            
            if current_unsat_count == 0:
                return assignment
        
        # End of restart
        unsat_cnt = current_unsat_count
        if unsat_cnt < best_unsat:
            best_unsat = unsat_cnt
            best_assign = assignment[:]

    if best_assign is not None:
        return best_assign
    
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
