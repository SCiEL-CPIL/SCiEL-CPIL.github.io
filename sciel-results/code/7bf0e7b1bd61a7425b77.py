"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n

    # 1. Pre-process
    # Clause data: list of lists of ints (signed literals)
    # We need:
    # - var_to_clauses: dict var_idx -> list of clause_indices
    # - clause_vars: list of lists of var_idx
    # - clause_signs: list of lists of bool (True for positive)
    
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []
    clause_signs = []
    
    for i, c in enumerate(clauses):
        cv = []
        cs = []
        for lit in c:
            v = abs(lit) - 1
            s = (lit > 0)
            cv.append(v)
            cs.append(s)
            var_to_clauses[v].append(i)
        clause_vars.append(cv)
        clause_signs.append(cs)
        
    num_clauses = len(clauses)
    
    # Phase initialization
    pos_counts = np.zeros(n, dtype=np.int32)
    neg_counts = np.zeros(n, dtype=np.int32)
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
                
    phase = pos_counts > neg_counts
    # Convert to bool list for easy manipulation
    phase_list = phase.tolist()
    
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_assignment = phase_list.copy()
    
    # Main Search Loop
    # We will use a restart strategy.
    # Each restart:
    #   - Reset assignment to phase
    #   - Run local search for X steps or until solved
    #   - If solved, return
    #   - If time runs out, break
    
    max_restarts = 100
    steps_per_restart = 10000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = phase_list.copy()
        
        # Helper to check if a clause is satisfied
        # We will maintain a set of unsatisfied clause indices for faster access?
        # Or just re-evaluate?
        # Given the small size, re-evaluating unsat clauses for the "is solved" check
        # and for the flip decision is feasible.
        
        # Let's maintain a list of unsat clauses to pick from.
        # Initial unsat clauses:
        unsat_clauses = []
        for i in range(num_clauses):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = any(assignment[v] == s for v, s in zip(cv, cs))
            if not sat:
                unsat_clauses.append(i)
                
        if not unsat_clauses:
            return assignment
            
        for step in range(steps_per_restart):
            if time.time() > deadline:
                # Return best effort
                # We don't track "best" unsat count explicitly in this simple loop,
                # so we just return current.
                return assignment
                
            if not unsat_clauses:
                return assignment
                
            # Pick a random unsat clause
            clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            cv = clause_vars[clause_idx]
            cs = clause_signs[clause_idx]
            
            # Evaluate flip for each literal in this clause
            best_var = -1
            best_delta = float('inf')
            best_tie_breaker = 0
            
            for k in range(3):
                v = cv[k]
                # Calculate delta if we flip v
                # Delta = (new_unsat_count - old_unsat_count)
                # We only care about clauses containing v.
                # Current state of clauses containing v:
                # We need to know which of these are currently satisfied/unsat.
                
                # Optimization: We can pre-compute the satisfaction status of all clauses?
                # Or just check the ones in var_to_clauses[v].
                
                delta = 0
                for c_idx in var_to_clauses[v]:
                    c_cv = clause_vars[c_idx]
                    c_cs = clause_signs[c_idx]
                    
                    # Check if clause c_idx is currently satisfied
                    # and if it would remain satisfied if v is flipped.
                    
                    # Current satisfaction:
                    # We can quickly check if any other literal in the clause is satisfied.
                    # If yes, flipping v doesn't affect satisfaction.
                    # If no, the clause is satisfied ONLY by v (if v is currently correct)
                    # or is unsatisfied (if v is currently incorrect).
                    
                    # Let's check if v is currently satisfying the clause.
                    v_sign = c_cs[c_cv.index(v)] # This is O(3)
                    # Wait, c_cv is length 3. index is fast.
                    
                    v_current_val = assignment[v]
                    v_satisfies_now = (v_current_val == v_sign)
                    
                    # Check if any OTHER literal in the clause is satisfied.
                    other_sat = False
                    for j in range(3):
                        if c_cv[j] != v:
                            if assignment[c_cv[j]] == c_cs[j]:
                                other_sat = True
                                break
                                
                    if other_sat:
                        # Clause is satisfied by others. Flipping v has no effect on this clause's status.
                        continue
                    else:
                        # Clause satisfaction depends entirely on v.
                        if v_satisfies_now:
                            # Currently satisfied. Flipping v makes it unsatisfied.
                            delta += 1
                        else:
                            # Currently unsatisfied. Flipping v makes it satisfied.
                            delta -= 1
                
                # Tie-breaking: random
                if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
                    best_delta = delta
                    best_var = v
                    best_tie_breaker = rng.random()
                    
            # Flip best_var
            assignment[best_var] = not assignment[best_var]
            
            # Update unsat_clauses list
            # This is the expensive part if done naively.
            # We need to update the status of all clauses containing best_var.
            
            # Let's rebuild the unsat list? Or update incrementally?
            # Rebuilding is O(C). C is small.
            # Let's just rebuild for simplicity and correctness, it's fast enough for 50 vars.
            # Or better: only update clauses containing best_var.
            
            # Clauses affected: var_to_clauses[best_var]
            # We need to know their new status.
            
            # Let's maintain a boolean array `is_satisfied` of length C.
            # We didn't initialize it above. Let's do it properly.
            
            # Refactoring: Initialize is_satisfied array.
            
            # Since I'm in the middle of the loop, let's just re-scan all clauses for simplicity 
            # given the low N and C.
            # Actually, re-scanning 500 clauses is trivial.
            unsat_clauses = []
            for i in range(num_clauses):
                cv = clause_vars[i]
                cs = clause_signs[i]
                sat = any(assignment[v] == s for v, s in zip(cv, cs))
                if not sat:
                    unsat_clauses.append(i)
                    
    return assignment
# EVOLVE-BLOCK-END
