"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0
    
    # Preprocessing
    m = len(clauses)
    # Build matrix representation for fast vectorized operations
    # clause_vars[ci] = list of variable indices (0-based)
    # clause_signs[ci] = list of signs (1 for +var, -1 for -var)
    # var_clauses[v] = list of clause indices containing variable v
    
    clause_vars = []
    clause_signs = []
    var_clauses = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        vars_list = []
        signs_list = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_list.append(v)
            signs_list.append(s)
            var_clauses[v].append(ci)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Helper to check clause satisfaction vectorized
    def compute_clause_sat(assignment_arr):
        """Compute satisfaction for all clauses using numpy."""
        # For each clause, check if any literal is satisfied
        # A literal (v, s) is satisfied if assignment[v] == (s == 1)
        # i.e., if s==1, need assignment[v]==True; if s==-1, need assignment[v]==False
        
        clause_sat = np.zeros(m, dtype=bool)
        for ci in range(m):
            sat = False
            for v, s in zip(clause_vars[ci], clause_signs[ci]):
                if s == 1:
                    if assignment_arr[v]:
                        sat = True
                        break
                else:  # s == -1
                    if not assignment_arr[v]:
                        sat = True
                        break
            clause_sat[ci] = sat
        return clause_sat
    
    def init_state(assignment_list):
        """Initialize clause_sat and num_unsat for a given assignment."""
        assignment_arr = np.array(assignment_list, dtype=bool)
        clause_sat = compute_clause_sat(assignment_arr)
        num_unsat = int(np.sum(~clause_sat))
        return clause_sat, num_unsat
    
    # Main search loop
    max_restarts = 1000
    steps_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
            
        rng = random.Random(restart * 12345 + 6789)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        assignment_arr = np.array(assignment, dtype=bool)
        clause_sat, num_unsat = init_state(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_unsat_count:
            best_unsat_count = num_unsat
            best_assignment = assignment[:]
            
        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break
                
            if num_unsat == 0:
                return assignment
                
            # Find unsatisfied clauses
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assignment
                
            # Pick a random unsatisfied clause
            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
            
            # Compute gain for each literal in this clause
            best_var = -1
            best_score = float('inf')
            
            for v, s in zip(clause_vars[ci], clause_signs[ci]):
                # Calculate gain if we flip v
                gain = 0
                for c_idx in var_clauses[v]:
                    # Check if clause c_idx is satisfied by literals other than v
                    satisfied_by_others = False
                    for v2, s2 in zip(clause_vars[c_idx], clause_signs[c_idx]):
                        if v2 != v:
                            if s2 == 1:
                                if assignment_arr[v2]:
                                    satisfied_by_others = True
                                    break
                            else:  # s2 == -1
                                if not assignment_arr[v2]:
                                    satisfied_by_others = True
                                    break
                    
                    if satisfied_by_others:
                        continue
                    else:
                        # Clause is sat iff v is set to satisfy it
                        if clause_sat[c_idx]:
                            gain -= 1
                        else:
                            gain += 1
                
                # Add noise
                noise = rng.random() * 0.5
                score = -gain + noise
                
                if score < best_score:
                    best_score = score
                    best_var = v
            
            if best_var == -1:
                v = rng.randrange(n)
            else:
                v = best_var
                
            # Flip v
            assignment[v] = not assignment[v]
            assignment_arr[v] = not assignment_arr[v]
            
            # Update clause_sat for clauses containing v
            for c_idx in var_clauses[v]:
                new_sat = False
                for v2, s2 in zip(clause_vars[c_idx], clause_signs[c_idx]):
                    if s2 == 1:
                        if assignment_arr[v2]:
                            new_sat = True
                            break
                    else:  # s2 == -1
                        if not assignment_arr[v2]:
                            new_sat = True
                            break
                
                old_sat = clause_sat[c_idx]
                
                if old_sat and not new_sat:
                    num_unsat += 1
                elif not old_sat and new_sat:
                    num_unsat -= 1
                
                clause_sat[c_idx] = new_sat
            
            if num_unsat == 0:
                return assignment
                
            if num_unsat < best_unsat_count:
                best_unsat_count = num_unsat
                best_assignment = assignment[:]
                
        # End of steps for this restart
        if time.time() >= deadline:
            break

    if best_assignment is not None:
        return best_assignment
        
    # Fallback
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
