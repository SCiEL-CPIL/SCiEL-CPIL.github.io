"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s kill

    # Build clause data for fast evaluation
    # Each clause: list of (var_idx, polarity) where polarity is 1 if positive, -1 if negative
    clause_vars = []  # list of lists of var indices
    clause_pols = []  # list of lists of polarities
    for cl in clauses:
        vars_in_cl = []
        pols_in_cl = []
        for lit in cl:
            var_idx = abs(lit) - 1
            pol = 1 if lit > 0 else -1
            vars_in_cl.append(var_idx)
            pols_in_cl.append(pol)
        clause_vars.append(vars_in_cl)
        clause_pols.append(pols_in_cl)
    
    num_clauses = len(clauses)

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        unsat_count = 0
        for i in range(num_clauses):
            cl_satisfied = False
            for j in range(len(clause_vars[i])):
                v = clause_vars[i][j]
                p = clause_pols[i][j]
                if assignment[v] == (p == 1):
                    cl_satisfied = True
                    break
            if not cl_satisfied:
                unsat_count += 1
        return unsat_count

    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses."""
        for i in range(num_clauses):
            cl_satisfied = False
            for j in range(len(clause_vars[i])):
                v = clause_vars[i][j]
                p = clause_pols[i][j]
                if assignment[v] == (p == 1):
                    cl_satisfied = True
                    break
            if not cl_satisfied:
                return False
        return True

    # Spectral initialization via Max-Cut formulation
    # Build graph: nodes are variables, edges represent conflicts
    # For each clause, add edges between variables that appear in opposite polarity
    # This creates a graph where cutting = satisfying
    def spectral_init():
        """Use spectral bisection to get initial assignment."""
        # Build adjacency matrix for conflict graph
        # Edge between i,j if there's a clause with one positive and one negative literal
        # Weight by number of such clauses
        adj = np.zeros((n, n), dtype=np.float64)
        for cl in clauses:
            pos_vars = []
            neg_vars = []
            for lit in cl:
                v = abs(lit) - 1
                if lit > 0:
                    pos_vars.append(v)
                else:
                    neg_vars.append(v)
            # Add edges between pos and neg variables in this clause
            for pv in pos_vars:
                for nv in neg_vars:
                    adj[pv, nv] += 1
                    adj[nv, pv] += 1
        
        # Compute Laplacian
        degree = adj.sum(axis=1)
        L = np.diag(degree) - adj
        
        # Get smallest non-trivial eigenvector
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # The second smallest eigenvector (Fiedler vector)
            fiedler = eigvecs[:, 1]
            # Assign based on sign
            assignment = [val > 0 for val in fiedler]
        except:
            assignment = [random.random() < 0.5 for _ in range(n)]
        
        return assignment

    # WalkSAT with adaptive parameters
    def walk_sat(assignment, max_steps, noise_schedule="linear"):
        """WalkSAT with given initial assignment and step limit."""
        assignment = list(assignment)
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment, 0
        
        # Track best solution
        best_assignment = list(assignment)
        best_unsat = current_unsat
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for i in range(num_clauses):
                cl_satisfied = False
                for j in range(len(clause_vars[i])):
                    v = clause_vars[i][j]
                    p = clause_pols[i][j]
                    if assignment[v] == (p == 1):
                        cl_satisfied = True
                        break
                if not cl_satisfied:
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assignment, step
            
            # Choose a random unsatisfied clause
            ci = random.choice(unsat_clauses)
            
            # Calculate flip scores: for each variable in the clause,
            # count how many clauses would be affected
            clause = clause_vars[ci]
            best_var = -1
            best_score = float('inf')
            
            # Adaptive noise parameter
            progress = step / max_steps
            if noise_schedule == "linear":
                noise = 0.5 * (1 - progress)
            else:
                noise = 0.5
            
            for v in clause:
                # Count satisfied clauses that would become unsatisfied
                # and unsatisfied clauses that would become satisfied
                sat_to_unsat = 0
                unsat_to_sat = 0
                
                for i in range(num_clauses):
                    cl_has_var = False
                    cl_satisfied = False
                    cl_would_be_satisfied = False
                    
                    for j in range(len(clause_vars[i])):
                        vv = clause_vars[i][j]
                        p = clause_pols[i][j]
                        if vv == v:
                            cl_has_var = True
                        if assignment[vv] == (p == 1):
                            cl_satisfied = True
                        # Check if clause would be satisfied if we flip v
                        if vv == v:
                            if (not assignment[v]) == (p == 1):
                                cl_would_be_satisfied = True
                        else:
                            if assignment[vv] == (p == 1):
                                cl_would_be_satisfied = True
                    
                    if cl_has_var:
                        if cl_satisfied and not cl_would_be_satisfied:
                            sat_to_unsat += 1
                        elif not cl_satisfied and cl_would_be_satisfied:
                            unsat_to_sat += 1
                
                score = sat_to_unsat - unsat_to_sat + noise * random.random() * 2
                if score < best_score:
                    best_score = score
                    best_var = v
            
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
                current_unsat = count_unsat(assignment)
                
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = list(assignment)
                
                if current_unsat == 0:
                    return assignment, step
        
        # Return best found
        return best_assignment, max_steps

    # Multi-restart strategy
    rng = random.Random(42)
    
    # First, try spectral initialization
    spectral_assignment = spectral_init()
    result, _ = walk_sat(spectral_assignment, 50000, "linear")
    if is_satisfied(result):
        return result
    
    # If spectral didn't work, try multiple random restarts
    best_result = result
    best_unsat_count = count_unsat(result)
    
    num_restarts = 20
    steps_per_restart = 30000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        result, _ = walk_sat(assignment, steps_per_restart, "linear")
        current_unsat = count_unsat(result)
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_result = result
        
        if current_unsat == 0:
            return result
        
        # Reduce steps for later restarts if time is running out
        if restart > num_restarts // 2:
            steps_per_restart = max(10000, steps_per_restart // 2)
    
    # Final: try a more aggressive local search on best result
    if best_unsat_count > 0 and time.time() < deadline - 10.0:
        # Use simulated annealing style on the best result
        assignment = list(best_result)
        current_unsat = best_unsat_count
        best_assignment = list(assignment)
        best_unsat = current_unsat
        temperature = 5.0
        cooling = 0.9995
        
        for step in range(100000):
            if time.time() > deadline - 2.0:
                break
            
            # Pick random variable
            v = rng.randint(0, n - 1)
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            new_unsat = count_unsat(assignment)
            
            delta = new_unsat - current_unsat
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_unsat = new_unsat
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = list(assignment)
                    if best_unsat == 0:
                        return best_assignment
            else:
                assignment[v] = old_val
            
            temperature *= cooling
            if temperature < 0.01:
                break
        
        if best_unsat < best_unsat_count:
            best_result = best_assignment
            best_unsat_count = best_unsat
    
    return best_result
# EVOLVE-BLOCK-END
