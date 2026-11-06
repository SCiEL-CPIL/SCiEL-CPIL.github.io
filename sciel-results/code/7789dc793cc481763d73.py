"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    timeout = 325.0  # Leave 5s margin before 330s kill
    
    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    
    for clause in clauses:
        vars_list = []
        signs_list = []
        for lit in clause:
            var_idx = abs(lit) - 1
            vars_list.append(var_idx)
            signs_list.append(lit > 0)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    m = len(clauses)
    
    # Build variable-to-clauses mapping with position info
    var_to_clauses = [[] for _ in range(n)]
    for i, vars_list in enumerate(clause_vars):
        for j, v in enumerate(vars_list):
            var_to_clauses[v].append((i, j))
    
    # Build clause-adjacency: for each clause, which other clauses share variables
    # This is the bipartite graph structure for LLL-inspired resampling
    clause_adj = [[] for _ in range(m)]
    for i in range(m):
        for j, v in enumerate(clause_vars[i]):
            for other_ci, _ in var_to_clauses[v]:
                if other_ci != i:
                    clause_adj[i].append(other_ci)
    
    # State tracking
    def init_state(assignment):
        """Initialize clause satisfaction state from assignment."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            count = 0
            for j, v in enumerate(clause_vars[i]):
                if assignment[v] == clause_signs[i][j]:
                    count += 1
            clause_sat_count[i] = count
        return clause_sat_count
    
    def flip_var(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count in place."""
        assignment[v] = not assignment[v]
        for ci, j in var_to_clauses[v]:
            if assignment[v] == clause_signs[ci][j]:
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1
    
    def set_var(v, val, assignment, clause_sat_count):
        """Set variable v to val and update clause_sat_count in place."""
        if assignment[v] == val:
            return
        assignment[v] = val
        for ci, j in var_to_clauses[v]:
            if val == clause_signs[ci][j]:
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1
    
    # LLL-inspired resampling: resample variables in unsatisfied clauses
    # with soft constraint satisfaction using clause-adjacency structure
    def lll_resample(assignment, clause_sat_count, rng, max_iters=200):
        """
        LLL-inspired algorithm: iteratively resample variables in currently
        unsatisfied clauses using soft constraint satisfaction.
        
        The key insight: for planted 3-SAT with low constraint density,
        we can use the clause-adjacency structure to guide resampling.
        Variables that appear in many unsatisfied clauses are "high-pressure"
        and should be resampled with higher probability.
        """
        for _ in range(max_iters):
            # Find all unsatisfied clauses
            unsat_mask = (clause_sat_count == 0)
            if not np.any(unsat_mask):
                return True
            
            # Collect variables in unsatisfied clauses with their "pressure"
            var_pressure = np.zeros(n, dtype=np.int32)
            for ci in range(m):
                if clause_sat_count[ci] == 0:
                    for j, v in enumerate(clause_vars[ci]):
                        var_pressure[v] += 1
            
            # Resample high-pressure variables using soft constraint approach
            # For each variable in an unsat clause, compute a "soft score"
            # based on how many adjacent clauses would be satisfied
            resampled_any = False
            unsat_indices = np.where(unsat_mask)[0]
            
            # Process unsatisfied clauses in random order
            rng.shuffle(unsat_indices.tolist())
            
            for ci in unsat_indices:
                if clause_sat_count[ci] > 0:
                    continue  # Already satisfied by previous resampling
                
                # For this clause, find the best variable to resample
                best_var = -1
                best_score = -1
                
                for j, v in enumerate(clause_vars[ci]):
                    # Soft score: count how many adjacent unsat clauses
                    # this variable appears in (excluding current clause)
                    score = 0
                    for other_ci, _ in var_to_clauses[v]:
                        if other_ci != ci and clause_sat_count[other_ci] == 0:
                            score += 1
                    
                    # Add a small random perturbation for diversity
                    score += rng.random() * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_var = v
                
                if best_var >= 0:
                    # Resample: set to the value that satisfies the current clause
                    # with some probability, otherwise random
                    target_clause = ci
                    j_idx = clause_vars[target_clause].index(best_var)
                    target_val = clause_signs[target_clause][j_idx]
                    
                    # With 70% probability, set to satisfy current clause
                    # With 30% probability, random (to escape local minima)
                    if rng.random() < 0.7:
                        new_val = target_val
                    else:
                        new_val = rng.random() < 0.5
                    
                    set_var(best_var, new_val, assignment, clause_sat_count)
                    resampled_any = True
            
            if not resampled_any:
                break
            
            # Check if all satisfied
            if not np.any(clause_sat_count == 0):
                return True
        
        return not np.any(clause_sat_count == 0)
    
    # Targeted repair: focus on most-constrained variables
    def targeted_repair(assignment, clause_sat_count, rng, max_iters=500):
        """
        Targeted repair: prioritize flips that resolve more conflicts.
        Uses variable scoring based on number of unsatisfied clauses
        the variable appears in.
        """
        for _ in range(max_iters):
            unsat_mask = (clause_sat_count == 0)
            if not np.any(unsat_mask):
                return True
            
            # Score each variable by how many unsat clauses it's in
            var_score = np.zeros(n, dtype=np.int32)
            for ci in range(m):
                if clause_sat_count[ci] == 0:
                    for j, v in enumerate(clause_vars[ci]):
                        var_score[v] += 1
            
            # Find the highest-scoring variable
            max_score = np.max(var_score)
            if max_score == 0:
                break
            
            # Get all variables with max score
            candidates = np.where(var_score == max_score)[0]
            v = int(candidates[rng.randint(0, len(candidates) - 1)])
            
            # Determine best value for this variable
            # Count how many clauses each value satisfies
            sat_true = 0
            sat_false = 0
            for ci, j in var_to_clauses[v]:
                if clause_signs[ci][j]:
                    sat_true += 1
                else:
                    sat_false += 1
            
            # Set to the value that satisfies more clauses
            # with some noise to avoid getting stuck
            if rng.random() < 0.8:
                new_val = (sat_true > sat_false)
            else:
                new_val = rng.random() < 0.5
            
            set_var(v, new_val, assignment, clause_sat_count)
            
            if not np.any(clause_sat_count == 0):
                return True
        
        return not np.any(clause_sat_count == 0)
    
    # Polarity-based initialization: use clause structure to guide initial assignment
    def polarity_init(rng):
        """
        Initialize using polarity information from clauses.
        For each variable, count how many times it appears positive vs negative.
        Bias initial assignment toward the more common polarity.
        """
        pos_count = np.zeros(n, dtype=np.int32)
        neg_count = np.zeros(n, dtype=np.int32)
        
        for i in range(m):
            for j, v in enumerate(clause_vars[i]):
                if clause_signs[i][j]:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        
        assignment = []
        for v in range(n):
            total = pos_count[v] + neg_count[v]
            if total == 0:
                assignment.append(rng.random() < 0.5)
            else:
                # Bias toward more common polarity, but with randomness
                p_pos = pos_count[v] / total
                # Add noise to avoid deterministic initialization
                p_pos = 0.3 + 0.4 * p_pos
                assignment.append(rng.random() < p_pos)
        
        return assignment
    
    # WalkSAT with restarts and time budget
    best_assignment = None
    best_unsat_count = m + 1
    
    rng = random.Random(42)
    
    # Phase 1: Try polarity-based initialization + LLL resampling
    num_polarity_attempts = min(5, int((timeout - (time.time() - start_time)) / 5.0))
    for attempt in range(num_polarity_attempts):
        if time.time() - start_time > timeout - 10.0:
            break
        
        assignment = polarity_init(rng)
        clause_sat_count = init_state(assignment)
        
        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
        
        # Apply LLL-inspired resampling
        if lll_resample(assignment, clause_sat_count, rng, max_iters=100):
            return assignment
        
        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
    
    # Phase 2: WalkSAT with adaptive noise and targeted repair
    max_flips_per_restart = 6000
    base_noise = 0.2
    noise_decay = 0.98  # Noise decreases over time within a restart
    
    while time.time() - start_time < timeout - 10.0:
        # Random restart with some polarity bias
        if rng.random() < 0.3:
            assignment = polarity_init(rng)
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        clause_sat_count = init_state(assignment)
        
        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
        
        noise = base_noise
        stall_count = 0
        prev_unsat = current_unsat
        
        for flip_num in range(max_flips_per_restart):
            if time.time() - start_time > timeout - 5.0:
                break
            
            # Find unsatisfied clauses
            unsat_indices = np.where(clause_sat_count == 0)[0]
            if len(unsat_indices) == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            ci = int(rng.choice(unsat_indices))
            
            # Adaptive WalkSAT heuristic
            if rng.random() < noise:
                # Random pick
                j = rng.randint(0, 2)
                v = clause_vars[ci][j]
            else:
                # Greedy: pick var that satisfies most other unsat clauses
                best_var = -1
                best_score = -1
                for j, v in enumerate(clause_vars[ci]):
                    score = 0
                    for other_ci, _ in var_to_clauses[v]:
                        if other_ci != ci and clause_sat_count[other_ci] == 0:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_var = v
                v = best_var
            
            if v >= 0:
                flip_var(v, assignment, clause_sat_count)
            
            current_unsat = int(np.sum(clause_sat_count == 0))
            if current_unsat == 0:
                return assignment
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
            
            # Track stalls and adapt noise
            if current_unsat >= prev_unsat:
                stall_count += 1
                noise = min(0.5, noise * 1.1)  # Increase noise when stuck
            else:
                stall_count = 0
                noise = max(base_noise, noise * noise_decay)
            
            prev_unsat = current_unsat
            
            # If stuck for too long, try targeted repair
            if stall_count > 200:
                if targeted_repair(assignment, clause_sat_count, rng, max_iters=100):
                    return assignment
                stall_count = 0
                noise = base_noise
    
    # Final phase: High-precision polish on best assignment
    if best_assignment is not None and time.time() - start_time < timeout - 5.0:
        assignment = best_assignment.copy()
        clause_sat_count = init_state(assignment)
        
        # Apply targeted repair with more iterations
        targeted_repair(assignment, clause_sat_count, rng, max_iters=2000)
        
        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
