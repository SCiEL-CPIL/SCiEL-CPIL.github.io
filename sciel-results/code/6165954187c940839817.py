"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(instance):
    """
    Solve a planted 3-SAT instance using a vectorized WalkSAT approach.
    
    The key improvement over the previous pure-Python implementation is using
    NumPy to evaluate all clauses simultaneously. This allows for thousands of
    flips per second instead of hundreds, which is crucial for finding the
    planted solution in 3-SAT instances.
    
    Strategy:
    1. Preprocess clauses into a NumPy array of literals.
    2. Use a vectorized function to count unsatisfied clauses.
    3. Perform WalkSAT with random restarts and adaptive noise.
    4. If local search fails, fall back to a simple random search or return best effort.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    if num_clauses == 0:
        return [False] * n
    
    # Preprocess: Convert clauses to a numpy array for vectorized operations
    # literals array shape: (num_clauses, 3)
    # Each literal is signed: +var for True, -var for False (1-based)
    # We map to 0-based indices for numpy operations.
    # Let's store (var_idx, polarity) where polarity is 1 for True, 0 for False.
    # Actually, for vectorized satisfaction check:
    # A literal (v, p) is satisfied if assignment[v] == p.
    # We can create a matrix L of shape (M, 3) with var indices (0-based).
    # And a matrix P of shape (M, 3) with polarities (0 or 1).
    
    L = np.zeros((num_clauses, 3), dtype=np.int32)
    P = np.zeros((num_clauses, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            var_idx = abs(lit) - 1
            L[i, j] = var_idx
            P[i, j] = 1 if lit > 0 else 0
    
    # Build variable to clause index mapping for efficient updates in WalkSAT
    # var_to_clauses[var] = list of clause indices containing this variable
    var_to_clauses = [[] for _ in range(n)]
    for i in range(num_clauses):
        for j in range(3):
            var_idx = L[i, j]
            var_to_clauses[var_idx].append(i)

    def count_unsat(assignment):
        """
        Vectorized count of unsatisfied clauses.
        assignment: numpy array of 0s and 1s of length n.
        """
        # Get values of variables for each literal in each clause
        # vals shape: (num_clauses, 3)
        vals = assignment[L]
        # Check satisfaction: vals == P
        sat = (vals == P)
        # A clause is satisfied if any literal is satisfied
        clause_sat = np.any(sat, axis=1)
        # Count unsatisfied
        return np.sum(~clause_sat)

    def walk_sat(assignment, rng, time_limit, noise_schedule):
        """
        Vectorized WalkSAT.
        assignment: numpy array of 0s and 1s.
        """
        start_time = time.time()
        max_flips = 200000  # High limit, controlled by time
        flips = 0
        last_unsat = count_unsat(assignment)
        
        while time.time() - start_time < time_limit:
            if last_unsat == 0:
                return assignment
            
            # Select an unsatisfied clause
            # We need to pick a random unsatisfied clause.
            # To do this efficiently, we can just pick a random clause index and check if it's unsat.
            # If it is, use it. If not, retry. This is inefficient if unsat ratio is low.
            # Better: maintain a list of unsat clauses? Updating it is expensive.
            # Alternative: Pick random clause. If sat, pick another.
            # Given 3-SAT is dense, unsat clauses are usually a significant fraction early on.
            # Late in the game, we might have few unsat clauses.
            
            # Let's try a different selection:
            # Compute satisfaction for all clauses (vectorized)
            vals = assignment[L]
            sat = (vals == P)
            clause_sat = np.any(sat, axis=1)
            unsat_indices = np.where(~clause_sat)[0]
            
            if len(unsat_indices) == 0:
                return assignment
                
            # Pick a random unsatisfied clause
            c_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # Get the 3 variables in this clause
            vars_in_clause = L[c_idx]
            
            # Evaluate potential gain/loss for flipping each of the 3 variables
            best_var = -1
            best_score = -1e9
            scores = []
            
            for v in vars_in_clause:
                # Flip v
                old_val = assignment[v]
                new_val = 1 - old_val
                
                # Calculate change in unsat count
                # This requires checking all clauses containing v.
                # To make this fast, we can precompute the impact?
                # Or just do a local check.
                # Since we are in a Python loop over 3 vars, and each var is in ~30-50 clauses,
                # this is ~150 clause checks per flip.
                # 200,000 flips * 150 checks = 30M checks. Might be slow in Python.
                
                # Optimization: Use the precomputed var_to_clauses.
                gain = 0
                loss = 0
                for c in var_to_clauses[v]:
                    # Check if clause c is currently satisfied
                    # We have 'clause_sat' from the last full evaluation? No, 'clause_sat' is for current state.
                    # We need to know if c is sat NOW, and if it would be sat AFTER flip.
                    
                    # Current state:
                    is_sat_now = clause_sat[c]
                    
                    # Future state:
                    # Check if flipping v satisfies c or unsatisfies it.
                    # A clause is satisfied if ANY literal is satisfied.
                    # If it's sat now, flipping v might unsat it ONLY IF v was the ONLY satisfied literal.
                    # If it's unsat now, flipping v might sat it ONLY IF the literal for v in c becomes sat.
                    
                    lit_idx = -1
                    for k in range(3):
                        if L[c, k] == v:
                            lit_idx = k
                            break
                    if lit_idx == -1:
                        continue # Should not happen since v is in c
                    
                    # Polarity of v in clause c
                    p = P[c, lit_idx]
                    # Is the literal (v, p) satisfied after flip?
                    # new_val == p
                    is_lit_sat_after = (new_val == p)
                    
                    if not is_sat_now:
                        # Clause is unsat now.
                        # If flipping v makes its literal sat, the clause becomes sat.
                        if is_lit_sat_after:
                            gain += 1
                    else:
                        # Clause is sat now.
                        # If flipping v makes its literal unsat, we need to check if other literals are sat.
                        if not is_lit_sat_after:
                            # Check if any other literal in c is sat.
                            other_sat = False
                            for k in range(3):
                                if k == lit_idx:
                                    continue
                                v_k = L[c, k]
                                p_k = P[c, k]
                                # Value of v_k is unchanged (unless v_k == v, which is handled)
                                if assignment[v_k] == p_k:
                                    other_sat = True
                                    break
                            if not other_sat:
                                # This clause will become unsat
                                loss += 1
                                # We can break early if loss > gain? No, we need total score.
                                # But for 3-SAT, loss is usually small.
                    
                    # Note: If is_sat_now is True and is_lit_sat_after is True, no change.
                    # If is_sat_now is False and is_lit_sat_after is False, no change.
                
                score = gain - loss
                scores.append((score, v))
            
            # Choose variable to flip
            # With probability noise, pick random; else pick best
            noise = noise_schedule(flips / max_flips)
            if rng.random() < noise:
                _, best_var = scores[rng.randint(0, 2)]
            else:
                best_var = max(scores, key=lambda x: x[0])[1]
            
            # Flip
            assignment[best_var] = 1 - assignment[best_var]
            flips += 1
            
            # Update last_unsat efficiently?
            # We can just recompute every 100 flips or so to save time, 
            # or every flip if fast enough.
            # Let's recompute every 10 flips to balance overhead.
            if flips % 10 == 0:
                last_unsat = count_unsat(assignment)
                if last_unsat == 0:
                    return assignment

        return None

    # Main Loop: Multiple restarts
    start_time = time.time()
    time_limit = 290
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    best_assignment = None
    best_unsat = float('inf')
    
    # Noise schedule: start with high noise (exploration), decrease (exploitation)
    # Or keep it constant around 0.5-0.7 for 3-SAT.
    # A common choice is 0.5.
    
    num_restarts = 0
    while time.time() - start_time < time_limit:
        num_restarts += 1
        # Random initial assignment
        assignment = np_rng.integers(0, 2, size=n).astype(np.int8)
        
        # Run WalkSAT for a fraction of the time or fixed flips
        # Let's give each restart a specific time budget or flip budget.
        # If we have 290s, and each flip is ~0.1ms (optimistic), we can do 2M flips.
        # But Python overhead is higher. Let's say 50k-100k flips per restart.
        
        # Adaptive restart: if we haven't found a solution, keep restarting.
        # We'll run WalkSAT until time runs out for this restart?
        # No, better to run for a fixed number of flips, then restart.
        # If we get stuck, restart helps.
        
        # Let's try: run for 50,000 flips or 5 seconds, whichever comes first.
        restart_time_budget = 5.0
        if time.time() - start_time > time_limit - restart_time_budget:
            break
            
        result = walk_sat(assignment, np_rng, restart_time_budget, lambda x: 0.5)
        
        if result is not None:
            # Convert numpy array to list of booleans
            return [bool(x) for x in result]
            
        # Track best
        unsat_count = count_unsat(assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
            
        # If we are very close, try a final push with lower noise?
        # Or just restart.
        
    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    
    return [False] * n
# EVOLVE-BLOCK-END
