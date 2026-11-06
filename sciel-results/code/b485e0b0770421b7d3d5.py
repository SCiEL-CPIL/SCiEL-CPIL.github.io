"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess: convert clauses to arrays of variable indices and signs
    # var_idx: 0-based index
    # sign: +1 for positive literal, -1 for negative
    m = len(clauses)
    if m == 0:
        return [False] * n
    
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1
    
    # Initialize assignment
    rng = np.random.RandomState(42)
    assignment = rng.rand(n) < 0.5
    
    # Track unsatisfied clauses
    # For each clause, we need to know if it's satisfied.
    # A clause is satisfied if any literal is true.
    # literal is true if (assignment[var] == (sign == 1))
    
    def compute_unsat():
        # Returns boolean array of unsatisfied clauses
        # assignment: (n,) bool
        # clause_vars: (m, 3) int32
        # clause_signs: (m, 3) int8
        
        # Get current assignment for all variables in clauses
        # (m, 3) bool
        clause_assignments = assignment[clause_vars]
        
        # A literal is satisfied if:
        # sign == 1 and assignment is True -> True
        # sign == -1 and assignment is False -> True
        # So: (sign == 1) == assignment
        # sign is 1 or -1. assignment is True/False (1/0)
        # If sign=1, we want assignment=True. If sign=-1, we want assignment=False.
        # This is equivalent to: (sign > 0) == assignment
        # Let's check:
        # sign=1 (True), assign=True (True) -> True == True -> True (Satisfied)
        # sign=1 (True), assign=False (False) -> True == False -> False (Unsat)
        # sign=-1 (False), assign=True (True) -> False == True -> False (Unsat)
        # sign=-1 (False), assign=False (False) -> False == False -> True (Satisfied)
        
        literal_satisfied = (clause_signs > 0) == clause_assignments
        
        # Clause is satisfied if ANY literal is satisfied
        clause_satisfied = np.any(literal_satisfied, axis=1)
        
        return ~clause_satisfied
    
    # WalkSAT with random restarts and clause weighting?
    # Let's stick to a robust WalkSAT with adaptive step size.
    
    # Parameters
    max_restarts = 20
    max_steps_per_restart = 10000
    p = 0.5  # Probability of following the "greedy" path (flip var that satisfies most clauses)
    
    for restart in range(max_restarts):
        # Random initialization
        assignment = rng.rand(n) < 0.5
        unsat_mask = compute_unsat()
        num_unsat = np.sum(unsat_mask)
        
        if num_unsat == 0:
            return assignment.tolist()
        
        # Precompute for each variable, which clauses it appears in?
        # This is expensive to build every time, but we can build it once.
        # However, the set of unsat clauses changes.
        
        for step in range(max_steps_per_restart):
            if num_unsat == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            unsat_indices = np.nonzero(unsat_mask)[0]
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            clause_vars_row = clause_vars[clause_idx]
            clause_signs_row = clause_signs[clause_idx]
            
            # Decide flip variable
            if rng.rand() < p:
                # Greedy: flip variable that satisfies the most other unsat clauses
                # For each of the 3 variables in this clause, calculate gain
                gains = np.zeros(3)
                for k in range(3):
                    var = clause_vars_row[k]
                    sign = clause_signs_row[k]
                    
                    # If we flip this var, how many unsat clauses become sat?
                    # And how many sat clauses become unsat?
                    # We only care about net gain in unsat count.
                    
                    # Current state of this literal:
                    # It is currently unsatisfied (since the clause is unsat, all its literals are unsat)
                    # So flipping it will satisfy this clause.
                    
                    # We need to check all other clauses where this variable appears.
                    # This requires a mapping var -> list of (clause_idx, sign_in_clause)
                    
                    # Since we don't have the mapping precomputed for efficiency in this scratch rewrite,
                    # let's just do a simple random flip or a very lightweight heuristic.
                    # Actually, building the map is O(m*3) which is fine.
                    
                    # Let's build the map once outside the loop for speed?
                    # No, the "unsat" status changes.
                    
                    # Alternative: Just use random flip with a slight bias?
                    # Or, compute the gain by iterating through all unsat clauses? That's O(m).
                    # With 150 vars, m is likely ~1000-2000. 10000 steps * 20 restarts * 3 vars * 2000 clauses = 1.2e9 ops. Might be too slow in Python.
                    
                    # Let's try a simpler approach: Random restart + Random Walk (no greedy) is weak.
                    # Let's use a precomputed incidence list.
                    
                    pass 
            else:
                # Random flip from the clause
                k = rng.randint(0, 2)
                var = clause_vars_row[k]
                assignment[var] = not assignment[var]
                # Update unsat_mask? 
                # Instead of updating incrementally, we can just recompute if m is small enough, 
                # or update incrementally.
                # Recomputing unsat_mask is O(m*3) = ~6000 ops. 10000 steps * 6000 = 60M ops per restart.
                # 20 restarts = 1.2B ops. In NumPy, this is vectorized.
                # np.any on (m,3) array is fast.
                unsat_mask = compute_unsat()
                num_unsat = np.sum(unsat_mask)
                continue
            
            # Since the greedy path is complex to implement efficiently without precomputation,
            # let's just do a pure random walk with restarts first? 
            # No, WalkSAT needs the greedy component to be effective.
            
            # Let's implement a proper WalkSAT with precomputed incidence lists.
            
            # Break out to implement proper WalkSAT below.
            break
        
        # If we broke out early, we restart
        pass
    
    # The loop above was a sketch. Let's write the actual efficient WalkSAT.
    
    # Precompute incidence: var -> list of (clause_idx, sign)
    incidence = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var = clause_vars[i, j]
            sign = clause_signs[i, j]
            incidence[var].append((i, sign))
    
    # State:
    # assignment: np.array bool (n,)
    # clause_sat_count: np.array int (m,) - number of satisfied literals in each clause
    # A clause is unsat if clause_sat_count[i] == 0
    
    def init_state():
        assignment = rng.rand(n) < 0.5
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            for j in range(3):
                var = clause_vars[i, j]
                sign = clause_signs[i, j]
                if (sign > 0) == assignment[var]:
                    clause_sat_count[i] += 1
        return assignment, clause_sat_count
    
    for restart in range(50):
        assignment, clause_sat_count = init_state()
        num_unsat = np.sum(clause_sat_count == 0)
        
        if num_unsat == 0:
            return assignment.tolist()
        
        for step in range(20000):
            if num_unsat == 0:
                return assignment.tolist()
            
            # Pick random unsat clause
            unsat_indices = np.nonzero(clause_sat_count == 0)[0]
            c_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # Pick variable to flip
            if rng.rand() < 0.5:
                # Greedy: flip var that minimizes num_unsat (maximizes gain)
                best_var = -1
                best_gain = -1000
                for j in range(3):
                    var = clause_vars[c_idx, j]
                    sign = clause_signs[c_idx, j]
                    
                    gain = 0
                    # This variable is in clause c_idx. Since c_idx is unsat, this literal is unsat.
                    # Flipping it will satisfy c_idx. So +1 gain from c_idx.
                    gain += 1
                    
                    # Check other clauses this variable is in
                    for (other_c, other_sign) in incidence[var]:
                        if other_c == c_idx:
                            continue
                        
                        # Current status of this literal in other_c:
                        # Is it satisfied?
                        # (other_sign > 0) == assignment[var]
                        current_sat = (other_sign > 0) == assignment[var]
                        
                        if current_sat:
                            # Flipping var will make this literal unsat.
                            # If clause other_c has sat_count == 1, it will become unsat (-1 gain)
                            # If clause other_c has sat_count > 1, it stays sat (0 gain)
                            if clause_sat_count[other_c] == 1:
                                gain -= 1
                        else:
                            # Flipping var will make this literal sat.
                            # Clause other_c becomes more satisfied.
                            # If clause other_c was unsat (count 0), it becomes sat (+1 gain)
                            # If clause other_c was sat (count > 0), it stays sat (0 gain)
                            if clause_sat_count[other_c] == 0:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                
                var = best_var
            else:
                # Random
                j = rng.randint(0, 2)
                var = clause_vars[c_idx, j]
            
            # Flip var
            assignment[var] = not assignment[var]
            
            # Update clause_sat_count for all clauses containing var
            for (c, sign) in incidence[var]:
                # Before flip, literal sat?
                # Note: assignment[var] has already been flipped.
                # So we need to check the state BEFORE flip to determine delta.
                # Let's re-evaluate based on the new assignment? No, that's easier.
                # Old sat status: (sign > 0) == old_assignment[var]
                # New sat status: (sign > 0) == new_assignment[var]
                # Since we flipped, old_sat = not new_sat.
                
                new_sat = (sign > 0) == assignment[var]
                
                if new_sat:
                    # It became satisfied.
                    # If it was unsat before, count increases by 1.
                    # We know it was unsat before because we flipped.
                    # Wait, if it was sat before, flipping makes it unsat.
                    # So:
                    # If new_sat is True, it was False before. Count += 1.
                    # If new_sat is False, it was True before. Count -= 1.
                    clause_sat_count[c] += 1
                else:
                    clause_sat_count[c] -= 1
            
            num_unsat = np.sum(clause_sat_count == 0)
    
    # If no solution found, return the last assignment
    return assignment.tolist()

# EVOLVE-BLOCK-END
