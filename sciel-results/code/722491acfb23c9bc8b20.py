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

    num_clauses = len(clauses)
    
    # Pre-process clauses into numpy arrays for speed
    # clause_vars: (num_clauses, 3) array of variable indices (0-based)
    # clause_signs: (num_clauses, 3) array of bools (True for positive literal)
    
    clause_vars = np.empty((num_clauses, 3), dtype=np.int32)
    clause_signs = np.empty((num_clauses, 3), dtype=bool)
    
    for i, c in enumerate(clauses):
        for k in range(3):
            lit = c[k]
            clause_vars[i, k] = abs(lit) - 1
            clause_signs[i, k] = (lit > 0)
            
    # Build var_to_clauses as a list of lists of clause indices
    var_to_clauses = [[] for _ in range(n)]
    for i in range(num_clauses):
        for k in range(3):
            v = clause_vars[i, k]
            var_to_clauses[v].append(i)
            
    # Phase initialization: majority phase
    pos_counts = np.zeros(n, dtype=np.int32)
    neg_counts = np.zeros(n, dtype=np.int32)
    for i in range(num_clauses):
        for k in range(3):
            v = clause_vars[i, k]
            if clause_signs[i, k]:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
                
    phase = pos_counts > neg_counts
    phase_list = phase.tolist()
    
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_assignment = phase_list.copy()
    best_unsat = num_clauses  # Worst case
    
    # We will use WalkSAT with restarts and adaptive p (probability of random flip)
    # Optimizations:
    # 1. Use numpy for satisfaction checks where possible, but the incremental update is key.
    # 2. Maintain a count of unsatisfied literals per clause? Or just satisfied/unsat status.
    # 3. For the delta calculation, we need to know if a clause is satisfied and by which variables.
    
    # Let's maintain:
    # - assignment: list of bools
    # - clause_sat: boolean array of length num_clauses (is clause satisfied?)
    # - clause_sat_count: int array of length num_clauses (how many literals satisfy the clause?)
    #   This helps in determining if flipping a variable makes a clause unsatisfied.
    
    def update_clause_sat(assignment):
        # Recompute clause_sat and clause_sat_count from scratch
        # This is O(C * 3)
        clause_sat = np.zeros(num_clauses, dtype=bool)
        clause_sat_count = np.zeros(num_clauses, dtype=np.int8)
        
        # Vectorized check:
        # sat_matrix[i, k] = assignment[clause_vars[i, k]] == clause_signs[i, k]
        # But assignment is a list. Let's convert to numpy bool array for speed.
        # Actually, accessing numpy array by index is fast.
        
        # Let's do it in a loop for clarity, but optimize inner loop.
        # Or use numpy broadcasting if we have assignment as numpy array.
        
        # Let's keep assignment as a numpy bool array for the core loop to allow vectorized ops?
        # But we need to update it frequently.
        
        # Let's stick to the incremental update strategy for the main loop.
        # Initial computation:
        for i in range(num_clauses):
            cv = clause_vars[i]
            cs = clause_signs[i]
            count = 0
            for k in range(3):
                if assignment[cv[k]] == cs[k]:
                    count += 1
            clause_sat_count[i] = count
            if count > 0:
                clause_sat[i] = True
        return clause_sat, clause_sat_count

    # Main Search Loop
    max_restarts = 500
    # Adaptive steps per restart: start high, decrease? Or fixed?
    # Let's try a fixed number of steps, but with a time check.
    # 10000 steps per restart might be too many if we are slow.
    # Let's estimate speed.
    
    # Optimization: Instead of rebuilding unsat_clauses list every time,
    # we can pick a random clause index and check if it's unsat.
    # Or maintain a list of unsat indices and update it.
    # Updating the list is O(degree of var).
    # Picking random from list is O(1).
    # Let's maintain a list of unsat clause indices.
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = np.array(phase_list, dtype=bool)
        
        # Initialize satisfaction status
        clause_sat, clause_sat_count = update_clause_sat(assignment)
        
        # Build initial unsat list
        unsat_indices = np.where(~clause_sat)[0]
        
        if len(unsat_indices) == 0:
            return assignment.tolist()
            
        # Track best
        current_unsat = len(unsat_indices)
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.tolist()
            
        # WalkSAT parameters
        # p = probability of flipping a random variable in the clause
        # Let's use a static p = 0.25 or adaptive.
        p = 0.25
        
        # Steps per restart: let's do 5000
        steps_per_restart = 5000
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return best_assignment
                
            if len(unsat_indices) == 0:
                return assignment.tolist()
                
            # Pick a random unsat clause
            idx = rng.randint(0, len(unsat_indices) - 1)
            clause_idx = unsat_indices[idx]
            
            cv = clause_vars[clause_idx]
            cs = clause_signs[clause_idx]
            
            # Decide which variable to flip
            if rng.random() < p:
                # Random flip
                k = rng.randint(0, 2)
                v = cv[k]
            else:
                # Flip variable that minimizes delta (number of newly unsat clauses)
                best_v = -1
                best_delta = float('inf')
                
                for k in range(3):
                    v = cv[k]
                    # Calculate delta for flipping v
                    # Delta = (new unsat count - old unsat count)
                    # Only clauses containing v are affected.
                    
                    delta = 0
                    for c_idx in var_to_clauses[v]:
                        # Check if flipping v changes the satisfaction of c_idx
                        # We need to know the current state of c_idx.
                        
                        # If c_idx is currently satisfied:
                        #   It becomes unsat if v was the ONLY satisfying literal.
                        #   i.e., clause_sat_count[c_idx] == 1 and v satisfies it.
                        #   Wait, if v satisfies it and count is 1, then flipping v makes it 0 (unsat).
                        #   If v does NOT satisfy it, flipping v might make it satisfy? No, if it's already satisfied by others, flipping v (which was false) to true might add a satisfying literal, but count increases. It stays sat.
                        #   Actually, we just need to check the change in sat status.
                        
                        # Let's determine if v currently satisfies c_idx.
                        # We need the sign of v in c_idx.
                        # Find k_idx such that clause_vars[c_idx, k_idx] == v
                        # This is O(3) search.
                        
                        # Optimization: Pre-compute the position of each variable in its clauses?
                        # Or just search.
                        
                        v_satisfies_now = False
                        v_pos_in_clause = -1
                        for j in range(3):
                            if clause_vars[c_idx, j] == v:
                                v_pos_in_clause = j
                                if assignment[v] == clause_signs[c_idx, j]:
                                    v_satisfies_now = True
                                break
                                
                        if v_satisfies_now:
                            # v is currently satisfying this clause.
                            # If we flip v, it will no longer satisfy it.
                            # If this was the only satisfying literal (count == 1), the clause becomes unsat.
                            if clause_sat_count[c_idx] == 1:
                                delta += 1
                        else:
                            # v is currently NOT satisfying this clause.
                            # If we flip v, it will satisfy it.
                            # If the clause was unsat (count == 0), it becomes sat.
                            if clause_sat_count[c_idx] == 0:
                                delta -= 1
                    # End for c_idx in var_to_clauses[v]
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_v = v
                        
                v = best_v
                
            # Flip v
            assignment[v] = not assignment[v]
            
            # Update clause_sat and clause_sat_count for clauses containing v
            # Also update unsat_indices
            
            # We need to update the status of all clauses in var_to_clauses[v]
            # Let's collect the affected clause indices
            affected_clauses = var_to_clauses[v]
            
            # We will update clause_sat_count and clause_sat for these.
            # And then rebuild or update the unsat_indices list.
            
            # Rebuilding unsat_indices is O(C).
            # Updating incrementally is O(degree).
            # Let's try incremental update of the list.
            
            # However, Python lists are slow for removal/insertion.
            # Let's just rebuild unsat_indices if the number of affected clauses is small?
            # Or just rebuild every time? C is ~500-1000. Rebuilding is fast.
            
            # Let's update the counts first.
            for c_idx in affected_clauses:
                # Find position of v in c_idx
                for j in range(3):
                    if clause_vars[c_idx, j] == v:
                        if assignment[v] == clause_signs[c_idx, j]:
                            # v now satisfies
                            clause_sat_count[c_idx] += 1
                        else:
                            # v no longer satisfies
                            clause_sat_count[c_idx] -= 1
                        break
                
                # Update sat status
                if clause_sat_count[c_idx] > 0:
                    if not clause_sat[c_idx]:
                        clause_sat[c_idx] = True
                else:
                    if clause_sat[c_idx]:
                        clause_sat[c_idx] = False
            # End for c_idx in affected_clauses
            
            # Update unsat_indices
            # This is the tricky part.
            # Simple approach: rebuild
            # unsat_indices = np.where(~clause_sat)[0]
            
            # Optimized approach:
            # We know which clauses changed status.
            # But it's complex to maintain the list exactly.
            # Given C is small, let's just rebuild.
            unsat_indices = np.where(~clause_sat)[0]
            
            current_unsat = len(unsat_indices)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment.tolist()

    return best_assignment
# EVOLVE-BLOCK-END
