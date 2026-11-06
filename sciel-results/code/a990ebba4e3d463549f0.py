import time
import numpy as np
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 315.0  # 15s safety margin before 330s kill

    m = len(clauses)
    if m == 0:
        return [False] * n

    # Precompute clause structure as numpy arrays
    # clause_vars: (m, 3) int array of 0-based variable indices
    # clause_signs: (m, 3) bool array
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=bool)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = lit > 0

    # Build incidence matrix: A[v, c] = sign (1 if positive, -1 if negative, 0 if absent)
    A = np.zeros((n, m), dtype=np.int8)
    for c in range(m):
        for j in range(3):
            v = clause_vars[c, j]
            A[v, c] = 1 if clause_signs[c, j] else -1

    # For each variable, precompute which clauses it appears in and the sign
    var_clauses = [[] for _ in range(n)]
    var_signs = [[] for _ in range(n)]
    for v in range(n):
        idxs = np.where(A[v] != 0)[0]
        var_clauses[v] = idxs
        var_signs[v] = A[v, idxs]  # +1 or -1

    # Precompute pure literal info
    pos_count = np.zeros(n, dtype=np.int32)
    neg_count = np.zeros(n, dtype=np.int32)
    for c in range(m):
        for j in range(3):
            v = clause_vars[c, j]
            if clause_signs[c, j]:
                pos_count[v] += 1
            else:
                neg_count[v] += 1

    # Precompute for each clause: which variables and their signs (for fast access)
    # We'll use numpy arrays directly

    def init_clause_sat(assignment):
        """Compute satisfaction count per clause using matrix ops."""
        # assignment: bool array of length n
        # For each clause c, count how many of its 3 literals are satisfied
        # lit is satisfied if assignment[v] == sign
        # sign is +1 (positive) or -1 (negative)
        # satisfied: (assignment[v] and sign==+1) or (not assignment[v] and sign==-1)
        # = (assignment[v] == (sign==+1))
        
        # Build literal satisfaction matrix: L[c, j] = 1 if clause c, literal j is satisfied
        # L[c,j] = 1 if assignment[clause_vars[c,j]] == clause_signs[c,j]
        
        # Vectorized: for each clause position j (0,1,2), check satisfaction
        sat = np.zeros((m, 3), dtype=np.int8)
        for j in range(3):
            vars_j = clause_vars[:, j]  # (m,)
            signs_j = clause_signs[:, j]  # (m,) bool
            vals_j = assignment[vars_j]  # (m,) bool
            sat[:, j] = (vals_j == signs_j).astype(np.int8)
        
        return sat.sum(axis=1)  # (m,) clause_sat_count

    def flip_var(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count in-place."""
        old_val = assignment[v]
        assignment[v] = not old_val
        idxs = var_clauses[v]
        signs = var_signs[v]
        # For each clause this var appears in:
        # if new value matches sign: sat_count += 1
        # else: sat_count -= 1
        matches = (assignment[v] == (signs > 0))
        # matches: bool array, True where new value satisfies the literal
        clause_sat_count[idxs] += 2 * matches.astype(np.int32) - 1

    def compute_deltas(assignment, clause_sat_count):
        """
        For each variable, compute the change in number of unsatisfied clauses
        if we flip that variable.
        Returns: (n,) array where deltas[v] = new_unsat - old_unsat for flipping v.
        """
        deltas = np.zeros(n, dtype=np.int32)
        
        for v in range(n):
            idxs = var_clauses[v]
            signs = var_signs[v]
            if len(idxs) == 0:
                continue
            
            old_sat = clause_sat_count[idxs]  # (k,)
            # New sat after flip:
            # if assignment[v] was True and sign is +1: was satisfied, now not -> -1
            # if assignment[v] was True and sign is -1: was not satisfied, now is -> +1
            # if assignment[v] was False and sign is +1: was not satisfied, now is -> +1
            # if assignment[v] was False and sign is -1: was satisfied, now not -> -1
            
            # matches_old: was the literal satisfied before?
            old_matches = (assignment[v] == (signs > 0))
            # delta_sat: +1 if now satisfied, -1 if now not satisfied
            delta_sat = np.where(old_matches, -1, 1)  # (k,)
            
            new_sat = old_sat + delta_sat
            
            # Change in unsat count:
            # old_unsat: 1 if old_sat == 0
            # new_unsat: 1 if new_sat == 0
            old_unsat = (old_sat == 0).astype(np.int32)
            new_unsat = (new_sat == 0).astype(np.int32)
            
            deltas[v] = int(np.sum(new_unsat - old_unsat))
        
        return deltas

    def unit_propagate(assignment, clause_sat_count):
        """
        Assign variables that are forced by unit clauses.
        Returns True if a contradiction is detected.
        """
        changed = True
        while changed:
            changed = False
            # Find unit clauses (exactly one literal unsatisfied)
            unit_mask = (clause_sat_count == 1)
            if not np.any(unit_mask):
                break
            
            unit_clauses = np.where(unit_mask)[0]
            for c in unit_clauses:
                # Find the unsatisfied literal in this clause
                for j in range(3):
                    v = clause_vars[c, j]
                    sign = clause_signs[c, j]
                    if assignment[v] != sign:
                        # This literal is unsatisfied; assign it
                        if assignment[v] != sign:
                            flip_var(v, assignment, clause_sat_count)
                            changed = True
                        break
            # Check for empty clauses
            if np.any(clause_sat_count == 0):
                # Check if it's actually empty (all 3 unsatisfied)
                # clause_sat_count == 0 means all 3 literals unsatisfied
                # This is a conflict
                return True
        return False

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    # Track learned clauses (conflicts) for guidance
    learned_clauses = []  # list of (clause_vars, clause_signs, weight)
    max_learned = 50

    # Main search loop with multiple strategies
    total_flips_budget = 200000
    flip_count = 0
    restart_count = 0
    stagnation_counter = 0
    best_stagnation = 0

    while time.time() - start_time < timeout:
        # Random restart with occasional warm start from best
        if best_assignment is not None and restart_count > 2 and rng.random() < 0.25:
            assignment = best_assignment.copy()
        else:
            assignment = np.random.random(n) < 0.5

        clause_sat_count = init_clause_sat(assignment)

        # Check if satisfied
        if np.all(clause_sat_count > 0):
            return assignment.tolist()

        # Try pure literal elimination
        for v in range(n):
            if pos_count[v] > 0 and neg_count[v] == 0:
                if not assignment[v]:
                    flip_var(v, assignment, clause_sat_count)
            elif neg_count[v] > 0 and pos_count[v] == 0:
                if assignment[v]:
                    flip_var(v, assignment, clause_sat_count)

        if np.all(clause_sat_count > 0):
            return assignment.tolist()

        # Try unit propagation
        if unit_propagate(assignment, clause_sat_count):
            # Conflict: restart
            restart_count += 1
            continue

        if np.all(clause_sat_count > 0):
            return assignment.tolist()

        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # WalkSAT with adaptive scoring
        max_flips_this_restart = min(50000, max(1000, 200000 // max(1, restart_count)))
        
        for flip_num in range(max_flips_this_restart):
            if time.time() - start_time > timeout - 10.0:
                break
            if flip_count >= total_flips_budget:
                break

            # Find unsatisfied clauses
            unsat_mask = (clause_sat_count == 0)
            if not np.any(unsat_mask):
                return assignment.tolist()

            unsat_indices = np.where(unsat_mask)[0]
            n_unsat = len(unsat_indices)

            # Adaptive weight: start higher, decay
            progress = flip_num / max(1, max_flips_this_restart)
            weight = 0.5 * (1.0 - 0.5 * progress)

            if rng.random() < weight:
                # Random walk: pick random unsat clause, pick random var in it
                ci = int(rng.choice(unsat_indices))
                j = rng.randrange(3)
                v = int(clause_vars[ci, j])
            else:
                # Score-based: compute deltas for variables in unsat clauses
                # Only compute for variables that appear in unsat clauses (faster)
                candidate_vars = set()
                for ci in unsat_indices:
                    for j in range(3):
                        candidate_vars.add(int(clause_vars[ci, j]))
                
                if not candidate_vars:
                    v = rng.randrange(n)
                else:
                    candidate_list = list(candidate_vars)
                    n_cand = len(candidate_list)
                    
                    # Compute deltas only for candidates
                    cand_deltas = np.zeros(n_cand, dtype=np.int32)
                    for idx, v in enumerate(candidate_list):
                        idxs = var_clauses[v]
                        signs = var_signs[v]
                        if len(idxs) == 0:
                            cand_deltas[idx] = 0
                            continue
                        old_sat = clause_sat_count[idxs]
                        old_matches = (assignment[v] == (signs > 0))
                        delta_sat = np.where(old_matches, -1, 1)
                        new_sat = old_sat + delta_sat
                        old_unsat = (old_sat == 0).astype(np.int32)
                        new_unsat = (new_sat == 0).astype(np.int32)
                        cand_deltas[idx] = int(np.sum(new_unsat - old_unsat))
                    
                    min_delta = int(np.min(cand_deltas))
                    best_indices = np.where(cand_deltas == min_delta)[0]
                    v = int(candidate_list[rng.choice(best_indices)])

            flip_var(v, assignment, clause_sat_count)
            flip_count += 1

            current_unsat = int(np.sum(clause_sat_count == 0))
            if current_unsat == 0:
                return assignment.tolist()

            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Early termination if too much stagnation
            if stagnation_counter > 10000 and rng.random() < 0.1:
                break

        # If we learned something from conflicts, store it
        restart_count += 1

    if best_assignment is not None:
        return best_assignment.tolist()
    else:
        return [False] * n
