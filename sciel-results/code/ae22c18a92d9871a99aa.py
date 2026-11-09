# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285.0  # safety margin before 300s kill

    if m == 0:
        return [False] * n

    # Convert to 0-based, store polarity as +1/-1
    var_idx = np.zeros((m, 3), dtype=np.int32)
    signs = np.zeros((m, 3), dtype=np.int8)  # +1 or -1
    
    for ci, c in enumerate(clauses):
        for i, l in enumerate(c):
            var_idx[ci, i] = abs(l) - 1
            signs[ci, i] = 1 if l > 0 else -1

    # Build variable to clause incidence
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for i in range(3):
            v = var_idx[ci, i]
            var_to_clauses[v].append(ci)

    # Precompute for each variable: list of (clause_idx, position_in_clause)
    var_occurrences = [[] for _ in range(n)]
    for ci in range(m):
        for i in range(3):
            v = var_idx[ci, i]
            var_occurrences[v].append((ci, i))

    # Clause satisfaction tracking: sat_count[c] = number of satisfied literals in clause c
    def compute_sat_counts(assignment):
        """Compute satisfaction count for each clause."""
        target = (signs == 1)  # bool: True if positive literal
        sat = (assignment[var_idx] == target)  # (m,3) bool
        return sat.sum(axis=1)  # (m,) int

    def count_unsat(sat_counts):
        return int((sat_counts == 0).sum())

    def find_unsat(sat_counts):
        return np.where(sat_counts == 0)[0]

    def update_sat_counts_flip(assignment, v, sat_counts):
        """
        Flip variable v and update sat_counts efficiently.
        Returns new sat_counts (modified in place).
        """
        old_val = assignment[v]
        assignment[v] = not old_val
        
        for ci, pos in var_occurrences[v]:
            # Check if literal at (ci, pos) was satisfied before flip
            s = signs[ci, pos]
            was_sat = (old_val == (s == 1))
            # After flip, it's satisfied iff was not
            if was_sat:
                sat_counts[ci] -= 1
            else:
                sat_counts[ci] += 1
    
    def walksat_step(assignment, unsat_indices, sat_counts, rng, p=0.55):
        """
        WalkSAT step with efficient delta computation.
        Returns modified assignment and updated sat_counts.
        """
        if len(unsat_indices) == 0:
            return assignment, sat_counts
        
        # Choose a random unsatisfied clause
        ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
        
        best_var = -1
        best_delta = -999
        deltas = [0] * 3
        
        for i in range(3):
            v = var_idx[ci, i]
            s = signs[ci, i]
            
            # After flip, this literal becomes satisfied (since clause is unsat,
            # and we're flipping a variable in it)
            # Actually check: new_val = not assignment[v]
            # satisfied if new_val == (s == 1)
            new_val = not assignment[v]
            will_satisfy = (new_val == (s == 1))
            
            delta = 1 if will_satisfy else 0
            
            # Check impact on other clauses containing v
            for oc, opos in var_occurrences[v]:
                if oc == ci:
                    continue
                
                oc_sat_count = sat_counts[oc]
                
                if oc_sat_count > 0:
                    # Clause is currently satisfied
                    # If v's literal was the only satisfied one, flipping makes it unsat
                    # Check if v's literal is currently satisfied
                    os = signs[oc, opos]
                    v_lit_sat = (assignment[v] == (os == 1))
                    if v_lit_sat and oc_sat_count == 1:
                        # v is the only satisfied literal -> flipping unsatisfies clause
                        delta -= 1
                else:
                    # Clause is currently unsatisfied
                    # Flipping v might satisfy it if all other literals are already satisfied
                    # But since clause is unsat, at least one other literal is false
                    # Check if all other literals in oc are satisfied
                    others_sat = True
                    for j in range(3):
                        if j == opos:
                            continue
                        ov = var_idx[oc, j]
                        os = signs[oc, j]
                        if not (assignment[ov] == (os == 1)):
                            others_sat = False
                            break
                    if others_sat:
                        delta += 1
            
            deltas[i] = delta
            if delta > best_delta:
                best_delta = delta
                best_var = v
        
        # Choose variable: greedy with probability p, random with probability (1-p)
        if rng.random() < p:
            max_d = max(deltas)
            candidates = [var_idx[ci, i] for i in range(3) if deltas[i] == max_d]
            v_to_flip = candidates[rng.randint(0, len(candidates) - 1)]
        else:
            v_to_flip = var_idx[ci, rng.randint(0, 2)]
        
        update_sat_counts_flip(assignment, v_to_flip, sat_counts)
        return assignment, sat_counts

    # Main search
    rng = random.Random(42)
    best_assignment = None
    best_unsat = float('inf')
    
    # Adaptive parameters
    p_values = [0.55, 0.5, 0.6, 0.45, 0.65, 0.52, 0.58]
    
    max_restarts = 500
    max_steps = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        p = p_values[restart % len(p_values)]
        
        # Random initialization
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        sat_counts = compute_sat_counts(assignment)
        unsat_count = count_unsat(sat_counts)
        if unsat_count == 0:
            return assignment.tolist()
        
        local_best_unsat = unsat_count
        local_best_assign = assignment.copy()
        
        # Adaptive step limit: reduce if we've been stuck
        if best_unsat < 10 and restart > 50:
            steps_this_restart = max(500, max_steps // 2)
        else:
            steps_this_restart = max_steps
        
        for step in range(steps_this_restart):
            if time.time() > deadline:
                break
            
            unsat_count = count_unsat(sat_counts)
            if unsat_count == 0:
                return assignment.tolist()
            
            if unsat_count < local_best_unsat:
                local_best_unsat = unsat_count
                local_best_assign = assignment.copy()
                if local_best_unsat < best_unsat:
                    best_unsat = local_best_unsat
                    best_assignment = local_best_assign.copy()
            
            unsat_indices = find_unsat(sat_counts)
            assignment, sat_counts = walksat_step(assignment, unsat_indices, sat_counts, rng, p)
        
        if local_best_unsat < best_unsat:
            best_unsat = local_best_unsat
            best_assignment = local_best_assign.copy()
        
        if best_unsat == 0:
            return best_assignment.tolist()
    
    # Final polishing phase: try to fix remaining unsatisfied clauses
    if best_assignment is not None and best_unsat > 0 and best_unsat <= 50:
        if time.time() < deadline - 10:
            assignment = best_assignment.copy()
            sat_counts = compute_sat_counts(assignment)
            
            for polish_step in range(5000):
                if time.time() > deadline - 5:
                    break
                
                unsat_count = count_unsat(sat_counts)
                if unsat_count == 0:
                    return assignment.tolist()
                
                unsat_indices = find_unsat(sat_counts)
                # Use more aggressive randomization for polishing
                assignment, sat_counts = walksat_step(assignment, unsat_indices, sat_counts, rng, p=0.3)
            
            final_unsat = count_unsat(sat_counts)
            if final_unsat < best_unsat:
                best_unsat = final_unsat
                best_assignment = assignment.copy()
                if best_unsat == 0:
                    return best_assignment.tolist()
    
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
