"""Synthetic planted 3-SAT solver (150 variables) with clause-learning WalkSAT."""
import time
import random
import math
from collections import defaultdict


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 300s kill

    # Precompute clause literals as (var_idx, polarity) for faster access
    # polarity is 1 if literal is positive (var is True), 0 if negative (var is False)
    clauses_opt = [[(abs(l) - 1, 1 if l > 0 else 0) for l in c] for c in clauses]
    num_clauses = len(clauses_opt)
    
    # Build variable to clause index map for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses_opt):
        for var_idx, polarity in clause:
            var_to_clauses[var_idx].append(i)

    # Build polarity lookup: var_to_clauses_polarity[var_idx] = dict {clause_idx: polarity}
    var_to_clauses_polarity = [defaultdict(int) for _ in range(n)]
    for i, clause in enumerate(clauses_opt):
        for var_idx, polarity in clause:
            var_to_clauses_polarity[var_idx][i] = polarity

    # State:
    # assignment: list of booleans (0/1)
    # clause_weights: list of ints, number of literals satisfied in each clause
    # unsat_indices: list of indices of unsatisfied clauses
    
    def init_state(assignment):
        """Initialize clause weights and unsat list from a given assignment."""
        clause_weights = [0] * num_clauses
        unsat_indices = []
        for i, clause in enumerate(clauses_opt):
            w = 0
            for var_idx, polarity in clause:
                if assignment[var_idx] == polarity:
                    w += 1
            clause_weights[i] = w
            if w == 0:
                unsat_indices.append(i)
        return clause_weights, unsat_indices

    def walksat_with_learning(assignment, rng, max_steps=5000):
        """
        WalkSAT with clause learning and better heuristics.
        Returns True if a solution is found, False otherwise.
        Modifies assignment in place.
        """
        clause_weights, unsat_indices = init_state(assignment)
        
        if not unsat_indices:
            return True

        # Track unsat clause set for O(1) lookup
        unsat_set = set(unsat_indices)
        
        # Variable activity scores (for VSIDS-like ordering)
        var_activity = [0.0] * n
        
        # Conflict trail for clause learning
        conflict_trail = []  # List of (var_idx, old_value) pairs
        
        # Learned clauses: list of lists of (var_idx, polarity)
        learned_clauses = []
        
        # Track best unsat count seen
        best_unsat_in_run = len(unsat_indices)
        best_assignment_in_run = assignment[:]
        
        # Adaptive noise parameter
        p_noise = 0.3  # Start with moderate noise
        
        for step in range(max_steps):
            if not unsat_indices:
                return True
            
            # Check deadline
            if time.time() > deadline:
                return False
            
            # Adaptive: increase noise if stuck
            if step % 500 == 0 and step > 0:
                # Increase noise slightly if we're not making progress
                p_noise = min(0.5, p_noise + 0.02)
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_indices)
            clause = clauses_opt[ci]
            
            # Calculate delta for each variable in the clause
            candidates = []
            for var_idx, polarity in clause:
                # Simulate flip
                delta = 0
                # 1. Current clause ci becomes sat. Delta -1.
                delta -= 1
                
                # 2. Check other clauses
                pol_map = var_to_clauses_polarity[var_idx]
                for j in var_to_clauses[var_idx]:
                    if j == ci:
                        continue
                    # Check if clause j is currently unsat
                    if clause_weights[j] == 0:
                        # It becomes sat
                        delta -= 1
                    else:
                        # It is currently sat.
                        pol_j = pol_map[j]
                        if assignment[var_idx] == pol_j:
                            # Literal is currently True. Flipping makes it False.
                            # Weight decreases by 1.
                            new_w = clause_weights[j] - 1
                            if new_w == 0:
                                # Becomes unsat
                                delta += 1
                
                # Add activity bonus to prefer variables with higher activity
                candidates.append((delta - var_activity[var_idx] * 0.1, var_idx, delta))
            
            # Pick the candidate with minimum effective delta
            # With noise: with prob p_noise, pick random; else pick best
            if rng.random() < p_noise:
                # Random move
                _, var_idx, delta = rng.choice(candidates)
            else:
                # Best move
                candidates.sort(key=lambda x: x[0])
                _, var_idx, delta = candidates[0]
            
            # Update activity
            var_activity[var_idx] += 1.0
            
            # Perform flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not assignment[var_idx]
            conflict_trail.append((var_idx, old_val))
            
            # Update state incrementally
            # 1. Update clause ci
            clause_weights[ci] = 1
            unsat_set.discard(ci)
            # Remove ci from unsat_indices list
            try:
                idx = unsat_indices.index(ci)
                unsat_indices.pop(idx)
            except ValueError:
                pass
            
            # 2. Update other clauses containing var_idx
            pol_map = var_to_clauses_polarity[var_idx]
            new_val = assignment[var_idx]
            for j in var_to_clauses[var_idx]:
                if j == ci:
                    continue
                
                pol_j = pol_map[j]
                old_lit_sat = (old_val == pol_j)
                new_lit_sat = (new_val == pol_j)
                
                if old_lit_sat and not new_lit_sat:
                    # Weight decreases by 1
                    clause_weights[j] -= 1
                    if clause_weights[j] == 0:
                        # Becomes unsat
                        unsat_indices.append(j)
                        unsat_set.add(j)
                elif not old_lit_sat and new_lit_sat:
                    # Weight increases by 1
                    clause_weights[j] += 1
                    # If it was unsat (weight 0), it becomes sat (weight 1)
                    if clause_weights[j] == 1 and j in unsat_set:
                        # Remove from unsat_indices
                        try:
                            idx = unsat_indices.index(j)
                            unsat_indices.pop(idx)
                        except ValueError:
                            pass
                        unsat_set.discard(j)

            if not unsat_indices:
                return True
            
            # Track best
            if len(unsat_indices) < best_unsat_in_run:
                best_unsat_in_run = len(unsat_indices)
                best_assignment_in_run = assignment[:]
            
            # Clause learning: if we're stuck (high unsat count), try to learn
            if step % 200 == 0 and len(unsat_indices) > best_unsat_in_run + 10 and len(conflict_trail) > 5:
                # Simple conflict analysis: find a variable that appears in many unsat clauses
                # and was recently flipped
                unsat_vars = defaultdict(int)
                for ci2 in unsat_indices:
                    for var_idx2, _ in clauses_opt[ci2]:
                        unsat_vars[var_idx2] += 1
                
                # Pick a variable with high unsat count that was recently flipped
                recent_vars = [v for v, _ in conflict_trail[-10:]]
                if recent_vars:
                    # Boost activity for variables in unsat clauses
                    for v in recent_vars:
                        if v in unsat_vars:
                            var_activity[v] += unsat_vars[v] * 0.5
                
                # Clear some of the trail to avoid memory issues
                if len(conflict_trail) > 100:
                    conflict_trail = conflict_trail[-50:]
        
        # If we didn't find a solution, restore best assignment found in this run
        if best_unsat_in_run < len(unsat_indices):
            assignment[:] = best_assignment_in_run
        return False

    # Strategy: Multiple restarts of WalkSAT with learning
    best_assignment = None
    best_unsat = float('inf')
    
    # Initial random assignment
    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    _, unsat_indices = init_state(assignment)
    best_unsat = len(unsat_indices)
    best_assignment = assignment[:]
    
    if best_unsat == 0:
        return assignment

    # Run WalkSAT with restarts
    num_restarts = 500
    max_steps_per_restart = 3000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # New random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT with learning
        success = walksat_with_learning(assignment, rng, max_steps=max_steps_per_restart)
        
        if success:
            return assignment
        
        # Track best
        _, unsat_indices = init_state(assignment)
        current_unsat = len(unsat_indices)
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]

    return best_assignment if best_assignment else [False] * n
