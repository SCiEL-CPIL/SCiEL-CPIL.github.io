# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    deadline = time.time() + 325.0

    # Convert clauses to 0-based (var_idx, polarity)
    clauses_v = [[(abs(l)-1, 1 if l > 0 else -1) for l in c] for c in clauses]
    m = len(clauses_v)
    
    # Build index: for each variable, list of (clause_idx, polarity)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_v):
        for var, pol in c:
            var_to_clauses[var].append((ci, pol))
    
    # ============================================================
    # PHASE 1: DPLL with unit propagation
    # ============================================================
    
    def dpll_solver(deadline):
        """Simple DPLL solver. Returns assignment list of bools or None."""
        assign = [-1] * n
        
        def propagate(assign):
            """Unit propagation. Modifies assign in place. Returns False if contradiction."""
            changed = True
            while changed:
                changed = False
                for ci in range(m):
                    if len(clauses_v[ci]) == 0:
                        continue
                    unassigned = []
                    false_count = 0
                    for var, pol in clauses_v[ci]:
                        if assign[var] == -1:
                            unassigned.append((var, pol))
                        else:
                            if assign[var] == (pol == 1):
                                break
                            else:
                                false_count += 0
                    else:
                        if len(unassigned) == 0:
                            return False
                        elif len(unassigned) == 1:
                            var, pol = unassigned[0]
                            assign[var] = 1 if pol == 1 else 0
                            changed = True
            return True

        def dpll_recursive(assign, depth):
            if time.time() > deadline:
                return None
            
            all_sat = True
            for ci in range(m):
                sat = False
                for var, pol in clauses_v[ci]:
                    if assign[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            
            if all_sat:
                for i in range(n):
                    if assign[i] == -1:
                        assign[i] = 0
                return assign[:]

            if not propagate(assign):
                return None
            
            all_sat = True
            for ci in range(m):
                sat = False
                for var, pol in clauses_v[ci]:
                    if assign[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            
            if all_sat:
                for i in range(n):
                    if assign[i] == -1:
                        assign[i] = 0
                return assign[:]

            var_scores = [0] * n
            for ci in range(m):
                sat = False
                for var, pol in clauses_v[ci]:
                    if assign[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    for var, pol in clauses_v[ci]:
                        if assign[var] == -1:
                            var_scores[var] += 1
            
            best_var = -1
            max_score = 0
            for i in range(n):
                if assign[i] == -1 and var_scores[i] > max_score:
                    max_score = var_scores[i]
                    best_var = i
            
            if best_var == -1:
                return None

            assign[best_var] = 1
            result = dpll_recursive(assign, depth + 1)
            if result is not None:
                return result
            
            assign[best_var] = 0
            result = dpll_recursive(assign, depth + 1)
            if result is not None:
                return result
            
            assign[best_var] = -1
            return None

        return dpll_recursive(assign, 0)

    dpll_deadline = time.time() + 10.0
    dpll_result = dpll_solver(dpll_deadline)
    if dpll_result is not None:
        all_sat = True
        for c in clauses_v:
            sat = False
            for var, pol in c:
                if dpll_result[var] == (pol == 1):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
        if all_sat:
            return dpll_result
    
    # ============================================================
    # PHASE 2: Max-SAT style stochastic local search with dynamic penalty
    # ============================================================
    
    def maxsat_search(deadline):
        rng = random.Random(42)
        best_assignment = None
        best_unsat_count = float('inf')
        
        max_restarts = 2000
        base_steps = 5000
        max_steps = 20000
        
        # Dynamic penalty parameters
        base_penalty = 1.0
        penalty_growth = 1.05
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Initialize assignment
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Compute initial unsat clauses
            unsat_list = []
            for ci in range(m):
                sat = False
                for var, pol in clauses_v[ci]:
                    if assignment[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    unsat_list.append(ci)
            
            current_unsat = len(unsat_list)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment
            
            steps = min(base_steps + restart * 50, max_steps)
            penalty = base_penalty
            
            for step in range(steps):
                if time.time() > deadline - 2.0:
                    break
                
                if not unsat_list:
                    return assignment
                
                if len(unsat_list) < best_unsat_count:
                    best_unsat_count = len(unsat_list)
                    best_assignment = assignment[:]
                
                # Pick random unsat clause
                ci = rng.choice(unsat_list)
                clause = clauses_v[ci]
                
                # Score literals with dynamic penalty weighting
                best_var = -1
                best_score = float('inf')
                best_flip = False
                
                for var, pol in clause:
                    new_val = not assignment[var]
                    
                    newly_unsat = 0
                    newly_sat = 0
                    
                    for cj, cpol in var_to_clauses[var]:
                        c_j = clauses_v[cj]
                        
                        cur_sat = False
                        for v, p in c_j:
                            if assignment[v] == (p == 1):
                                cur_sat = True
                                break
                        
                        new_sat = False
                        for v, p in c_j:
                            if v == var:
                                if new_val == (p == 1):
                                    new_sat = True
                                    break
                            else:
                                if assignment[v] == (p == 1):
                                    new_sat = True
                                    break
                        
                        if cur_sat and not new_sat:
                            newly_unsat += 1
                        elif not cur_sat and new_sat:
                            newly_sat += 1
                    
                    # Weighted score: minimize newly unsat, maximize newly sat
                    # Penalty increases as we progress in the search
                    score = newly_unsat * penalty - newly_sat
                    
                    if score < best_score:
                        best_score = score
                        best_var = var
                        best_flip = new_val
                
                # Exploration: random flip with prob that decreases slightly
                # as penalty increases (more exploitation later in search)
                explore_prob = 0.3 / (1.0 + 0.1 * (penalty - 1.0))
                if rng.random() < explore_prob:
                    var, pol = rng.choice(clause)
                    new_val = not assignment[var]
                else:
                    var = best_var
                    new_val = best_flip
                
                if var == -1:
                    continue
                
                assignment[var] = new_val
                
                # Update unsat_list
                unsat_list = []
                for cj in range(m):
                    sat = False
                    for v, p in clauses_v[cj]:
                        if assignment[v] == (p == 1):
                            sat = True
                            break
                    if not sat:
                        unsat_list.append(cj)
                
                # Increase penalty if no improvement in last few steps
                # This encourages more aggressive exploitation
                if step % 100 == 0 and unsat_list:
                    penalty *= penalty_growth
                
                # Reset penalty occasionally to allow exploration
                if step % 1000 == 0:
                    penalty = base_penalty
            
            if not unsat_list:
                return assignment
        
        return best_assignment

    return maxsat_search(deadline)
# EVOLVE-BLOCK-END
