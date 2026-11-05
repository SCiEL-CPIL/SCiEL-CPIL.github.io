"""Simulated Annealing + WalkSAT hybrid for planted 3-SAT (50 vars)."""
# EVOLVE-BLOCK-START
import random
import math
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Precompute variable-to-clause incidence for fast delta computation
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)
    
    deadline = time.time() + 280  # Safety margin: 330s limit, 280s for solving
    
    def make_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]
    
    def count_satisfied(assignment):
        sat = 0
        for clause in clauses:
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat += 1
                    break
        return sat
    
    def is_satisfied(assignment):
        for clause in clauses:
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    break
            else:
                return False
        return True
    
    def delta_satisfied(assignment, var_idx):
        """Compute change in # satisfied clauses if we flip var_idx."""
        delta = 0
        for ci in var_to_clauses[var_idx]:
            clause = clauses[ci]
            # Count satisfied literals in this clause under current assignment
            sat_count = 0
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat_count += 1
            # After flip, recompute
            new_sat_count = 0
            for lit in clause:
                v = abs(lit) - 1
                if v == var_idx:
                    # This literal flips
                    if not (assignment[v] == (lit > 0)):
                        new_sat_count += 1
                else:
                    if assignment[v] == (lit > 0):
                        new_sat_count += 1
            if sat_count == 0 and new_sat_count > 0:
                delta += 1
            elif sat_count > 0 and new_sat_count == 0:
                delta -= 1
        return delta
    
    best_assignment = None
    best_score = -1
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 1)
        assignment = make_assignment(rng)
        
        # Phase 1: Simulated Annealing
        T0 = 5.0
        T1 = 0.01
        alpha = 0.995
        T = T0
        current_score = count_satisfied(assignment)
        best_local = assignment[:]
        best_local_score = current_score
        
        steps = 0
        max_steps = 5000
        while T > T1 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            
            # Pick a variable to flip
            var_idx = rng.randint(0, n - 1)
            d = delta_satisfied(assignment, var_idx)
            
            if d >= 0 or (rng.random() < math.exp(d / T)):
                assignment[var_idx] = not assignment[var_idx]
                current_score += d
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = assignment[:]
                    if best_local_score == num_clauses:
                        break
            
            T *= alpha
        
        # Phase 2: WalkSAT polish
        if best_local_score == num_clauses:
            best_assignment = best_local
            break
        
        # Start WalkSAT from best local
        walk_assign = best_local[:]
        for step in range(20000):
            if time.time() > deadline:
                break
            # Find unsatisfied clauses
            unsat = []
            for ci, clause in enumerate(clauses):
                for lit in clause:
                    if walk_assign[abs(lit) - 1] == (lit > 0):
                        break
                else:
                    unsat.append(clause)
            
            if not unsat:
                best_assignment = walk_assign
                break
            
            # Pick clause with fewest satisfied literals (or random)
            clause = rng.choice(unsat)
            
            # Choose literal: with prob 0.7 pick one that satisfies most unsat clauses
            # Count how many unsat clauses each literal appears in
            if len(unsat) <= 200:
                lit_counts = {}
                for c in unsat:
                    for lit in c:
                        lit_counts[lit] = lit_counts.get(lit, 0) + 1
                # Weighted choice
                weights = []
                for lit in clause:
                    weights.append(lit_counts.get(lit, 0))
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen_lit = clause[-1]
                for lit, w in zip(clause, weights):
                    cum += w
                    if r <= cum:
                        chosen_lit = lit
                        break
            else:
                chosen_lit = rng.choice(clause)
            
            walk_assign[abs(chosen_lit) - 1] = not walk_assign[abs(chosen_lit) - 1]
        
        # Track global best
        final_score = count_satisfied(walk_assign)
        if final_score > best_score:
            best_score = final_score
            best_assignment = walk_assign[:]
        
        if best_score == num_clauses:
            break
    
    if best_assignment is None:
        best_assignment = make_assignment(random.Random(0))
    
    return best_assignment
# EVOLVE-BLOCK-END
