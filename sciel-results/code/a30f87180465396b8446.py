"""Improved planted 3-SAT solver using incremental WalkSAT with simulated annealing."""
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: kill at 325s to stay under 330s wall-clock
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Build variable-to-clauses index for O(1) amortized delta computation
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(i)
    
    rng = random.Random(0)
    
    def random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]
    
    def compute_state(assignment):
        """Compute unsat count per clause and total unsat."""
        unsat_counts = [0] * m
        total_unsat = 0
        for i, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_counts[i] = 1
                total_unsat += 1
        return unsat_counts, total_unsat
    
    def flip_delta(var_idx, assignment, unsat_counts):
        """Compute delta in total unsat if we flip var_idx. Returns (delta, best_clause_idx_for_picking)."""
        # For each clause containing this variable, check if flipping changes its satisfaction
        delta = 0
        # We need to know current satisfaction status of each clause
        # unsat_counts[i] is 1 if unsat, 0 if sat
        # If clause is currently sat: flipping var might make it unsat if var was the only satisfying literal
        # If clause is currently unsat: flipping var might make it sat if var's literal becomes true
        
        # We need to track for each clause whether the variable is the unique satisfying literal
        # To do this efficiently, we need to know the current state of all literals in the clause
        # This is O(clause_size) per clause, but clauses have size 3, so it's bounded
        
        for clause_idx in var_to_clauses[var_idx]:
            clause = clauses[clause_idx]
            # Find the literal in this clause that corresponds to var_idx
            lit = None
            for l in clause:
                if abs(l) - 1 == var_idx:
                    lit = l
                    break
            if lit is None:
                continue
            
            currently_true = assignment[var_idx] == (lit > 0)
            
            if unsat_counts[clause_idx] == 0:
                # Clause is currently satisfied
                # Flipping var: lit becomes false. Is there another true literal?
                if currently_true:
                    # Check if any other literal in clause is true
                    has_other_true = False
                    for l in clause:
                        if abs(l) - 1 != var_idx and assignment[abs(l) - 1] == (l > 0):
                            has_other_true = True
                            break
                    if not has_other_true:
                        delta += 1  # Clause becomes unsat
            else:
                # Clause is currently unsat
                # Flipping var: lit becomes true (if currently false) or false (if currently true)
                if not currently_true:
                    # After flip, lit becomes true, so clause becomes sat
                    delta -= 1  # Clause becomes sat
                # If currently_true, clause is unsat despite this literal being true? 
                # That can't happen in 3-SAT if literal is true... wait, if literal is true, clause is sat.
                # So if unsat_counts[clause_idx] == 1, no literal is true, so currently_true must be False.
                # This branch is actually unreachable for valid SAT instances, but keep for safety.
        
        return delta
    
    def find_min_unsat_clause(unsat_counts):
        """Return index of a random clause with minimum unsat count among unsat clauses."""
        min_count = 3  # max possible
        candidates = []
        for i in range(m):
            if unsat_counts[i] > 0:
                if unsat_counts[i] < min_count:
                    min_count = unsat_counts[i]
                    candidates = [i]
                elif unsat_counts[i] == min_count:
                    candidates.append(i)
        if not candidates:
            return None
        return rng.choice(candidates)
    
    best_total_unsat = float('inf')
    best_assignment = None
    
    # Multiple restarts with simulated annealing
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
        
        assignment = random_assignment()
        unsat_counts, total_unsat = compute_state(assignment)
        
        if total_unsat == 0:
            return assignment
        
        if total_unsat < best_total_unsat:
            best_total_unsat = total_unsat
            best_assignment = assignment[:]
        
        # Simulated annealing parameters
        T_start = 1.0
        T_end = 0.01
        steps_per_restart = 50000
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, steps_per_restart))
        T = T_start
        
        for step in range(steps_per_restart):
            if time.time() >= deadline - 1.0:  # Safety margin
                break
            
            if total_unsat == 0:
                return assignment
            
            if total_unsat < best_total_unsat:
                best_total_unsat = total_unsat
                best_assignment = assignment[:]
            
            # Pick a random unsat clause
            unsat_indices = [i for i in range(m) if unsat_counts[i] > 0]
            if not unsat_indices:
                return assignment
            
            clause_idx = rng.choice(unsat_indices)
            clause = clauses[clause_idx]
            
            # Pick a literal from the clause to flip
            # Weight towards literals that would reduce unsat count
            var_choices = []
            for lit in clause:
                var_idx = abs(lit) - 1
                delta = flip_delta(var_idx, assignment, unsat_counts)
                var_choices.append((var_idx, delta))
            
            # Use soft-min to pick: prefer negative delta (reduces unsat)
            # Temperature controls randomness
            weights = []
            for var_idx, delta in var_choices:
                # Lower delta is better. Use exp(-delta/T) as weight
                if T > 0.001:
                    w = math.exp(-delta / T)
                else:
                    w = 1.0 if delta <= 0 else 0.0
                weights.append(w)
            
            # Normalize
            total_w = sum(weights)
            if total_w == 0:
                weights = [1.0] * len(weights)
                total_w = len(weights)
            
            r = rng.random() * total_w
            cumsum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    chosen_idx = i
                    break
            
            var_idx = var_choices[chosen_idx][0]
            delta = var_choices[chosen_idx][1]
            
            # Perform the flip
            assignment[var_idx] = not assignment[var_idx]
            total_unsat += delta
            
            # Update unsat_counts for affected clauses
            for c_idx in var_to_clauses[var_idx]:
                clause = clauses[c_idx]
                sat = False
                for lit in clause:
                    if assignment[abs(lit) - 1] == (lit > 0):
                        sat = True
                        break
                unsat_counts[c_idx] = 0 if sat else 1
            
            # Cool down
            T *= cooling_rate
        
        # End of restart
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return assignment
