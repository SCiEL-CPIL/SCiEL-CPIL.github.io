"""Rewritten program with dynamic clause weights and adaptive restarts."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    """
    Solves a 3-SAT instance using Weighted WalkSAT with adaptive restarts.
    
    Key features:
    1. Dynamic clause weights: Unsatisfied clauses gain weight, biasing search.
    2. Adaptive restarts: Restart from best-so-far perturbed state on stagnation.
    3. Efficient incremental tracking of unsatisfied clauses.
    """
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Pre-process clauses: convert to 0-based indices and signs
    # clause_data[i] = list of (var_idx, sign) where sign is 1 or -1
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])
    
    num_clauses = len(clause_data)
    
    # Build index: var_idx -> list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append((ci, sign))
    
    # Helper to check if a clause is satisfied given an assignment
    def is_satisfied(ci, assignment):
        for var_idx, sign in clause_data[ci]:
            if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                return True
        return False
    
    def count_unsat(assignment):
        return sum(1 for ci in range(num_clauses) if not is_satisfied(ci, assignment))

    def solve_with_weights(rng, max_time_budget):
        """
        Core solver loop with dynamic weights and adaptive restarts.
        Returns the assignment if solved, else None.
        """
        # Initialize weights to 1.0
        weights = [1.0] * num_clauses
        # Decay factor for weights (to prevent infinite growth)
        weight_decay = 0.99
        # Increment for unsatisfied clauses
        weight_increment = 1.0
        
        # Stagnation limit: if no improvement in unsat count for this many steps, restart
        stagnation_limit = 100
        # Perturbation strength: number of bits to flip on restart
        perturbation_strength = 5
        
        best_assignment = None
        best_unsat = float('inf')
        
        # Initial random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment
        
        best_assignment = assignment[:]
        best_unsat = current_unsat
        
        unsat_clauses = set()
        for ci in range(num_clauses):
            if not is_satisfied(ci, assignment):
                unsat_clauses.add(ci)
        
        steps_since_improvement = 0
        local_step = 0
        
        while time.time() < deadline - max_time_budget:
            if not unsat_clauses:
                return assignment
            
            # 1. Update weights
            # Decay all weights
            for i in range(num_clauses):
                weights[i] *= weight_decay
            # Increment unsatisfied clauses
            for ci in unsat_clauses:
                weights[ci] += weight_increment
            
            # 2. Select a clause
            # Weighted random selection among unsatisfied clauses
            total_weight = sum(weights[ci] for ci in unsat_clauses)
            if total_weight == 0:
                # Fallback if weights are zero (shouldn't happen with increment)
                ci = rng.choice(list(unsat_clauses))
            else:
                r = rng.random() * total_weight
                cum = 0.0
                ci = None
                for c in unsat_clauses:
                    cum += weights[c]
                    if r <= cum:
                        ci = c
                        break
                if ci is None:
                    ci = list(unsat_clauses)[-1]
            
            clause = clause_data[ci]
            
            # 3. Select a variable to flip
            # Calculate delta for each variable in the clause
            # delta = (new_unsat_count - old_unsat_count)
            # We want to minimize delta.
            best_delta = float('inf')
            best_var_idx = -1
            best_sign = 0
            candidates = []
            
            for var_idx, sign in clause:
                # Calculate gain/loss
                gain = 0
                loss = 0
                for (uci, u_sign) in var_to_clauses[var_idx]:
                    if uci == ci:
                        # This literal is in the selected clause.
                        # If we flip var_idx, this literal becomes true (if it was false) or false (if it was true).
                        # Since the clause is unsatisfied, all literals are false.
                        # So flipping var_idx makes this literal true.
                        # This satisfies the clause ci.
                        gain += 1
                        continue
                    
                    if uci in unsat_clauses:
                        # Clause uci is currently unsatisfied.
                        # If we flip var_idx, does it satisfy uci?
                        # The literal in uci is currently false.
                        # Flipping var_idx makes it true.
                        # So it satisfies uci.
                        gain += 1
                    else:
                        # Clause uci is currently satisfied.
                        # If we flip var_idx, does it break uci?
                        # The literal in uci is currently true (or at least one is).
                        # If this specific literal is the only true one, flipping breaks it.
                        # We need to check if other literals in uci are true.
                        lit_true = (assignment[var_idx] and u_sign == 1) or (not assignment[var_idx] and u_sign == -1)
                        if lit_true:
                            # This literal is currently true.
                            # Check if any other literal in uci is true.
                            other_true = False
                            for (v2, s2) in clause_data[uci]:
                                if v2 == var_idx:
                                    continue
                                if (assignment[v2] and s2 == 1) or (not assignment[v2] and s2 == -1):
                                    other_true = True
                                    break
                            if not other_true:
                                # Flipping var_idx breaks uci.
                                loss += 1
                
                delta = loss - gain
                if delta < best_delta:
                    best_delta = delta
                    best_var_idx = var_idx
                    best_sign = sign
                    candidates = [var_idx]
                elif delta == best_delta:
                    candidates.append(var_idx)
            
            # Pick randomly among best candidates
            chosen_var_idx = rng.choice(candidates)
            
            # 4. Flip the variable
            assignment[chosen_var_idx] = not assignment[chosen_var_idx]
            
            # 5. Update unsat_clauses incrementally
            for (uci, u_sign) in var_to_clauses[chosen_var_idx]:
                was_unsat = uci in unsat_clauses
                is_now_satisfied = is_satisfied(uci, assignment)
                if was_unsat and is_now_satisfied:
                    unsat_clauses.discard(uci)
                elif (not was_unsat) and (not is_now_satisfied):
                    unsat_clauses.add(uci)
            
            current_unsat = len(unsat_clauses)
            
            # 6. Check for improvement and update best
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            local_step += 1
            
            # 7. Adaptive Restart
            if steps_since_improvement > stagnation_limit:
                # Restart from best_assignment with perturbation
                assignment = best_assignment[:]
                # Perturb
                for _ in range(perturbation_strength):
                    v = rng.randint(0, n - 1)
                    assignment[v] = not assignment[v]
                
                # Recalculate unsat set
                unsat_clauses = set()
                for ci in range(num_clauses):
                    if not is_satisfied(ci, assignment):
                        unsat_clauses.add(ci)
                
                current_unsat = len(unsat_clauses)
                steps_since_improvement = 0
                
                # Reset weights? Or keep them? 
                # Keeping them might be good to maintain bias towards hard clauses.
                # But if we restart from a different state, the "history" of unsat might be less relevant.
                # Let's decay them heavily to reset bias.
                for i in range(num_clauses):
                    weights[i] = 1.0

        # Time limit reached for this sub-routine, return best found
        if best_unsat == 0:
            return best_assignment
        return None

    # Main loop: multiple independent runs to cover the time budget
    # Each run gets a portion of the time.
    # We use a fixed seed for reproducibility in evaluation, but random for general use?
    # The task says "locally generated", so we can assume some structure.
    # Let's use a fixed seed for the solver's internal randomness to ensure deterministic behavior for a given instance?
    # No, better to use a seed derived from the instance or just a fixed one.
    # The prompt implies we should solve it. Let's use a fixed seed for the RNG to be consistent.
    rng = random.Random(42)
    
    # Divide time into chunks
    # 290 seconds total.
    # Let's try 10 restarts of the entire weighted walk, each with 29 seconds.
    # Or just one long run with adaptive restarts.
    # The adaptive restarts inside solve_with_weights already handle local minima.
    # Let's do multiple independent calls to solve_with_weights with decreasing time budgets?
    # Or just one call that runs until deadline?
    # The inner loop of solve_with_weights checks `time.time() < deadline - max_time_budget`.
    # If we pass max_time_budget=0, it runs until deadline.
    # But we might want to try different weight parameters?
    # Let's stick to one robust run.
    
    result = solve_with_weights(rng, 0.0)
    
    if result is not None:
        return result
    
    # If the main run didn't find a solution (returned None because time expired),
    # we return the best assignment found inside? 
    # The function solve_with_weights returns None if it times out without solving.
    # We lost the best_assignment. Let's modify solve_with_weights to return the best assignment even if not solved?
    # Or just do a final check?
    
    # Let's refactor slightly: solve_with_weights should return the best assignment found, and a flag if solved.
    # But the interface requires a list of booleans.
    # If not solved, any assignment is a guess. The best one (min unsat) is the best guess.
    
    # Let's re-run a simpler logic if the complex one fails?
    # Actually, let's just make sure solve_with_weights returns the best assignment.
    
    # Re-implementing the tail end to ensure we return something valid.
    # Since solve_with_weights returns None on timeout, we need to capture the best assignment.
    
    # Let's modify the inner function to return (assignment, solved)
    
    # --- Revised Inner Function ---
    def solve_inner(rng, max_time_budget):
        weights = [1.0] * num_clauses
        weight_decay = 0.99
        weight_increment = 1.0
        stagnation_limit = 100
        perturbation_strength = 5
        
        best_assignment = None
        best_unsat = float('inf')
        
        assignment = [rng.random() < 0.5 for _ in range(n)]
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment, True
        
        best_assignment = assignment[:]
        best_unsat = current_unsat
        
        unsat_clauses = set()
        for ci in range(num_clauses):
            if not is_satisfied(ci, assignment):
                unsat_clauses.add(ci)
        
        steps_since_improvement = 0
        
        while time.time() < deadline - max_time_budget:
            if not unsat_clauses:
                return assignment, True
            
            # Update weights
            for i in range(num_clauses):
                weights[i] *= weight_decay
            for ci in unsat_clauses:
                weights[ci] += weight_increment
            
            # Select clause
            total_weight = sum(weights[ci] for ci in unsat_clauses)
            if total_weight == 0:
                ci = rng.choice(list(unsat_clauses))
            else:
                r = rng.random() * total_weight
                cum = 0.0
                ci = None
                for c in unsat_clauses:
                    cum += weights[c]
                    if r <= cum:
                        ci = c
                        break
                if ci is None:
                    ci = list(unsat_clauses)[-1]
            
            clause = clause_data[ci]
            
            # Select variable
            best_delta = float('inf')
            candidates = []
            for var_idx, sign in clause:
                gain = 0
                loss = 0
                for (uci, u_sign) in var_to_clauses[var_idx]:
                    if uci == ci:
                        gain += 1
                    elif uci in unsat_clauses:
                        gain += 1
                    else:
                        lit_true = (assignment[var_idx] and u_sign == 1) or (not assignment[var_idx] and u_sign == -1)
                        if lit_true:
                            other_true = False
                            for (v2, s2) in clause_data[uci]:
                                if v2 == var_idx:
                                    continue
                                if (assignment[v2] and s2 == 1) or (not assignment[v2] and s2 == -1):
                                    other_true = True
                                    break
                            if not other_true:
                                loss += 1
                delta = loss - gain
                if delta < best_delta:
                    best_delta = delta
                    candidates = [var_idx]
                elif delta == best_delta:
                    candidates.append(var_idx)
            
            chosen_var_idx = rng.choice(candidates)
            assignment[chosen_var_idx] = not assignment[chosen_var_idx]
            
            # Update unsat
            for (uci, u_sign) in var_to_clauses[chosen_var_idx]:
                was_unsat = uci in unsat_clauses
                is_now_satisfied = is_satisfied(uci, assignment)
                if was_unsat and is_now_satisfied:
                    unsat_clauses.discard(uci)
                elif (not was_unsat) and (not is_now_satisfied):
                    unsat_clauses.add(uci)
            
            current_unsat = len(unsat_clauses)
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            if steps_since_improvement > stagnation_limit:
                assignment = best_assignment[:]
                for _ in range(perturbation_strength):
                    v = rng.randint(0, n - 1)
                    assignment[v] = not assignment[v]
                
                unsat_clauses = set()
                for ci in range(num_clauses):
                    if not is_satisfied(ci, assignment):
                        unsat_clauses.add(ci)
                
                current_unsat = len(unsat_clauses)
                steps_since_improvement = 0
                for i in range(num_clauses):
                    weights[i] = 1.0

        return best_assignment, (best_unsat == 0)

    # Run the solver
    final_assignment, solved = solve_inner(rng, 0.0)
    
    return final_assignment

# EVOLVE-BLOCK-END
