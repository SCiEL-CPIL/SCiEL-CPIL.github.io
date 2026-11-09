"""Synthetic planted 3-SAT solver (150 variables) with multiple strategies."""
import time
import random
import math


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

    def walksat(assignment, rng, max_steps=5000):
        """
        Efficient WalkSAT with incremental clause weight updates.
        Returns True if a solution is found, False otherwise.
        Modifies assignment in place.
        """
        clause_weights, unsat_indices = init_state(assignment)
        
        if not unsat_indices:
            return True

        # WalkSAT parameters
        # p is the probability of flipping a random literal in the clause
        # vs the best literal (minimizing weight)
        p = 0.5 
        
        for step in range(max_steps):
            if not unsat_indices:
                return True
            
            # Check deadline
            if time.time() > deadline:
                return False

            # Pick a random unsatisfied clause
            # Using a list of indices allows O(1) selection if we just pick random index from list
            # But the list changes. For simplicity and speed in Python, 
            # picking a random element from the list of unsat indices is fine.
            # To optimize, we could maintain a set or heap, but list is fast enough for n=150, m~600
            ci = rng.choice(unsat_indices)
            clause = clauses_opt[ci]
            
            # Choose variable to flip
            # Heuristic: flip variable that minimizes the weight of the clause after flip
            # With noise: with prob p, pick random var in clause; else pick best
            best_var_idx = None
            best_weight_delta = 100 # Max possible weight is 3, so delta is small
            best_new_weight = 100
            
            # We want to minimize the number of satisfied literals in the clause AFTER the flip?
            # No, standard WalkSAT: 
            # 1. Pick random unsat clause.
            # 2. With prob p, flip a random literal in the clause.
            # 3. With prob 1-p, flip the literal that satisfies the most OTHER clauses (or minimizes weight of this clause? No, usually minimizes weight of the clause is not the right metric for "other clauses").
            # Actually, the standard heuristic is: 
            # "WalkSAT: ... flip a literal in the clause that is currently false. 
            #  - With probability p, choose a literal at random.
            #  - With probability 1-p, choose the literal that satisfies the most other clauses."
            
            # However, a simpler and very effective variant for 3-SAT is:
            # Minimize the weight of the current clause? No, that would always flip to satisfy the clause if possible, which is just greedy.
            # The "noise" is what allows escaping local minima.
            
            # Let's use the "minimize weight of the clause" heuristic as the "best" move?
            # If we flip a literal that is currently False in the clause, it becomes True.
            # The weight of the clause increases by 1.
            # The weight of OTHER clauses might change.
            
            # Let's stick to a simpler, robust heuristic often used in competitive programming for small SAT:
            # Pick a random unsat clause.
            # For each variable in the clause:
            #   Calculate the "cost" of flipping it:
            #   - It satisfies this clause (weight +1)
            #   - It may unsatisfy other clauses (weight -1 for each)
            #   - It may satisfy other clauses (weight +1 for each)
            #   Net change in total unsat count is what we care about? 
            #   Or just local weight?
            
            # Let's implement a local search step that minimizes the total number of unsatisfied clauses.
            # This is essentially Local Search (MinConflicts style).
            
            # Find the variable in the clause that results in the largest decrease in total unsat clauses.
            # If tied, pick randomly.
            
            best_var = None
            best_delta_unsat = 0 # We want to maximize decrease (minimize increase)
            # Actually, let's look at the impact on the total unsat count.
            
            # Current total unsat count is len(unsat_indices).
            
            candidates = []
            for var_idx, polarity in clause:
                # Simulate flip
                # 1. This clause becomes satisfied (weight 0 -> 1). So it leaves unsat_indices.
                #    Wait, if weight was 0, and we flip a literal to satisfy it, weight becomes 1.
                #    So this clause is no longer unsat.
                
                # 2. Check other clauses containing var_idx
                #    For each other clause j containing var_idx:
                #    - If clause j was unsat (weight 0):
                #       - If the literal in j is satisfied by the flip, weight becomes 1 -> clause j becomes sat. (Unsat count -1)
                #       - If the literal in j is not satisfied by the flip, weight stays 0? 
                #         Wait, if weight was 0, all literals are false.
                #         If we flip var_idx, the literal for var_idx becomes true (if polarity matches) or false (if not).
                #         If polarity matches, weight becomes 1. Clause j becomes sat.
                #         If polarity doesn't match, weight stays 0? No, if polarity doesn't match, the literal was false (because assignment != polarity). 
                #         If we flip assignment, it becomes true? 
                #         Let's be precise.
                #         Literal l is (v, p). 
                #         Current value: assignment[v] == p ? True : False.
                #         New value: (not assignment[v]) == p ? True : False.
                #         If current is False (assignment[v] != p), then new is True (assignment[v] == p is false -> not assignment[v] == p is true? No.
                #         If assignment[v] != p, then assignment[v] is 1-p.
                #         New assignment is p.
                #         So new value is p == p -> True.
                #         So flipping a variable ALWAYS changes the value of the literal from False to True or True to False.
                
                #    So:
                #    - If clause j was unsat (weight 0):
                #       - The literal for var_idx in j was False.
                #       - After flip, it becomes True.
                #       - Weight becomes 1. Clause j becomes Satisfied.
                #       - Unsat count decreases by 1.
                #    - If clause j was sat (weight > 0):
                #       - If weight == 1:
                #         - The literal for var_idx in j was True (since it's the only one).
                #         - After flip, it becomes False.
                #         - Weight becomes 0. Clause j becomes UNSAT.
                #         - Unsat count increases by 1.
                #       - If weight > 1:
                #         - The literal for var_idx in j might be True or False.
                #         - If it was True, weight decreases by 1. If new weight > 0, still sat. If new weight == 0, becomes unsat.
                #         - If it was False, weight increases by 1. Still sat.
                
                # This is getting complicated to calculate exactly for "best" move without simulating.
                # Given n=150, m~600, we can afford to simulate the flip for the 3 variables in the clause.
                
                # Let's just count the change in unsat count for each of the 3 variables.
                delta = 0
                # 1. Current clause ci becomes sat. Delta -1.
                delta -= 1
                
                # 2. Check other clauses
                for j in var_to_clauses[var_idx]:
                    if j == ci:
                        continue
                    # Check if clause j is currently unsat
                    if clause_weights[j] == 0:
                        # It becomes sat
                        delta -= 1
                    else:
                        # It is currently sat.
                        # Check if the literal for var_idx in clause j is currently satisfied.
                        # Find the polarity of var_idx in clause j
                        # We need to know if assignment[var_idx] == polarity_j
                        # We can store polarity in a dict or just search (k=3, so search is fast)
                        pol_j = None
                        for v, p in clauses_opt[j]:
                            if v == var_idx:
                                pol_j = p
                                break
                        if assignment[var_idx] == pol_j:
                            # Literal is currently True. Flipping makes it False.
                            # Weight decreases by 1.
                            new_w = clause_weights[j] - 1
                            if new_w == 0:
                                # Becomes unsat
                                delta += 1
                        else:
                            # Literal is currently False. Flipping makes it True.
                            # Weight increases by 1.
                            # No change in sat/unsat status (was sat, remains sat)
                            pass
                
                candidates.append((delta, var_idx))
            
            # Pick the candidate with minimum delta (most negative is best)
            # If random noise is desired, we can mix in random choice.
            # Standard WalkSAT uses random with prob p.
            # Let's use a "soft" min: pick min delta, but with some probability pick random.
            # Or just pick min delta. This is pure local search (greedy).
            # To escape local minima, we need random restarts or random moves.
            # Let's add a small probability of picking a random candidate.
            
            if rng.random() < 0.1:
                # Random move
                _, var_idx = rng.choice(candidates)
            else:
                # Best move
                candidates.sort(key=lambda x: x[0])
                _, var_idx = candidates[0]
            
            # Perform flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update state incrementally
            # 1. Update clause ci
            # It was unsat (weight 0). Now we flipped a var in it.
            # It must be sat now? 
            # Wait, we flipped a var in the clause. 
            # The literal for that var in the clause was False (since clause was unsat).
            # Now it is True.
            # So weight becomes 1.
            clause_weights[ci] = 1
            # Remove ci from unsat_indices
            # unsat_indices is a list. Removing an element is O(N).
            # We can mark it as -1 or use a set? 
            # For efficiency, let's just rebuild or use a list and filter periodically?
            # Or better: maintain a set of unsat indices? 
            # Set lookup is O(1). 
            # But we need to iterate? No, we just pick random.
            # Let's switch unsat_indices to a set for O(1) removal/addition.
            # But rng.choice requires a sequence.
            # We can maintain a list and a set, or just list and accept O(N) removal.
            # N_unsat is small usually.
            
            # Let's stick to list for simplicity, but optimize removal.
            # Find index of ci in unsat_indices
            try:
                idx = unsat_indices.index(ci)
                unsat_indices.pop(idx)
            except ValueError:
                pass # Should not happen
            
            # 2. Update other clauses containing var_idx
            for j in var_to_clauses[var_idx]:
                if j == ci:
                    continue
                
                # Determine old and new status of literal in clause j
                # Find polarity
                pol_j = None
                for v, p in clauses_opt[j]:
                    if v == var_idx:
                        pol_j = p
                        break
                
                old_val = assignment[var_idx] # This is already flipped! 
                # We need to know if the literal WAS satisfied.
                # Before flip: assignment_old[var_idx]
                # After flip: assignment_new[var_idx]
                # We flipped assignment[var_idx].
                # So old_val = not assignment[var_idx]
                
                old_val = not assignment[var_idx]
                new_val = assignment[var_idx]
                
                old_lit_sat = (old_val == pol_j)
                new_lit_sat = (new_val == pol_j)
                
                if old_lit_sat and not new_lit_sat:
                    # Weight decreases by 1
                    clause_weights[j] -= 1
                    if clause_weights[j] == 0:
                        # Becomes unsat
                        unsat_indices.append(j)
                elif not old_lit_sat and new_lit_sat:
                    # Weight increases by 1
                    clause_weights[j] += 1
                    # If it was unsat (weight 0), it becomes sat (weight 1)
                    if clause_weights[j] == 1:
                        # Remove from unsat_indices
                        try:
                            idx = unsat_indices.index(j)
                            unsat_indices.pop(idx)
                        except ValueError:
                            pass

            if not unsat_indices:
                return True

        return False

    # Strategy: Multiple restarts of WalkSAT
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
    # Since WalkSAT is fast now, we can do many restarts.
    # Each restart: random assignment, run WalkSAT for max_steps.
    # If found, return.
    
    num_restarts = 1000
    max_steps_per_restart = 2000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # New random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Run WalkSAT
        success = walksat(assignment, rng, max_steps=max_steps_per_restart)
        
        if success:
            return assignment
        
        # Track best
        _, unsat_indices = init_state(assignment)
        current_unsat = len(unsat_indices)
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]

    return best_assignment if best_assignment else [False] * n
