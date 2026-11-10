"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    """Simulated Annealing SAT solver with incremental sat_count tracking."""
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    # Pre-extract clause literals as (var_idx, sign)
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])
    
    num_clauses = len(clause_data)

    # Build index: var_idx -> list of (clause_idx, sign_in_that_clause)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append((ci, sign))

    rng = random.Random(42)
    
    # SA Parameters
    T_initial = 10.0
    alpha = 0.99
    min_T = 0.01
    
    # Multi-restart strategy
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize sat_count: number of true literals in each clause
        sat_count = [0] * num_clauses
        unsat_clauses_list = []
        unsat_count = 0
        
        for ci, clause in enumerate(clause_data):
            count = 0
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    count += 1
            sat_count[ci] = count
            if count == 0:
                unsat_count += 1
                unsat_clauses_list.append(ci)
        
        if unsat_count == 0:
            return assignment

        T = T_initial
        steps = 0
        max_steps = 50000 # Steps per restart
        
        while steps < max_steps:
            if time.time() > deadline:
                return assignment
            
            if unsat_count == 0:
                return assignment

            steps += 1
            
            # 1. Select a move
            # Pick a random unsatisfied clause
            if not unsat_clauses_list:
                return assignment
            
            ci = rng.choice(unsat_clauses_list)
            # Pick a random variable in this clause to flip
            var_idx, sign = clause_data[ci][rng.randrange(3)]
            
            # 2. Evaluate the delta of unsat_count if we flip var_idx
            # We iterate over all clauses containing var_idx
            delta_unsat = 0
            
            for (j, s_j) in var_to_clauses[var_idx]:
                # Check current truth value of literal (var_idx, s_j)
                # lit_true = (assignment[var_idx] and s_j == 1) or (not assignment[var_idx] and s_j == -1)
                # Instead of checking assignment, we can just look at sat_count logic?
                # No, we need to know if the literal is currently true or false.
                
                # Current value of var_idx is assignment[var_idx]
                # Literal is true iff (assignment[var_idx] == (s_j == 1))
                # i.e., if s_j==1, true if assignment is True. If s_j==-1, true if assignment is False.
                lit_is_true = (assignment[var_idx] and s_j == 1) or (not assignment[var_idx] and s_j == -1)
                
                if lit_is_true:
                    # Literal is currently True. After flip, it becomes False.
                    # sat_count[j] will decrease by 1.
                    if sat_count[j] == 1:
                        # It was the only true literal. Clause becomes UNSAT.
                        delta_unsat += 1
                else:
                    # Literal is currently False. After flip, it becomes True.
                    # sat_count[j] will increase by 1.
                    if sat_count[j] == 0:
                        # It was the only true literal? No, count was 0.
                        # Now it becomes 1. Clause becomes SAT.
                        delta_unsat -= 1
            
            # 3. Acceptance Rule
            accept = False
            if delta_unsat < 0:
                accept = True
            else:
                # Accept with probability exp(-delta / T)
                if T > min_T:
                    prob = math.exp(-delta_unsat / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                # Apply the flip
                assignment[var_idx] = not assignment[var_idx]
                
                # Update sat_count and unsat_clauses_list
                for (j, s_j) in var_to_clauses[var_idx]:
                    # Determine if literal (var_idx, s_j) is now true or false
                    # It just flipped, so its state is inverted.
                    # Previously:
                    # lit_was_true = (assignment_old[var_idx] and s_j == 1) ...
                    # Now:
                    # lit_is_true_now = (assignment_new[var_idx] and s_j == 1) ...
                    
                    # We can just check the new state
                    lit_is_true_now = (assignment[var_idx] and s_j == 1) or (not assignment[var_idx] and s_j == -1)
                    
                    if lit_is_true_now:
                        # Literal became True. sat_count increases.
                        if sat_count[j] == 0:
                            # Clause was unsat, now sat.
                            unsat_count -= 1
                            # Remove j from unsat_clauses_list
                            # Find index of j
                            # Optimization: Instead of linear search, we can just mark it?
                            # Or swap-remove.
                            try:
                                idx = unsat_clauses_list.index(j)
                                unsat_clauses_list[idx] = unsat_clauses_list[-1]
                                unsat_clauses_list.pop()
                            except ValueError:
                                pass # Should not happen if logic is correct
                        sat_count[j] += 1
                    else:
                        # Literal became False. sat_count decreases.
                        if sat_count[j] == 1:
                            # Clause was sat (count 1), now unsat (count 0).
                            unsat_count += 1
                            unsat_clauses_list.append(j)
                        sat_count[j] -= 1
                        
                # Cool down
                T *= alpha
            else:
                # Rejected, no state change
                # Still cool down? Usually yes in SA.
                T *= alpha
                
        # End of restart
        # If we finished a restart without solving, we can check if this is the best solution
        # but we return the first solution found.
        
    # If no solution found within deadline, return last assignment
    # We need to keep track of the best assignment? 
    # The problem asks for a satisfying assignment. If none found, score is 0 for that instance?
    # "Score: fraction of satisfiable instances solved". 
    # So if we don't find a solution, it's a fail. 
    # Returning a random assignment is fine, it just won't be satisfied.
    
    return assignment

# EVOLVE-BLOCK-END
