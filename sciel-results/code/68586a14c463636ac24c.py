"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-compute variable to clause indices mapping for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l) - 1].append(i)
            
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    rng = random.Random(42)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Weights for variables, used to bias random flips
    var_weights = [0.0] * n
    
    while time.time() < deadline:
        # 1. Initialize assignment
        # Use a mix of random and weight-influenced init? 
        # Simple random is usually fine, weights help in later restarts.
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # 2. Initial state calculation
        clause_satisfied = [False] * m
        unsat_count = 0
        for i, c in enumerate(clauses):
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            clause_satisfied[i] = sat
            if not sat:
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        stagnation = 0
        # Randomized restart threshold
        max_stagnation = 200 + rng.randint(0, 200)
        
        while time.time() < deadline:
            if unsat_count == 0:
                return assignment
                
            # Check stagnation
            stagnation += 1
            if stagnation > max_stagnation:
                break # Restart this run
            
            # Pick a random unsatisfied clause
            # Find an index where clause_satisfied is False
            # Using a list of unsat indices is faster, but maintaining it is complex.
            # For M~600, scanning is fast.
            unsat_indices = [i for i in range(m) if not clause_satisfied[i]]
            if not unsat_indices:
                return assignment
            c_idx = rng.choice(unsat_indices)
            c = clauses[c_idx]
            
            # Evaluate flips for literals in c
            best_delta = float('inf')
            best_lit_idx = -1
            flip_candidates = []
            
            for k, l in enumerate(c):
                var_idx = abs(l) - 1
                
                # Simulate flip
                # We need to check clauses containing var_idx
                # Save old state
                old_val = assignment[var_idx]
                new_val = not old_val
                
                delta = 0
                for cl_idx in var_to_clauses[var_idx]:
                    cl = clauses[cl_idx]
                    was_sat = clause_satisfied[cl_idx]
                    
                    # Check if clause is satisfied with new_val
                    is_sat = False
                    for ll in cl:
                        v_idx = abs(ll) - 1
                        if v_idx == var_idx:
                            if new_val == (ll > 0):
                                is_sat = True
                                break
                        else:
                            if assignment[v_idx] == (ll > 0):
                                is_sat = True
                                break
                    
                    if was_sat and not is_sat:
                        delta += 1
                    elif not was_sat and is_sat:
                        delta -= 1
                        
                flip_candidates.append((l, var_idx, delta))
                
                if delta < best_delta:
                    best_delta = delta
                    best_lit_idx = k
                    
            # Choose flip
            # 50% Greedy, 50% Random (weighted)
            if rng.random() < 0.5:
                chosen_k = best_lit_idx
            else:
                # Weighted random selection
                # Weights: 1.0 + var_weights[var_idx]
                weights = []
                for k, (l, var_idx, delta) in enumerate(flip_candidates):
                    w = 1.0 + var_weights[var_idx]
                    weights.append(w)
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen_k = 0
                for k, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_k = k
                        break
            
            l, var_idx, delta = flip_candidates[chosen_k]
            
            # Apply flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update state
            for cl_idx in var_to_clauses[var_idx]:
                cl = clauses[cl_idx]
                was_sat = clause_satisfied[cl_idx]
                
                # Check if clause is satisfied with new assignment
                is_sat = False
                for ll in cl:
                    v_idx = abs(ll) - 1
                    if assignment[v_idx] == (ll > 0):
                        is_sat = True
                        break
                
                if was_sat and not is_sat:
                    unsat_count += 1
                elif not was_sat and is_sat:
                    unsat_count -= 1
                clause_satisfied[cl_idx] = is_sat
                
            # Update weights
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                stagnation = 0 # Reset stagnation on improvement
                
            # Weight update logic
            if unsat_count == 0:
                return assignment
                
            if stagnation > 50:
                # Boost weights of variables in unsatisfied clauses
                for cl_idx in range(m):
                    if not clause_satisfied[cl_idx]:
                        for l in clauses[cl_idx]:
                            v_idx = abs(l) - 1
                            var_weights[v_idx] += 0.5
                # Decay all weights slightly to prevent explosion
                for i in range(n):
                    var_weights[i] *= 0.9
            else:
                # Slow decay
                for i in range(n):
                    var_weights[i] *= 0.99
                    
    return best_assignment
# EVOLVE-BLOCK-END
