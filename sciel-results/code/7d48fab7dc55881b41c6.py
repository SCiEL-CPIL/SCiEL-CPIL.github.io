import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    # Preprocessing
    # clauses are lists of signed 1-based literals
    # Convert to (var_idx, polarity)
    # var_idx 0-based, polarity True/False
    
    # Data structures
    clause_data = [] # list of lists of (var_idx, polarity)
    var_clauses = [[] for _ in range(n)] # list of lists of (clause_idx, polarity)
    
    for i, c in enumerate(clauses):
        lit_data = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            lit_data.append((v, p))
            var_clauses[v].append((i, p))
        clause_data.append(lit_data)
        
    m = len(clauses)
    
    # If m is very small relative to n, try a simpler approach
    if m < n:
        # Under-constrained: likely satisfiable with random assignment
        # Try a few random assignments
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            rng = random.Random(attempt * 13 + 7)
            assignment = [rng.random() < 0.5 for _ in range(n)]
            # Check if satisfied
            sat = True
            for c in clause_data:
                if not any(assignment[v] == p for v, p in c):
                    sat = False
                    break
            if sat:
                return assignment
        # Fall through to main solver
    
    # Spectral Clustering / Component Detection
    # We build a graph where variables are nodes and edges exist if they share a clause.
    # Then we find connected components.
    # If an instance decomposes into independent components, we can solve them separately.
    # Even if not fully independent, solving smaller sub-problems can be more efficient.
    # However, for a single planted 3-SAT instance of 150 vars, it's likely one big component.
    # But let's check for small components that might be trivially solvable or separable.
    
    # Build adjacency list for variables
    var_adj = [[] for _ in range(n)]
    for c in clauses:
        vars_in_clause = [abs(l) - 1 for l in c]
        for i in range(len(vars_in_clause)):
            for j in range(i+1, len(vars_in_clause)):
                u, v = vars_in_clause[i], vars_in_clause[j]
                if v not in var_adj[u]:
                    var_adj[u].append(v)
                if u not in var_adj[v]:
                    var_adj[v].append(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            # BFS
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                u = queue.pop(0)
                comp.append(u)
                for v in var_adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            components.append(comp)
    
    # If we have multiple components, we can solve them independently.
    # This is a powerful optimization if the instance is decomposable.
    # For a single large component, we fall back to the global solver.
    
    if len(components) > 1:
        # Solve each component separately
        # We need to map global variables to local variables for each component
        full_assignment = [False] * n
        
        time_per_comp = (deadline - time.time()) / len(components)
        
        for comp in components:
            if time.time() > deadline - 1: # Leave 1s for final assembly
                break
            
            comp_vars = comp
            n_comp = len(comp_vars)
            
            # Map global var index to local index
            global_to_local = {v: i for i, v in enumerate(comp_vars)}
            
            # Filter clauses that belong to this component
            # A clause belongs to a component if all its variables are in that component
            comp_clauses = []
            for i, c in enumerate(clauses):
                vars_in_c = [abs(l) - 1 for l in c]
                if all(v in global_to_local for v in vars_in_c):
                    # Convert to local indices
                    local_c = []
                    for l in c:
                        v = abs(l) - 1
                        p = l > 0
                        local_idx = global_to_local[v]
                        if p:
                            local_c.append(local_idx + 1)
                        else:
                            local_c.append(-(local_idx + 1))
                    comp_clauses.append(local_c)
            
            if not comp_clauses:
                # No constraints, any assignment works
                continue
            
            # Create a sub-instance
            sub_instance = {
                "n": n_comp,
                "clauses": comp_clauses
            }
            
            # We need a solver that works on a sub-instance.
            # To avoid recursion depth issues or overhead, we'll inline the solving logic 
            # or call a helper. Since solve() is the entry point, we can't easily call it recursively 
            # without changing the interface or passing state. 
            # However, the problem size is small (150 vars). 
            # If we have many small components, we can use a simpler solver for them.
            # Let's implement a simple WalkSAT for the sub-instance here.
            
            # For now, if we find multiple components, we assume they are small enough 
            # to be solved quickly with a dedicated local search.
            
            # Let's reuse the main solving logic but scoped to this component.
            # This is complex to inline. Instead, let's just run the main solver 
            # on the full instance if there's only one component.
            # If there are multiple, we'll solve them one by one.
            
            # Given the complexity of refactoring the solver to be callable on sub-instances 
            # without breaking the single-entry-point design, and the likelihood that 
            # planted 3-SAT instances are highly connected (one big component), 
            # we will only use this decomposition if it significantly reduces the problem size.
            
            # For this implementation, if len(components) > 1, we'll try to solve each component 
            # using a simplified WalkSAT.
            
            # Simplified WalkSAT for component
            n_sub = n_comp
            m_sub = len(comp_clauses)
            
            # Preprocess sub-instance
            sub_clause_data = []
            sub_var_clauses = [[] for _ in range(n_sub)]
            
            for i, c in enumerate(comp_clauses):
                lit_data = []
                for l in c:
                    v = abs(l) - 1
                    p = l > 0
                    lit_data.append((v, p))
                    sub_var_clauses[v].append((i, p))
                sub_clause_data.append(lit_data)
            
            # Solve sub-instance
            sub_assignment = [False] * n_sub
            solved = False
            
            # Try a few restarts
            for restart in range(500):
                if time.time() > deadline - 1:
                    break
                
                rng = random.Random(restart * 7 + 1)
                sub_assignment = [rng.random() < 0.5 for _ in range(n_sub)]
                
                sat_count = [0] * m_sub
                is_unsat = [False] * m_sub
                unsat_list = []
                
                for c_idx in range(m_sub):
                    cnt = 0
                    for v, p in sub_clause_data[c_idx]:
                        if sub_assignment[v] == p:
                            cnt += 1
                    sat_count[c_idx] = cnt
                    if cnt == 0:
                        is_unsat[c_idx] = True
                        unsat_list.append(c_idx)
                
                if not unsat_list:
                    solved = True
                    break
                
                # Walk
                for step in range(200):
                    if time.time() > deadline - 1:
                        break
                    
                    if not unsat_list:
                        solved = True
                        break
                    
                    c_idx = unsat_list[rng.randrange(len(unsat_list))]
                    if not is_unsat[c_idx]:
                        # Find a valid unsat clause
                        found = False
                        for idx in range(m_sub):
                            if is_unsat[idx]:
                                c_idx = idx
                                found = True
                                break
                        if not found:
                            solved = True
                            break
                    
                    # Min-conflicts
                    best_var = -1
                    best_score = float('inf')
                    
                    for v, p in sub_clause_data[c_idx]:
                        broken = 0
                        fixed = 0
                        for c2_idx, p2 in sub_var_clauses[v]:
                            was_sat = (sub_assignment[v] == p2)
                            if was_sat:
                                if sat_count[c2_idx] == 1:
                                    broken += 1
                            else:
                                if sat_count[c2_idx] == 0:
                                    fixed += 1
                        score = broken - fixed
                        if score < best_score:
                            best_score = score
                            best_var = v
                    
                    if rng.random() < 0.3:
                        v_to_flip = sub_clause_data[c_idx][rng.randrange(3)][0]
                    else:
                        v_to_flip = best_var
                    
                    if v_to_flip == -1:
                        continue
                    
                    old_val = sub_assignment[v_to_flip]
                    new_val = not old_val
                    sub_assignment[v_to_flip] = new_val
                    
                    for c2_idx, p2 in sub_var_clauses[v_to_flip]:
                        was_sat = (old_val == p2)
                        if was_sat:
                            sat_count[c2_idx] -= 1
                            if sat_count[c2_idx] == 0:
                                if not is_unsat[c2_idx]:
                                    is_unsat[c2_idx] = True
                                    unsat_list.append(c2_idx)
                        else:
                            sat_count[c2_idx] += 1
                            if sat_count[c2_idx] == 1:
                                if is_unsat[c2_idx]:
                                    is_unsat[c2_idx] = False
                                    # Remove from unsat_list? No, just mark as sat.
                                    # We'll filter unsat_list if it gets too big.
                    
                    # Clean up unsat_list occasionally
                    if len(unsat_list) > m_sub * 2:
                        unsat_list = [i for i in unsat_list if is_unsat[i]]
                    
                    if not unsat_list:
                        solved = True
                        break
            
            # Map sub_assignment back to full_assignment
            for local_idx, val in enumerate(sub_assignment):
                global_idx = comp_vars[local_idx]
                full_assignment[global_idx] = val
        
        # Verify solution
        # (Skipped for speed, assuming solver is correct)
        return full_assignment
    
    # Single component case: Use the main WalkSAT solver
    best_unsat_count = float('inf')
    best_assignment = None
    
    # Parameters tuned for 150 vars, ~450-600 clauses
    max_restarts = 3000 
    steps_per_restart = 200
    
    # Compute clause frequency for each variable to bias initialization
    var_pos_count = [0] * n  # count of positive literals
    var_neg_count = [0] * n  # count of negative literals
    for c in clause_data:
        for v, p in c:
            if p:
                var_pos_count[v] += 1
            else:
                var_neg_count[v] += 1
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 7 + 1)
        
        # Biased initialization: set variable to polarity that appears more often
        assignment = [False] * n
        for v in range(n):
            if var_pos_count[v] > var_neg_count[v]:
                assignment[v] = True
            elif var_neg_count[v] > var_pos_count[v]:
                assignment[v] = False
            else:
                assignment[v] = rng.random() < 0.5
        
        # Calc initial sat counts
        sat_count = [0] * m
        is_unsat = [False] * m
        unsat_list = []
        unsat_count = 0
        
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if assignment[v] == p:
                    cnt += 1
            sat_count[c_idx] = cnt
            if cnt == 0:
                is_unsat[c_idx] = True
                unsat_list.append(c_idx)
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Adaptive noise: start high, decrease as we make progress
        base_noise = 0.25 + 0.15 * (unsat_count / max(1, m))
        
        # Walk
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            # Pick random unsat clause
            c_idx = None
            for _ in range(5):
                idx = unsat_list[rng.randrange(len(unsat_list))]
                if is_unsat[idx]:
                    c_idx = idx
                    break
            
            if c_idx is None:
                # Fallback: scan for an unsat clause
                for idx in range(m):
                    if is_unsat[idx]:
                        c_idx = idx
                        break
                if c_idx is None:
                    return assignment

            # Evaluate flips for literals in c_idx
            best_var = -1
            best_score = float('inf')
            
            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0
                
                for c2_idx, p2 in var_clauses[v]:
                    was_sat = (assignment[v] == p2)
                    
                    if was_sat:
                        if sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        if sat_count[c2_idx] == 0:
                            fixed += 1
                            
                score = broken - fixed
                if score < best_score:
                    best_score = score
                    best_var = v
                    
            # Adaptive noise: more noise when stuck, less when making progress
            noise = base_noise * (1.0 + 0.5 * (unsat_count / max(1, m)))
            if rng.random() < noise:
                v_to_flip = clause_data[c_idx][rng.randrange(3)][0]
            else:
                v_to_flip = best_var
                
            if v_to_flip == -1:
                continue

            # Apply flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update state
            for c2_idx, p2 in var_clauses[v_to_flip]:
                was_sat = (old_val == p2)
                
                if was_sat:
                    sat_count[c2_idx] -= 1
                    if sat_count[c2_idx] == 0:
                        if not is_unsat[c2_idx]:
                            is_unsat[c2_idx] = True
                            unsat_list.append(c2_idx)
                            unsat_count += 1
                else:
                    sat_count[c2_idx] += 1
                    if sat_count[c2_idx] == 1:
                        if is_unsat[c2_idx]:
                            is_unsat[c2_idx] = False
                            unsat_count -= 1
                            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                
            # Early termination if we're stuck
            if step > 100 and unsat_count > 0.9 * m:
                break

    # Fallback
    if best_assignment is not None:
        return best_assignment
    return assignment
