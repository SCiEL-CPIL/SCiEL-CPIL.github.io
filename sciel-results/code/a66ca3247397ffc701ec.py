# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 300s limit, use 285s to be safe
    start_time = time.time()
    deadline = start_time + 285.0
    
    # Preprocess: convert to 0-based indices with signs
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])
    
    # Build variable-to-clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for vi, _ in clause_vars[ci]:
            var_to_clauses[vi].append(ci)
    
    # Spectral clustering: build incidence graph and find connected components
    # Two variables are connected if they appear in the same clause
    # Build adjacency matrix
    row_indices = []
    col_indices = []
    for ci in range(m):
        vars_in_clause = [vi for vi, _ in clause_vars[ci]]
        for i in range(len(vars_in_clause)):
            for j in range(len(vars_in_clause)):
                if i != j:
                    row_indices.append(vars_in_clause[i])
                    col_indices.append(vars_in_clause[j])
    
    if row_indices:
        data = np.ones(len(row_indices), dtype=np.float64)
        adj_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    else:
        adj_matrix = csr_matrix((n, n))
    
    # Find connected components
    n_components, component_labels = connected_components(adj_matrix, directed=False)
    
    # Group variables by component
    components = {}
    for vi in range(n):
        comp = component_labels[vi]
        if comp not in components:
            components[comp] = []
        components[comp].append(vi)
    
    # Sort components by size (largest first)
    sorted_components = sorted(components.keys(), key=lambda c: len(components[c]), reverse=True)
    
    # Assign time budget to each component proportional to its clause count
    comp_clause_counts = {}
    for comp in sorted_components:
        comp_vars = set(components[comp])
        count = 0
        for ci in range(m):
            if any(vi in comp_vars for vi, _ in clause_vars[ci]):
                count += 1
        comp_clause_counts[comp] = count
    
    total_clause_weight = sum(comp_clause_counts.values())
    if total_clause_weight == 0:
        total_clause_weight = 1
    
    # Allocate time budget with safety margin
    total_budget = deadline - start_time - 5.0  # 5s safety margin
    comp_time_budgets = {}
    for comp in sorted_components:
        # Proportional allocation with minimum 1s for small components
        weight = comp_clause_counts[comp] / total_clause_weight
        budget = max(1.0, weight * total_budget)
        comp_time_budgets[comp] = min(budget, total_budget)
    
    # Solve each component independently
    final_assignment = [False] * n
    solved_components = []
    
    for comp in sorted_components:
        if time.time() > deadline - 5.0:
            break
        
        comp_vars = components[comp]
        comp_var_set = set(comp_vars)
        comp_size = len(comp_vars)
        
        # Find clauses that belong to this component
        comp_clauses = []
        for ci in range(m):
            if all(vi in comp_var_set for vi, _ in clause_vars[ci]):
                comp_clauses.append(ci)
        
        if not comp_clauses:
            # No clauses, assign arbitrarily
            for vi in comp_vars:
                final_assignment[vi] = False
            continue
        
        # Create local problem for this component
        # Map global var indices to local 0..comp_size-1
        var_map = {vi: i for i, vi in enumerate(comp_vars)}
        inv_var_map = {i: vi for i, vi in enumerate(comp_vars)}
        
        local_clauses = []
        for ci in comp_clauses:
            local_clause = []
            for vi, sign in clause_vars[ci]:
                local_clause.append((var_map[vi], sign))
            local_clauses.append(local_clause)
        
        local_var_to_clauses = [[] for _ in range(comp_size)]
        for ci in range(len(local_clauses)):
            for vi, _ in local_clauses[ci]:
                local_var_to_clauses[vi].append(ci)
        
        # Time budget for this component
        comp_deadline = start_time + comp_time_budgets[comp]
        comp_deadline = min(comp_deadline, deadline - 5.0)
        
        # Solve the component using WalkSAT + local search
        rng = random.Random(hash(comp) % (2**32))
        
        def local_evaluate(assignment):
            unsat = 0
            for ci in range(len(local_clauses)):
                sat = False
                for vi, sign in local_clauses[ci]:
                    if assignment[vi] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    unsat += 1
            return unsat
        
        def local_walksat_step(assignment, rng, p_flip=0.55):
            unsat_clauses = []
            for ci in range(len(local_clauses)):
                sat = False
                for vi, sign in local_clauses[ci]:
                    if assignment[vi] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return assignment
            
            ci = rng.choice(unsat_clauses)
            clause = local_clauses[ci]
            
            if rng.random() < p_flip:
                vi, sign = rng.choice(clause)
                assignment[vi] = not assignment[vi]
            else:
                best_var = None
                best_cost = float('inf')
                for vi, sign in clause:
                    old_val = assignment[vi]
                    assignment[vi] = not old_val
                    cost = 0
                    for cj in local_var_to_clauses[vi]:
                        sat = False
                        for vj, sj in local_clauses[cj]:
                            if assignment[vj] == (sj == 1):
                                sat = True
                                break
                        if not sat:
                            cost += 1
                    assignment[vi] = old_val
                    if cost < best_cost:
                        best_cost = cost
                        best_var = vi
                
                if best_var is not None:
                    assignment[best_var] = not assignment[best_var]
            
            return assignment
        
        def local_search(assignment, rng, max_iters=3000):
            for _ in range(max_iters):
                if time.time() > comp_deadline:
                    break
                
                unsat_clauses = []
                for ci in range(len(local_clauses)):
                    sat = False
                    for vi, sign in local_clauses[ci]:
                        if assignment[vi] == (sign == 1):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(ci)
                
                if not unsat_clauses:
                    break
                
                ci = rng.choice(unsat_clauses)
                clause = local_clauses[ci]
                
                best_var = None
                best_cost = float('inf')
                for vi, sign in clause:
                    old_val = assignment[vi]
                    assignment[vi] = not old_val
                    cost = 0
                    for cj in local_var_to_clauses[vi]:
                        sat = False
                        for vj, sj in local_clauses[cj]:
                            if assignment[vj] == (sj == 1):
                                sat = True
                                break
                        if not sat:
                            cost += 1
                    assignment[vi] = old_val
                    if cost < best_cost:
                        best_cost = cost
                        best_var = vi
                
                if best_var is not None and best_cost < float('inf'):
                    assignment[best_var] = not assignment[best_var]
                else:
                    break
            
            return assignment
        
        # Multiple restarts for this component
        best_local_assignment = None
        best_local_unsat = float('inf')
        restart = 0
        
        while time.time() < comp_deadline:
            restart += 1
            
            local_assignment = [rng.random() < 0.5 for _ in range(comp_size)]
            
            p_flip = 0.5 + 0.2 * rng.random()
            steps = 2000 + 2000 * rng.random()
            
            for _ in range(int(steps)):
                if time.time() > comp_deadline:
                    break
                local_walksat_step(local_assignment, rng, p_flip)
            
            local_assignment = local_search(local_assignment, rng, max_iters=1000)
            
            unsat = local_evaluate(local_assignment)
            if unsat == 0:
                best_local_assignment = local_assignment
                best_local_unsat = 0
                break
            
            if unsat < best_local_unsat:
                best_local_unsat = unsat
                best_local_assignment = local_assignment[:]
        
        # Assign solution to global variables
        if best_local_assignment is not None:
            for i, vi in enumerate(comp_vars):
                final_assignment[vi] = best_local_assignment[i]
            solved_components.append(comp)
    
    # If some components weren't solved, try global fallback with remaining time
    unsolved_comps = [c for c in sorted_components if c not in solved_components]
    if unsolved_comps and time.time() < deadline - 3.0:
        # Combine unsolved variables and try global search on just those
        unsolved_vars = []
        for comp in unsolved_comps:
            unsolved_vars.extend(components[comp])
        
        if unsolved_vars:
            unsolved_set = set(unsolved_vars)
            
            # Find clauses involving unsolved vars
            relevant_clauses = []
            for ci in range(m):
                if any(vi in unsolved_set for vi, _ in clause_vars[ci]):
                    relevant_clauses.append(ci)
            
            if relevant_clauses:
                # Create subproblem with only unsolved vars and their clauses
                var_map = {vi: i for i, vi in enumerate(unsolved_vars)}
                inv_var_map = {i: vi for i, vi in enumerate(unsolved_vars)}
                sub_size = len(unsolved_vars)
                
                sub_clauses = []
                for ci in relevant_clauses:
                    sub_clause = []
                    for vi, sign in clause_vars[ci]:
                        if vi in unsolved_set:
                            sub_clause.append((var_map[vi], sign))
                    if sub_clause:
                        sub_clauses.append(sub_clause)
                
                sub_var_to_clauses = [[] for _ in range(sub_size)]
                for ci in range(len(sub_clauses)):
                    for vi, _ in sub_clauses[ci]:
                        sub_var_to_clauses[vi].append(ci)
                
                rng = random.Random(12345)
                
                def sub_evaluate(assignment):
                    unsat = 0
                    for ci in range(len(sub_clauses)):
                        sat = False
                        for vi, sign in sub_clauses[ci]:
                            if assignment[vi] == (sign == 1):
                                sat = True
                                break
                        if not sat:
                            unsat += 1
                    return unsat
                
                def sub_walksat_step(assignment, rng, p_flip=0.55):
                    unsat_clauses = []
                    for ci in range(len(sub_clauses)):
                        sat = False
                        for vi, sign in sub_clauses[ci]:
                            if assignment[vi] == (sign == 1):
                                sat = True
                                break
                        if not sat:
                            unsat_clauses.append(ci)
                    
                    if not unsat_clauses:
                        return assignment
                    
                    ci = rng.choice(unsat_clauses)
                    clause = sub_clauses[ci]
                    
                    if rng.random() < p_flip:
                        vi, sign = rng.choice(clause)
                        assignment[vi] = not assignment[vi]
                    else:
                        best_var = None
                        best_cost = float('inf')
                        for vi, sign in clause:
                            old_val = assignment[vi]
                            assignment[vi] = not old_val
                            cost = 0
                            for cj in sub_var_to_clauses[vi]:
                                sat = False
                                for vj, sj in sub_clauses[cj]:
                                    if assignment[vj] == (sj == 1):
                                        sat = True
                                        break
                                if not sat:
                                    cost += 1
                            assignment[vi] = old_val
                            if cost < best_cost:
                                best_cost = cost
                                best_var = vi
                        
                        if best_var is not None:
                            assignment[best_var] = not assignment[best_var]
                    
                    return assignment
                
                def sub_local_search(assignment, rng, max_iters=2000):
                    for _ in range(max_iters):
                        if time.time() > deadline - 3.0:
                            break
                        
                        unsat_clauses = []
                        for ci in range(len(sub_clauses)):
                            sat = False
                            for vi, sign in sub_clauses[ci]:
                                if assignment[vi] == (sign == 1):
                                    sat = True
                                    break
                            if not sat:
                                unsat_clauses.append(ci)
                        
                        if not unsat_clauses:
                            break
                        
                        ci = rng.choice(unsat_clauses)
                        clause = sub_clauses[ci]
                        
                        best_var = None
                        best_cost = float('inf')
                        for vi, sign in clause:
                            old_val = assignment[vi]
                            assignment[vi] = not old_val
                            cost = 0
                            for cj in sub_var_to_clauses[vi]:
                                sat = False
                                for vj, sj in sub_clauses[cj]:
                                    if assignment[vj] == (sj == 1):
                                        sat = True
                                        break
                                if not sat:
                                    cost += 1
                            assignment[vi] = old_val
                            if cost < best_cost:
                                best_cost = cost
                                best_var = vi
                        
                        if best_var is not None and best_cost < float('inf'):
                            assignment[best_var] = not assignment[best_var]
                        else:
                            break
                    
                    return assignment
                
                # Initial assignment: use current best for these vars
                sub_assignment = []
                for vi in unsolved_vars:
                    sub_assignment.append(final_assignment[vi])
                
                restart = 0
                while time.time() < deadline - 3.0:
                    restart += 1
                    
                    if restart == 1:
                        pass  # Use initial
                    else:
                        sub_assignment = [rng.random() < 0.5 for _ in range(sub_size)]
                    
                    p_flip = 0.5 + 0.2 * rng.random()
                    steps = 3000 + 3000 * rng.random()
                    
                    for _ in range(int(steps)):
                        if time.time() > deadline - 3.0:
                            break
                        sub_walksat_step(sub_assignment, rng, p_flip)
                    
                    sub_assignment = sub_local_search(sub_assignment, rng, max_iters=1500)
                    
                    # Check full instance satisfaction
                    # Apply sub_assignment to final_assignment
                    for i, vi in enumerate(unsolved_vars):
                        final_assignment[vi] = sub_assignment[i]
                    
                    # Check if full instance is satisfied
                    full_unsat = 0
                    for ci in range(m):
                        sat = False
                        for vi, sign in clause_vars[ci]:
                            if final_assignment[vi] == (sign == 1):
                                sat = True
                                break
                        if not sat:
                            full_unsat += 1
                    
                    if full_unsat == 0:
                        return final_assignment[:]
    
    # Final check
    full_unsat = 0
    for ci in range(m):
        sat = False
        for vi, sign in clause_vars[ci]:
            if final_assignment[vi] == (sign == 1):
                sat = True
                break
        if not sat:
            full_unsat += 1
    
    if full_unsat == 0:
        return final_assignment
    
    # If still not solved, do a final global WalkSAT attempt with remaining time
    if time.time() < deadline - 2.0:
        rng = random.Random(99999)
        
        def global_evaluate(assignment):
            unsat = 0
            for ci in range(m):
                sat = False
                for vi, sign in clause_vars[ci]:
                    if assignment[vi] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    unsat += 1
            return unsat
        
        def global_walksat_step(assignment, rng, p_flip=0.55):
            unsat_clauses = []
            for ci in range(m):
                sat = False
                for vi, sign in clause_vars[ci]:
                    if assignment[vi] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)
            
            if not unsat_clauses:
                return assignment
            
            ci = rng.choice(unsat_clauses)
            clause = clause_vars[ci]
            
            if rng.random() < p_flip:
                vi, sign = rng.choice(clause)
                assignment[vi] = not assignment[vi]
            else:
                best_var = None
                best_cost = float('inf')
                for vi, sign in clause:
                    old_val = assignment[vi]
                    assignment[vi] = not old_val
                    cost = 0
                    for cj in var_to_clauses[vi]:
                        sat = False
                        for vj, sj in clause_vars[cj]:
                            if assignment[vj] == (sj == 1):
                                sat = True
                                break
                        if not sat:
                            cost += 1
                    assignment[vi] = old_val
                    if cost < best_cost:
                        best_cost = cost
                        best_var = vi
                
                if best_var is not None:
                    assignment[best_var] = not assignment[best_var]
            
            return assignment
        
        def global_local_search(assignment, rng, max_iters=2000):
            for _ in range(max_iters):
                if time.time() > deadline - 2.0:
                    break
                
                unsat_clauses = []
                for ci in range(m):
                    sat = False
                    for vi, sign in clause_vars[ci]:
                        if assignment[vi] == (sign == 1):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(ci)
                
                if not unsat_clauses:
                    break
                
                ci = rng.choice(unsat_clauses)
                clause = clause_vars[ci]
                
                best_var = None
                best_cost = float('inf')
                for vi, sign in clause:
                    old_val = assignment[vi]
                    assignment[vi] = not old_val
                    cost = 0
                    for cj in var_to_clauses[vi]:
                        sat = False
                        for vj, sj in clause_vars[cj]:
                            if assignment[vj] == (sj == 1):
                                sat = True
                                break
                        if not sat:
                            cost += 1
                    assignment[vi] = old_val
                    if cost < best_cost:
                        best_cost = cost
                        best_var = vi
                
                if best_var is not None and best_cost < float('inf'):
                    assignment[best_var] = not assignment[best_var]
                else:
                    break
            
            return assignment
        
        assignment = final_assignment[:]
        restart = 0
        while time.time() < deadline - 2.0:
            restart += 1
            if restart > 1:
                assignment = [rng.random() < 0.5 for _ in range(n)]
            
            p_flip = 0.5 + 0.2 * rng.random()
            steps = 5000 + 5000 * rng.random()
            
            for _ in range(int(steps)):
                if time.time() > deadline - 2.0:
                    break
                global_walksat_step(assignment, rng, p_flip)
            
            assignment = global_local_search(assignment, rng, max_iters=2000)
            
            if global_evaluate(assignment) == 0:
                return assignment
    
    return final_assignment
# EVOLVE-BLOCK-END
