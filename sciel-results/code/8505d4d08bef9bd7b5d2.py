"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict, deque


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 285.0
    
    # Convert to 0-based with signs
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])
    
    # Build variable-clause incidence
    var_to_clauses = defaultdict(list)
    for ci, cv in enumerate(clause_vars):
        for var, sign in cv:
            var_to_clauses[var].append(ci)
    
    # Find connected components via BFS
    visited_vars = [False] * n
    visited_clauses = [False] * m
    components = []
    
    for v in range(n):
        if visited_vars[v]:
            continue
        comp_vars = set()
        comp_clauses = set()
        queue = deque([(v, 'var')])
        visited_vars[v] = True
        
        while queue:
            node, node_type = queue.popleft()
            if node_type == 'var':
                comp_vars.add(node)
                for ci in var_to_clauses[node]:
                    if not visited_clauses[ci]:
                        visited_clauses[ci] = True
                        queue.append((ci, 'clause'))
            else:
                comp_clauses.add(node)
                for var, sign in clause_vars[node]:
                    if not visited_vars[var]:
                        visited_vars[var] = True
                        queue.append((var, 'var'))
        
        components.append((list(comp_vars), list(comp_clauses)))
    
    # Sort by size descending
    components.sort(key=lambda x: -len(x[0]))
    
    assignment = [False] * n
    rng = random.Random(42)
    
    for comp_idx, (comp_vars, comp_clauses) in enumerate(components):
        if time.time() > deadline - 3.0:
            break
        
        num_vars = len(comp_vars)
        num_clauses = len(comp_clauses)
        
        # Time budget: proportional to size
        time_budget = min(80.0, max(1.0, 150.0 * num_vars / n))
        comp_start = time.time()
        
        var_map = {v: i for i, v in enumerate(comp_vars)}
        local_n = len(comp_vars)
        
        local_clauses = []
        for ci in comp_clauses:
            local_c = []
            for var, sign in clause_vars[ci]:
                local_c.append((var_map[var], sign))
            local_clauses.append(local_c)
        
        local_var_to_clauses = [[] for _ in range(local_n)]
        for ci, lc in enumerate(local_clauses):
            for var, sign in lc:
                local_var_to_clauses[var].append(ci)
        
        # Strategy 1: DPLL with unit propagation and pure literal elimination
        found = False
        dpll_assignment = [None] * local_n  # None = unassigned
        
        def dpll_solve(clauses_list, assigned, depth=0):
            """DPLL with unit propagation and pure literal elimination."""
            if time.time() > comp_start + time_budget:
                return False, None
            
            # Check if all clauses satisfied
            remaining = []
            for c in clauses_list:
                sat = False
                for var, sign in c:
                    if assigned[var] is not None:
                        if assigned[var] == sign:
                            sat = True
                            break
                if not sat:
                    remaining.append(c)
            
            if not remaining:
                return True, assigned
            
            # Unit propagation
            changed = True
            while changed:
                changed = False
                new_remaining = []
                for c in remaining:
                    unassigned = [(v, s) for v, s in c if assigned[v] is None]
                    if len(unassigned) == 0:
                        return False, None  # Conflict
                    elif len(unassigned) == 1:
                        v, s = unassigned[0]
                        if assigned[v] is None:
                            assigned[v] = s
                            changed = True
                        # This clause is satisfied, skip
                        continue
                    else:
                        new_remaining.append(c)
                remaining = new_remaining
            
            if not remaining:
                return True, assigned
            
            # Pure literal elimination
            pure_found = True
            while pure_found:
                pure_found = False
                for var in range(local_n):
                    if assigned[var] is not None:
                        continue
                    signs_in_clauses = set()
                    for ci in local_var_to_clauses[var]:
                        for v, s in remaining:
                            if v == var:
                                signs_in_clauses.add(s)
                    if len(signs_in_clauses) == 1:
                        assigned[var] = list(signs_in_clauses)[0]
                        pure_found = True
                        break
            
            if pure_found:
                return dpll_solve(remaining, assigned, depth + 1)
            
            if depth > 20:  # Limit branching depth
                return False, None
            
            # Choose variable with highest occurrence in remaining clauses
            var_counts = defaultdict(int)
            for c in remaining:
                for var, sign in c:
                    if assigned[var] is None:
                        var_counts[var] += 1
            
            if not var_counts:
                return True, assigned
            
            # Pick variable with max count
            best_var = max(var_counts, key=lambda v: var_counts[v])
            
            # Try True first, then False
            for val in [True, False]:
                trial_assigned = assigned[:]
                trial_assigned[best_var] = val
                success, result = dpll_solve(remaining, trial_assigned, depth + 1)
                if success:
                    # Copy back
                    for i in range(local_n):
                        if result[i] is not None:
                            assigned[i] = result[i]
                    return True, assigned
            
            return False, None
        
        # Try DPLL
        dpll_assigned = [None] * local_n
        success, dpll_result = dpll_solve(local_clauses, dpll_assigned)
        
        if success and dpll_result:
            # Fill any unassigned with random
            for i in range(local_n):
                if dpll_result[i] is None:
                    dpll_result[i] = rng.random() < 0.5
            found = True
            comp_assignment = dpll_result
        else:
            # Strategy 2: WalkSAT with gain heuristic
            comp_assignment = [rng.random() < 0.5 for _ in range(local_n)]
            
            def local_satisfied(clause):
                return any(comp_assignment[var] == sign for var, sign in clause)
            
            def count_unsat():
                return sum(1 for c in local_clauses if not local_satisfied(c))
            
            max_iters = min(80000, max(2000, 80 * num_clauses))
            
            for it in range(max_iters):
                if time.time() > comp_start + time_budget:
                    break
                
                unsat_count = count_unsat()
                if unsat_count == 0:
                    found = True
                    break
                
                unsat_clauses = [c for c in local_clauses if not local_satisfied(c)]
                if not unsat_clauses:
                    found = True
                    break
                
                clause = rng.choice(unsat_clauses)
                
                # Gain-based flip selection
                best_var = None
                best_gain = -1
                for var, sign in clause:
                    gain = 0
                    for ci in local_var_to_clauses[var]:
                        c = local_clauses[ci]
                        is_sat = local_satisfied(c)
                        new_val = not comp_assignment[var]
                        clause_sat_flip = any(
                            (comp_assignment[v] if v != var else new_val) == s
                            for v, s in c
                        )
                        if is_sat and not clause_sat_flip:
                            gain -= 1
                        elif not is_sat and clause_sat_flip:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                
                # Noise: randomly pick from clause if no good gain
                if best_gain <= 0 and rng.random() < 0.5:
                    best_var = rng.choice(clause)[0]
                
                if best_var is None:
                    best_var = rng.choice(clause)[0]
                
                comp_assignment[best_var] = not comp_assignment[best_var]
            
            # Random restarts if not found
            if not found:
                for restart in range(5):
                    if time.time() > comp_start + time_budget:
                        break
                    comp_assignment = [rng.random() < 0.5 for _ in range(local_n)]
                    for it in range(max_iters // 3):
                        if time.time() > comp_start + time_budget:
                            break
                        unsat_count = count_unsat()
                        if unsat_count == 0:
                            found = True
                            break
                        unsat_clauses = [c for c in local_clauses if not local_satisfied(c)]
                        if not unsat_clauses:
                            found = True
                            break
                        clause = rng.choice(unsat_clauses)
                        var, sign = rng.choice(clause)
                        comp_assignment[var] = not comp_assignment[var]
                    if found:
                        break
        
        for i, v in enumerate(comp_vars):
            assignment[v] = comp_assignment[i]
    
    return assignment
# EVOLVE-BLOCK-END
