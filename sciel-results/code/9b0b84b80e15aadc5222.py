import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 285.0
    
    # --- Preprocessing: Pure Literal Elimination ---
    # If a variable appears only with one sign, we can assign it to satisfy those clauses.
    # This reduces the problem size.
    
    # We need to track which variables are "pure".
    # Let's do a simple pass.
    pos_counts = [0] * n
    neg_counts = [0] * n
    
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
                
    assignment = [None] * n # None means unassigned
    
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if assignment[v] is not None:
                continue
            if pos_counts[v] > 0 and neg_counts[v] == 0:
                assignment[v] = True
                changed = True
            elif neg_counts[v] > 0 and pos_counts[v] == 0:
                assignment[v] = False
                changed = True
                
    # Apply assignments to simplify clauses
    # Remove satisfied clauses, remove true literals, remove false literals (leaving unit/empty)
    new_clauses = []
    for c in clauses:
        new_c = []
        satisfied = False
        for lit in c:
            v = abs(lit) - 1
            if assignment[v] is not None:
                if assignment[v] == (lit > 0):
                    satisfied = True
                    break
                else:
                    continue # Literal is false, remove it
            else:
                new_c.append(lit)
                
        if not satisfied:
            if len(new_c) == 0:
                # Unsatisfiable? Task says satisfiable. 
                # If we hit this, fallback to random.
                return [random.random() < 0.5 for _ in range(n)]
            new_clauses.append(new_c)
            
    clauses = new_clauses
    
    # If no clauses left, we have a solution for the simplified instance.
    # We need to map back to the original n variables.
    if not clauses:
        final_assignment = [False] * n
        for i in range(n):
            if assignment[i] is not None:
                final_assignment[i] = assignment[i]
            else:
                final_assignment[i] = random.random() < 0.5
        return final_assignment

    # --- WalkSAT on the simplified instance ---
    
    # Re-index variables for the simplified instance? 
    # It's easier to keep the original indices but only flip unassigned ones?
    # Or just map simplified vars to a 0..m-1 range.
    
    # Let's map to a compact range for speed.
    var_map = {}
    new_n = 0
    for v in range(n):
        if assignment[v] is None:
            var_map[v] = new_n
            new_n += 1
            
    # Remap clauses
    remapped_clauses = []
    for c in clauses:
        new_c = []
        for lit in c:
            v = abs(lit) - 1
            new_v = var_map[v]
            new_lit = new_v + 1
            if lit < 0:
                new_lit = -new_lit
            new_c.append(new_lit)
        remapped_clauses.append(new_c)
        
    # Now we have a smaller instance: new_n variables, remapped_clauses
    # Initialize assignment for the new instance
    rng = random.Random(42)
    curr_assignment = [rng.random() < 0.5 for _ in range(new_n)]
    
    # Build index
    var_to_clauses = [[] for _ in range(new_n)]
    for i, c in enumerate(remapped_clauses):
        for lit in c:
            var_to_clauses[abs(lit) - 1].append(i)
            
    def is_satisfied(assignment, cl):
        for lit in cl:
            v = abs(lit) - 1
            if assignment[v] == (lit > 0):
                return True
        return False
        
    def count_unsat(assignment):
        return sum(1 for c in remapped_clauses if not is_satisfied(assignment, c))
        
    best_assignment = curr_assignment[:]
    best_unsat = count_unsat(curr_assignment)
    
    if best_unsat == 0:
        # Map back
        final_assignment = [False] * n
        for i in range(n):
            if assignment[i] is not None:
                final_assignment[i] = assignment[i]
            else:
                # Find original var index
                # We need reverse map
                pass
        # Let's just map back properly
        rev_map = {v: k for k, v in var_map.items()}
        for new_v, val in enumerate(best_assignment):
            orig_v = rev_map[new_v]
            final_assignment[orig_v] = val
        return final_assignment

    walk_prob = 0.3
    min_walk_prob = 0.05
    max_walk_prob = 0.8
    
    steps_since_improvement = 0
    
    while time.time() < deadline:
        unsat_clauses = [c for c in remapped_clauses if not is_satisfied(curr_assignment, c)]
        curr_unsat = len(unsat_clauses)
        
        if curr_unsat == 0:
            best_assignment = curr_assignment[:]
            break
            
        if curr_unsat < best_unsat:
            best_unsat = curr_unsat
            best_assignment = curr_assignment[:]
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            
        # Adaptive walk prob
        if steps_since_improvement > 200:
            walk_prob = min(max_walk_prob, walk_prob * 1.05)
        elif steps_since_improvement > 1000:
            # Restart
            for i in range(new_n):
                if rng.random() < 0.5:
                    curr_assignment[i] = rng.random() < 0.5
            walk_prob = 0.3
            steps_since_improvement = 0

        if not unsat_clauses:
            break

        clause = rng.choice(unsat_clauses)
        
        if rng.random() < walk_prob:
            lit = rng.choice(clause)
        else:
            min_gain = float('inf')
            best_lit = None
            for lit in clause:
                v = abs(lit) - 1
                gain = 0
                for c_idx in var_to_clauses[v]:
                    c = remapped_clauses[c_idx]
                    if c == clause:
                        continue
                    # Check if c is satisfied
                    is_sat = False
                    only_this = True
                    for l in c:
                        vv = abs(l) - 1
                        if curr_assignment[vv] == (l > 0):
                            is_sat = True
                            if vv != v:
                                only_this = False
                                break
                    if is_sat and only_this:
                        gain += 1
                if gain < min_gain:
                    min_gain = gain
                    best_lit = lit
            lit = best_lit
            
        v = abs(lit) - 1
        curr_assignment[v] = not curr_assignment[v]
        
    # Map best_assignment back to original
    rev_map = {v: k for k, v in var_map.items()}
    final_assignment = [False] * n
    for i in range(n):
        if assignment[i] is not None:
            final_assignment[i] = assignment[i]
        else:
            # Find new index
            # This is inefficient. Let's pre-compute reverse mapping list
            pass
            
    # Better reverse mapping
    orig_to_new = [-1] * n
    for orig_v, new_v in var_map.items():
        orig_to_new[orig_v] = new_v
        
    for i in range(n):
        if assignment[i] is not None:
            final_assignment[i] = assignment[i]
        else:
            new_v = orig_to_new[i]
            final_assignment[i] = best_assignment[new_v]
            
    return final_assignment
