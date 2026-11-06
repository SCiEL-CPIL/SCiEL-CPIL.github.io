import time
import random
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285  # Safety margin

    # Convert to 0-based internal representation
    # clauses_internal: list of tuples (var, is_pos)
    clauses_internal = []
    for c in clauses:
        clauses_internal.append(tuple((abs(l) - 1, l > 0) for l in c))
    
    m = len(clauses_internal)
    
    # If no clauses, any assignment works
    if m == 0:
        return [False] * n

    # Helper: Check if assignment satisfies all clauses
    def is_satisfied(assign):
        for c in clauses_internal:
            sat = False
            for var, is_pos in c:
                if assign[var] == is_pos:
                    sat = True
                    break
            if not sat:
                return False
        return True

    # Helper: Unit Propagation and Pure Literal Elimination
    # Returns (new_assignment, remaining_clauses, unit_vars, pure_vars, success)
    # new_assignment: dict var -> bool
    # remaining_clauses: list of clauses with unit/pure literals removed
    # success: False if contradiction (empty clause)
    def preprocess(assign_dict, active_clauses):
        # active_clauses: list of clause indices or list of clause tuples? 
        # Let's work with lists of clause tuples for simplicity in DPLL recursion, 
        # but for performance, indices are better. 
        # For DPLL, we pass a list of clause tuples (filtered).
        
        unit_vars = {}
        pure_vars = {}
        changed = True
        current_clauses = active_clauses
        
        while changed:
            changed = False
            
            # 1. Unit Propagation
            # Find clauses with only one unassigned literal
            new_clauses = []
            for c in current_clauses:
                unassigned_lits = []
                for var, is_pos in c:
                    if var in assign_dict:
                        if assign_dict[var] == is_pos:
                            break # Clause satisfied
                        else:
                            continue # Literal false, ignore
                    else:
                        unassigned_lits.append((var, is_pos))
                
                if len(unassigned_lits) == 0:
                    return None, None, None, None, False # Contradiction: empty clause
                elif len(unassigned_lits) == 1:
                    var, is_pos = unassigned_lits[0]
                    if var in assign_dict:
                        # Already assigned, check consistency
                        if assign_dict[var] != is_pos:
                            return None, None, None, None, False
                    else:
                        assign_dict[var] = is_pos
                        unit_vars[var] = is_pos
                        changed = True
                    # Clause is satisfied by this assignment, so we don't need to keep it
                    # unless we want to track it, but logically it's done.
                else:
                    new_clauses.append(tuple(unassigned_lits))
            
            current_clauses = new_clauses
            
            # 2. Pure Literal Elimination
            if not current_clauses:
                break
                
            lit_counts = {}
            for c in current_clauses:
                for var, is_pos in c:
                    if var not in assign_dict:
                        key = (var, is_pos)
                        if key not in lit_counts:
                            lit_counts[key] = 0
                        lit_counts[key] += 1
            
            for var in assign_dict.keys():
                pass # Just checking keys, not modifying during iteration
            
            # Identify pure literals
            to_remove = set()
            for (var, is_pos), count in lit_counts.items():
                if var in assign_dict:
                    continue
                opposite = (var, not is_pos)
                if opposite not in lit_counts:
                    # Pure literal
                    assign_dict[var] = is_pos
                    pure_vars[var] = is_pos
                    to_remove.add(var)
                    changed = True
            
            if to_remove:
                # Filter clauses again
                new_clauses = []
                for c in current_clauses:
                    unassigned_lits = []
                    for var, is_pos in c:
                        if var in assign_dict:
                            if assign_dict[var] == is_pos:
                                break
                            else:
                                continue
                        else:
                            unassigned_lits.append((var, is_pos))
                    if len(unassigned_lits) > 0:
                        new_clauses.append(tuple(unassigned_lits))
                current_clauses = new_clauses
        
        return assign_dict, current_clauses, unit_vars, pure_vars, True

    # DPLL Solver
    def dpll(clause_list, assign_dict):
        # clause_list: list of tuples of (var, is_pos)
        # assign_dict: dict var -> bool
        
        # Preprocess
        res = preprocess(assign_dict, clause_list)
        if res is None:
            return None
        assign_dict, remaining_clauses, _, _, success = res
        if not success:
            return None
        
        if not remaining_clauses:
            return assign_dict
        
        # Choose a variable
        # Heuristic: Variable appearing in most clauses
        var_counts = {}
        for c in remaining_clauses:
            for var, _ in c:
                var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            return assign_dict
            
        # Pick variable with highest count
        var_to_split = max(var_counts, key=var_counts.get)
        
        # Try True
        assign_dict[var_to_split] = True
        new_clauses = []
        for c in remaining_clauses:
            new_c = []
            sat = False
            for var, is_pos in c:
                if var == var_to_split:
                    if is_pos:
                        sat = True
                        break
                    else:
                        continue
                else:
                    new_c.append((var, is_pos))
            if not sat:
                if len(new_c) > 0:
                    new_clauses.append(tuple(new_c))
        
        result = dpll(new_clauses, assign_dict)
        if result is not None:
            return result
            
        # Try False
        assign_dict[var_to_split] = False
        new_clauses = []
        for c in remaining_clauses:
            new_c = []
            sat = False
            for var, is_pos in c:
                if var == var_to_split:
                    if not is_pos:
                        sat = True
                        break
                    else:
                        continue
                else:
                    new_c.append((var, is_pos))
            if not sat:
                if len(new_c) > 0:
                    new_clauses.append(tuple(new_c))
        
        result = dpll(new_clauses, assign_dict)
        if result is not None:
            return result
            
        # Backtrack
        del assign_dict[var_to_split]
        return None

    # WalkSAT Solver with Phase Saving
    rng = random.Random(42)
    
    def walksat(initial_assign, max_steps=5000):
        assign = initial_assign.copy()
        # Calculate initial unsat count
        unsat_clauses = []
        for i, c in enumerate(clauses_internal):
            sat = False
            for var, is_pos in c:
                if assign[var] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(i)
        
        if not unsat_clauses:
            return assign
        
        best_assign = assign.copy()
        best_unsat = len(unsat_clauses)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assign
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            c = clauses_internal[ci]
            
            # Pick a variable in the clause to flip
            # Weighted by frequency in unsat clauses? 
            # Simple: random choice from clause literals
            var, is_pos = rng.choice(c)
            
            # Flip variable
            assign[var] = not assign[var]
            
            # Update unsat clauses
            # This is inefficient if done naively, but for n=150, m~450, it's okay.
            # Optimization: Only check clauses containing 'var'
            # We need an index for this.
            
            new_unsat = []
            for i, cl in enumerate(clauses_internal):
                sat = False
                for v, p in cl:
                    if assign[v] == p:
                        sat = True
                        break
                if not sat:
                    new_unsat.append(i)
            
            unsat_clauses = new_unsat
            
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assign = assign.copy()
                
        return best_assign

    # Build index for faster WalkSAT updates
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses_internal):
        for var, _ in c:
            var_to_clauses[var].append(i)

    def walksat_optimized(initial_assign, max_steps=5000):
        assign = initial_assign.copy()
        
        # Precompute unsat status for each clause
        is_unsat = [False] * m
        unsat_count = 0
        for i, c in enumerate(clauses_internal):
            sat = False
            for var, is_pos in c:
                if assign[var] == is_pos:
                    sat = True
                    break
            if not sat:
                is_unsat[i] = True
                unsat_count += 1
        
        if unsat_count == 0:
            return assign
        
        best_assign = assign.copy()
        best_unsat = unsat_count
        unsat_indices = [i for i in range(m) if is_unsat[i]]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_indices:
                return assign
            
            # Pick random unsat clause
            ci = rng.choice(unsat_indices)
            c = clauses_internal[ci]
            
            # Pick variable to flip
            # Weighted by how many unsat clauses it appears in?
            # For now, random from clause
            var, is_pos = rng.choice(c)
            
            # Flip
            assign[var] = not assign[var]
            
            # Update unsat status for clauses containing var
            affected = var_to_clauses[var]
            for i in affected:
                # Check if clause i is now satisfied
                sat = False
                for v, p in clauses_internal[i]:
                    if assign[v] == p:
                        sat = True
                        break
                
                was_unsat = is_unsat[i]
                is_unsat[i] = not sat
                
                if was_unsat and not is_unsat[i]:
                    unsat_count -= 1
                elif not was_unsat and is_unsat[i]:
                    unsat_count += 1
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign.copy()
            
            # Rebuild unsat_indices list? 
            # Doing this every step is O(m). 
            # Better: maintain a set or list and update it.
            # For simplicity and given n=150, rebuilding is O(m) which is small.
            # But let's optimize slightly: only rebuild if we need to pick from it.
            # We pick from unsat_indices. If we just updated, some might be stale.
            # Let's just rebuild every step. m is ~450. 450 * 5000 steps = 2.25M ops. Fast enough.
            unsat_indices = [i for i in range(m) if is_unsat[i]]
            
            if unsat_count == 0:
                return assign
                
        return best_assign

    # Main Hybrid Loop
    # Start with random assignment
    current_assign = [rng.random() < 0.5 for _ in range(n)]
    best_assign = current_assign.copy()
    
    # Initial Preprocessing
    assign_dict = {}
    res = preprocess(assign_dict, clauses_internal)
    if res is None:
        # Contradiction? Should not happen for planted SAT, but just in case
        pass
    else:
        assign_dict, remaining_clauses, _, _, success = res
        if success:
            # Apply assignments
            for var, val in assign_dict.items():
                current_assign[var] = val
            # Check if solved
            if is_satisfied(current_assign):
                return current_assign
            
            # Use remaining clauses for DPLL if needed
            # But DPLL needs the full structure or reduced.
            # Let's just run DPLL on the reduced problem if it's small enough?
            # Or run DPLL on the original problem with initial assignments?
            # DPLL with initial assignments is tricky. 
            # Let's stick to running DPLL on the original problem, 
            # but it will quickly unit propagate the same things.
            
    # Phase Saving: Keep track of best assignment to guide restarts
    phase = best_assign.copy()
    
    max_restarts = 20
    walk_steps = 2000
    dpll_depth_limit = 50 # Not strictly depth, but we limit time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # 1. Try WalkSAT
        # Initialize with phase (best known solution) or random
        if restart == 0:
            init = current_assign.copy()
        else:
            # Mix phase and random
            init = []
            for i in range(n):
                if rng.random() < 0.8:
                    init.append(phase[i])
                else:
                    init.append(rng.random() < 0.5)
        
        ws_result = walksat_optimized(init, walk_steps)
        
        if is_satisfied(ws_result):
            return ws_result
        
        # Update best if better
        # Calculate unsat count for ws_result
        unsat_ws = 0
        for c in clauses_internal:
            sat = False
            for var, is_pos in c:
                if ws_result[var] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat_ws += 1
        
        unsat_best = 0
        for c in clauses_internal:
            sat = False
            for var, is_pos in c:
                if best_assign[var] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat_best += 1
                
        if unsat_ws < unsat_best:
            best_assign = ws_result
            phase = ws_result
        
        # 2. Try DPLL
        # DPLL is expensive, so we run it with a time limit or simple heuristic.
        # We'll run it on the original problem.
        # To speed it up, we can use the current best assignment as a heuristic for variable ordering?
        # Standard DPLL:
        
        # Check time before starting DPLL
        if time.time() > deadline - 10:
            continue
            
        dpll_start = time.time()
        # Run DPLL with a simple timeout check inside recursion? 
        # Python recursion limit might be an issue for n=150.
        # sys.setrecursionlimit(10000)
        # Let's implement iterative DPLL or limit depth.
        # For now, let's try a limited DPLL search.
        
        # We'll use a simple recursive DPLL but with a node limit.
        node_limit = 10000
        nodes_visited = 0
        
        def dpll_limited(clause_list, assign_dict):
            nonlocal nodes_visited
            if nodes_visited > node_limit:
                return None
            nodes_visited += 1
            
            if time.time() > deadline - 5:
                return None
                
            # Preprocess
            res = preprocess(assign_dict, clause_list)
            if res is None:
                return None
            assign_dict, remaining_clauses, _, _, success = res
            if not success:
                return None
            
            if not remaining_clauses:
                return assign_dict
            
            # Choose variable
            var_counts = {}
            for c in remaining_clauses:
                for var, _ in c:
                    var_counts[var] = var_counts.get(var, 0) + 1
            
            if not var_counts:
                return assign_dict
                
            var_to_split = max(var_counts, key=var_counts.get)
            
            # Try True
            assign_dict[var_to_split] = True
            new_clauses = []
            for c in remaining_clauses:
                new_c = []
                sat = False
                for var, is_pos in c:
                    if var == var_to_split:
                        if is_pos:
                            sat = True
                            break
                        else:
                            continue
                    else:
                        new_c.append((var, is_pos))
                if not sat:
                    if len(new_c) > 0:
                        new_clauses.append(tuple(new_c))
            
            result = dpll_limited(new_clauses, assign_dict)
            if result is not None:
                return result
                
            # Try False
            assign_dict[var_to_split] = False
            new_clauses = []
            for c in remaining_clauses:
                new_c = []
                sat = False
                for var, is_pos in c:
                    if var == var_to_split:
                        if not is_pos:
                            sat = True
                            break
                        else:
                            continue
                    else:
                        new_c.append((var, is_pos))
                if not sat:
                    if len(new_c) > 0:
                        new_clauses.append(tuple(new_c))
            
            result = dpll_limited(new_clauses, assign_dict)
            if result is not None:
                return result
                
            del assign_dict[var_to_split]
            return None

        # Run DPLL
        dpll_result = dpll_limited(clauses_internal, {})
        
        if dpll_result is not None:
            # Convert dict to list
            final_assign = [False] * n
            for var in range(n):
                if var in dpll_result:
                    final_assign[var] = dpll_result[var]
                else:
                    # Variable not assigned, pick random or from phase
                    final_assign[var] = phase[var]
            
            if is_satisfied(final_assign):
                return final_assign
        
        # If DPLL didn't find a solution, continue loop
        # Adjust walk_steps slightly?
        walk_steps = int(walk_steps * 1.1)
        
    # Return best found
    return best_assign

# Set recursion limit just in case DPLL goes deep
sys.setrecursionlimit(10000)
