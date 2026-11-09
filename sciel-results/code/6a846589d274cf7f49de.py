"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    """Hybrid solver: DPLL with watch-based propagation, fallback to WalkSAT."""
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 300s wall-clock limit, use 295s for safety margin.
    deadline = time.time() + 295.0
    dpll_deadline = time.time() + 120.0  # Give DPLL up to 120s before switching to WalkSAT
    
    # Convert clauses to list of (var, polarity) for faster access
    # var is 0-based, polarity is 1 for positive, 0 for negative
    clauses_v = [[(abs(l)-1, 1 if l > 0 else 0) for l in c] for c in clauses]
    
    # --- Watch-based unit propagation ---
    def build_watches(clauses_v, assignment):
        """Build watch structure. Returns (watches, active_clauses)."""
        # watches[var] = list of (clause_idx, other_var, other_pol)
        watches = [[] for _ in range(n)]
        active = [True] * len(clauses_v)
        for ci, clause in enumerate(clauses_v):
            if len(clause) <= 1:
                continue
            # Watch the first two unassigned literals
            unassigned = [(v, p) for v, p in clause if v not in assignment]
            if len(unassigned) >= 2:
                v1, p1 = unassigned[0]
                v2, p2 = unassigned[1]
                watches[v1].append((ci, v2, p2))
                watches[v2].append((ci, v1, p1))
            elif len(unassigned) == 1:
                # This is already a unit clause
                pass
            else:
                active[ci] = False  # clause is satisfied or empty
        return watches, active
    
    def unit_propagate_watch(clauses_v, assignment, watches, active, prop_deadline):
        """
        Watch-based unit propagation.
        Returns (new_assignment, new_watches, new_active, contradiction).
        """
        assignment = assignment.copy()
        queue = []  # variables to process
        
        # Initially, we need to find which variables are assigned
        # We'll process all assigned variables
        for var in assignment:
            queue.append(var)
        
        while queue:
            if time.time() > prop_deadline:
                return assignment, watches, active, None  # timeout, return partial
            
            var = queue.pop()
            val = assignment[var]
            
            for ci, other_var, other_pol in watches[var]:
                if not active[ci]:
                    continue
                
                clause = clauses_v[ci]
                
                # Check if clause is satisfied by current assignment
                satisfied = False
                for v, p in clause:
                    if v in assignment and assignment[v] == (p == 1):
                        satisfied = True
                        break
                
                if satisfied:
                    active[ci] = False
                    continue
                
                # Check if other_var is assigned
                if other_var in assignment:
                    if assignment[other_var] == (other_pol == 1):
                        # Satisfied
                        active[ci] = False
                        continue
                    else:
                        # Both watches are falsified - need to find a new watch
                        # This means the clause has become unit or empty
                        remaining = [(v, p) for v, p in clause if v not in assignment]
                        if len(remaining) == 0:
                            return assignment, watches, active, True  # contradiction
                        elif len(remaining) == 1:
                            new_var, new_pol = remaining[0]
                            if new_var not in assignment:
                                assignment[new_var] = (new_pol == 1)
                                queue.append(new_var)
                        # Remove this watch entry (it's stale)
                        # We'll rebuild watches at the end
                        continue
                else:
                    # other_var is unassigned, so this watch is valid
                    # But we need to remove the watch for 'var' since var is now assigned
                    # Actually, in watch scheme, we keep watches on unassigned vars
                    pass
            
            # After processing, clean up watches for var
            # Remove all watches involving var
            watches[var] = []
        
        # Rebuild watches for remaining active clauses
        new_watches = [[] for _ in range(n)]
        new_active = [False] * len(clauses_v)
        for ci, clause in enumerate(clauses_v):
            unassigned = [(v, p) for v, p in clause if v not in assignment]
            if len(unassigned) == 0:
                # Check if satisfied
                sat = False
                for v, p in clause:
                    if v in assignment and assignment[v] == (p == 1):
                        sat = True
                        break
                if not sat:
                    return assignment, new_watches, new_active, True  # contradiction
            elif len(unassigned) == 1:
                v, p = unassigned[0]
                assignment[v] = (p == 1)
                queue.append(v)
            else:
                new_active[ci] = True
                v1, p1 = unassigned[0]
                v2, p2 = unassigned[1]
                new_watches[v1].append((ci, v2, p2))
                new_watches[v2].append((ci, v1, p1))
        
        # Process any new unit clauses
        while queue:
            if time.time() > prop_deadline:
                break
            var = queue.pop()
            val = assignment[var]
            # Rebuild is simplified; for correctness, do a full pass
            # This is getting complex; let's use a simpler approach
        
        return assignment, new_watches, new_active, False
    
    # --- Simpler, more robust DPLL with efficient propagation ---
    def dpll_solve(clauses, assignment, deadline):
        """DPLL with efficient unit propagation."""
        if time.time() > deadline:
            return None
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                remaining = []
                for var, pol in clause:
                    if var not in assignment:
                        remaining.append((var, pol))
                    else:
                        if assignment[var] == (pol == 1):
                            remaining = None
                            break
                if remaining is None:
                    continue
                elif len(remaining) == 0:
                    return None
                elif len(remaining) == 1:
                    var, pol = remaining[0]
                    if var not in assignment:
                        assignment[var] = (pol == 1)
                        changed = True
                    new_clauses.append(remaining)
                else:
                    new_clauses.append(remaining)
            clauses = new_clauses
        
        if len(clauses) == 0:
            return assignment
        
        # Pure literal elimination
        literal_counts = {}
        for clause in clauses:
            for var, pol in clause:
                if var not in assignment:
                    key = (var, pol)
                    literal_counts[key] = literal_counts.get(key, 0) + 1
        
        pure_vars = set()
        for (var, pol) in literal_counts:
            if (var, 1 - pol) not in literal_counts:
                pure_vars.add(var)
        
        if pure_vars:
            for var in pure_vars:
                for (v, p) in literal_counts:
                    if v == var and var not in assignment:
                        assignment[var] = (p == 1)
                        break
        
        # Re-propagate after pure literal
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                remaining = []
                for var, pol in clause:
                    if var not in assignment:
                        remaining.append((var, pol))
                    else:
                        if assignment[var] == (pol == 1):
                            remaining = None
                            break
                if remaining is None:
                    continue
                elif len(remaining) == 0:
                    return None
                elif len(remaining) == 1:
                    var, pol = remaining[0]
                    if var not in assignment:
                        assignment[var] = (pol == 1)
                        changed = True
                    new_clauses.append(remaining)
                else:
                    new_clauses.append(remaining)
            clauses = new_clauses
        
        if len(clauses) == 0:
            return assignment
        
        if time.time() > deadline:
            return None
        
        # Variable selection: most occurrences
        var_counts = {}
        for clause in clauses:
            for var, pol in clause:
                var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            return assignment
        
        var = max(var_counts, key=lambda v: var_counts[v])
        
        # Branch
        for val in [True, False]:
            new_assignment = dict(assignment)
            new_assignment[var] = val
            new_clauses = []
            for clause in clauses:
                remaining = []
                for v, pol in clause:
                    if v == var:
                        if (pol == 1) == val:
                            remaining = None
                            break
                    else:
                        remaining.append((v, pol))
                if remaining is None:
                    continue
                elif len(remaining) == 0:
                    break
                else:
                    new_clauses.append(remaining)
            
            if len(remaining) == 0 if remaining is not None else False:
                continue
            
            result = dpll_solve(new_clauses, new_assignment, deadline)
            if result is not None:
                return result
        
        return None
    
    # --- WalkSAT fallback ---
    def walksat(clauses_v, assignment, deadline):
        """WalkSAT with adaptive noise."""
        if time.time() > deadline:
            return None
        
        n_clauses = len(clauses_v)
        if n_clauses == 0:
            return assignment
        
        # Count unsatisfied clauses
        def count_unsat(assignment):
            count = 0
            for clause in clauses_v:
                sat = False
                for var, pol in clause:
                    if var in assignment and assignment[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    count += 1
            return count
        
        # Initial random assignment
        assignment = {i: random.random() < 0.5 for i in range(n)}
        
        # Precompute clause info
        clause_vars = [set(v for v, p in clause) for clause in clauses_v]
        
        best_assignment = dict(assignment)
        best_unsat = count_unsat(assignment)
        
        max_steps = 500000
        step = 0
        flip_rate = 0.25
        
        while step < max_steps and time.time() < deadline:
            unsat_count = count_unsat(assignment)
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = dict(assignment)
            
            # Pick a random unsatisfied clause
            while True:
                ci = random.randint(0, n_clauses - 1)
                clause = clauses_v[ci]
                sat = False
                for var, pol in clause:
                    if var in assignment and assignment[var] == (pol == 1):
                        sat = True
                        break
                if not sat:
                    break
            
            # Pick a variable from this clause
            var, pol = random.choice(clause)
            
            if random.random() < flip_rate:
                # Flip to satisfy this clause
                assignment[var] = (pol == 1)
            else:
                # Flip based on which assignment minimizes total unsat
                # Try both values
                current_val = assignment[var]
                assignment[var] = not current_val
                unsat_after = count_unsat(assignment)
                if unsat_after > unsat_count:
                    assignment[var] = current_val
                # Keep the better one
            
            step += 1
            if step % 10000 == 0:
                flip_rate = max(0.1, min(0.5, flip_rate * 0.95))
        
        if best_unsat == 0:
            return best_assignment
        return None
    
    # Start DPLL
    initial_assignment = {}
    result = dpll_solve(clauses_v, initial_assignment, dpll_deadline)
    
    if result is not None:
        final_assignment = [result.get(i, False) for i in range(n)]
        return final_assignment
    
    # Fallback to WalkSAT
    if time.time() < deadline - 5.0:
        ws_result = walksat(clauses_v, {}, deadline)
        if ws_result is not None:
            final_assignment = [ws_result.get(i, False) for i in range(n)]
            return final_assignment
    
    # Final fallback: return all False
    return [False] * n
# EVOLVE-BLOCK-END
