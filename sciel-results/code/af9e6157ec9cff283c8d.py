def solve(instance):
    import time
    import random
    import sys
    from collections import defaultdict

    n = instance.get("n", 0)
    raw_clauses = instance.get("clauses", [])

    # Robust parsing
    clauses = []
    if isinstance(raw_clauses, (list, tuple)):
        if len(raw_clauses) > 0:
            if isinstance(raw_clauses[0], (list, tuple)):
                clauses = [list(c) for c in raw_clauses]
            else:
                clauses = [list(raw_clauses[i:i+3]) for i in range(0, len(raw_clauses), 3)]
        else:
            clauses = []
    elif isinstance(raw_clauses, int):
        clauses = []
    else:
        clauses = []

    if n == 0 or not clauses:
        return [False] * n

    deadline = time.time() + 290
    m = len(clauses)

    # Convert to 0-based signed literals
    # Each clause is a list of signed literals (1-based, sign indicates polarity)
    # We'll work with (var, is_positive) tuples internally
    clauses_data = []
    for c in clauses:
        cd = []
        for lit in c:
            var = abs(lit) - 1
            pos = lit > 0
            if 0 <= var < n:
                cd.append((var, pos))
        if cd:
            clauses_data.append(cd)
    m = len(clauses_data)

    if m == 0:
        return [False] * n

    # Build variable to clause index map
    var_to_clauses = defaultdict(list)
    for ci, c in enumerate(clauses_data):
        for var, pos in c:
            var_to_clauses[var].append(ci)

    def evaluate_full(assignment):
        """Check if assignment satisfies all clauses."""
        for c in clauses_data:
            sat = False
            for var, pos in c:
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                return False
        return True

    # ============ DPLL with Unit Propagation and Pure Literal ============
    
    def dpll_solve(assignment, remaining_clauses, deadline):
        """
        DPLL solver with unit propagation and pure literal elimination.
        assignment: list of (True/False/None) for n variables
        remaining_clauses: list of clause indices that are still unsatisfied
        Returns: assignment if solved, None if unsat or timeout
        """
        if time.time() > deadline:
            return None
        
        # Work with a copy of assignment
        assign = assignment[:]
        
        # Active clauses: those not yet satisfied
        # We'll track which clauses are still active
        active = set(remaining_clauses)
        
        # Unit propagation queue
        changed = True
        while changed:
            changed = False
            
            # Check for satisfied clauses and remove them
            to_remove = set()
            for ci in active:
                c = clauses_data[ci]
                for var, pos in c:
                    if assign[var] is not None and assign[var] == pos:
                        to_remove.add(ci)
                        break
            if to_remove:
                active -= to_remove
                changed = True
            
            if not active:
                # All clauses satisfied
                # Fill in unassigned variables with False
                result = [False if a is None else a for a in assign]
                return result
            
            # Check for empty clause (conflict)
            for ci in active:
                c = clauses_data[ci]
                all_assigned = True
                for var, pos in c:
                    if assign[var] is None:
                        all_assigned = False
                        break
                if all_assigned:
                    # Clause is not satisfied (we already removed satisfied ones)
                    return None  # Conflict
            
            # Unit propagation: find clauses with exactly one unassigned literal
            unit_clauses = []
            for ci in active:
                c = clauses_data[ci]
                unassigned = []
                satisfied = False
                for var, pos in c:
                    if assign[var] is not None:
                        if assign[var] == pos:
                            satisfied = True
                            break
                    else:
                        unassigned.append((var, pos))
                if not satisfied:
                    if len(unassigned) == 1:
                        unit_clauses.append((ci, unassigned[0]))
            
            if unit_clauses:
                for ci, (var, pos) in unit_clauses:
                    if assign[var] is None:
                        assign[var] = pos
                        changed = True
                continue
            
            # Pure literal elimination
            # Find literals that only appear in one polarity
            lit_polarity = {}  # (var, pos) -> set of clause indices
            for ci in active:
                c = clauses_data[ci]
                for var, pos in c:
                    if assign[var] is None:
                        key = (var, pos)
                        if key not in lit_polarity:
                            lit_polarity[key] = set()
                        lit_polarity[key].add(ci)
            
            # Find pure literals
            pure = {}
            for (var, pos), cis in lit_polarity.items():
                # Check if the opposite polarity appears
                opp_key = (var, not pos)
                if opp_key not in lit_polarity:
                    pure[var] = pos
            
            if pure:
                for var, pos in pure.items():
                    if assign[var] is None:
                        assign[var] = pos
                changed = True
                continue
        
        # Now we need to branch
        # Pick a variable to branch on - use VSIDS-like heuristic
        # Count occurrences in active clauses
        var_score = {}
        for ci in active:
            c = clauses_data[ci]
            for var, pos in c:
                if assign[var] is None:
                    var_score[var] = var_score.get(var, 0) + 1
        
        if not var_score:
            # No unassigned variables in active clauses
            result = [False if a is None else a for a in assign]
            return result
        
        # Pick variable with highest score
        var = max(var_score, key=lambda v: var_score[v])
        
        # Try True first, then False
        for val in [True, False]:
            new_assign = assign[:]
            new_assign[var] = val
            
            # Determine which clauses remain active
            new_active = set()
            for ci in active:
                c = clauses_data[ci]
                sat = False
                for v, p in c:
                    if new_assign[v] is not None and new_assign[v] == p:
                        sat = True
                        break
                if not sat:
                    new_active.add(ci)
            
            if not new_active:
                result = [False if a is None else a for a in new_assign]
                return result
            
            # Recurse
            result = dpll_solve(new_assign, new_active, deadline)
            if result is not None:
                return result
        
        return None  # Both branches failed

    # ============ Improved WalkSAT as fallback ============
    
    def walk_sat(deadline):
        """Improved WalkSAT with incremental updates."""
        best_assignment = None
        best_unsat = m + 1
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            rng = random.Random(attempt * 7919 + 13)
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Precompute masks
            def sat_mask(assignment, ci):
                c = clauses_data[ci]
                mask = 0
                for k, (var, pos) in enumerate(c):
                    if assignment[var] == pos:
                        mask |= (1 << k)
                return mask
            
            masks = [sat_mask(assignment, ci) for ci in range(m)]
            unsat = set(ci for ci in range(m) if masks[ci] == 0)
            
            if not unsat:
                return assignment
            
            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]
            
            noise = 0.15
            max_flips = 5000
            last_unsat_count = len(unsat)
            no_improve = 0
            
            for _flip in range(max_flips):
                if time.time() > deadline:
                    break
                if not unsat:
                    return assignment
                
                cur_unsat = len(unsat)
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                
                if cur_unsat < last_unsat_count:
                    no_improve = 0
                    last_unsat_count = cur_unsat
                else:
                    no_improve += 1
                
                if no_improve > 50:
                    noise = min(0.6, noise + 0.03)
                    no_improve = 0
                
                # Select unsatisfied clause
                ci = rng.choice(list(unsat))
                c = clauses_data[ci]
                
                # Score each literal flip
                best_score = -1
                best_k = -1
                for k, (var, pos) in enumerate(c):
                    score = 0
                    for cj in var_to_clauses[var]:
                        if masks[cj] == 0:
                            cj_data = clauses_data[cj]
                            for k2, (v2, p2) in enumerate(cj_data):
                                if v2 == var and p2 == (not assignment[var]):
                                    score += 1
                                    break
                    score += rng.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_k = k
                
                if best_k == -1:
                    continue
                
                if rng.random() > noise:
                    k = best_k
                else:
                    unsat_positions = [k2 for k2, (v2, p2) in enumerate(c)
                                      if not (assignment[v2] == p2)]
                    if unsat_positions and rng.random() > 0.3:
                        k = rng.choice(unsat_positions)
                    else:
                        k = rng.randrange(len(c))
                
                var, pos = c[k]
                assignment[var] = not assignment[var]
                
                # Update masks
                for cj in var_to_clauses[var]:
                    new_mask = sat_mask(assignment, cj)
                    old_mask = masks[cj]
                    if old_mask == 0 and new_mask != 0:
                        unsat.discard(cj)
                    elif old_mask != 0 and new_mask == 0:
                        unsat.add(cj)
                    masks[cj] = new_mask
        
        if best_assignment is not None:
            return best_assignment
        
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]

    # ============ Main strategy ============
    
    # Try DPLL first with a time limit
    dpll_deadline = time.time() + 120  # Give DPLL up to 120s
    
    # Initialize with None for all variables
    initial_assign = [None] * n
    initial_active = set(range(m))
    
    # Increase recursion limit for DPLL
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100000)
    
    try:
        result = dpll_solve(initial_assign, initial_active, dpll_deadline)
        if result is not None:
            # Verify
            if evaluate_full(result):
                return result
    except RecursionError:
        pass
    except Exception:
        pass
    finally:
        sys.setrecursionlimit(old_limit)
    
    # Fall back to WalkSAT
    remaining_deadline = deadline - 10
    result = walk_sat(remaining_deadline)
    
    if result is not None:
        if evaluate_full(result):
            return result
    
    # Last resort: random
    rng = random.Random(42)
    return [rng.random() < 0.5 for _ in range(n)]
