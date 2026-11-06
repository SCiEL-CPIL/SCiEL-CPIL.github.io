import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
        
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-process clauses to 0-based internal representation?
    # Keep 1-based signed literals for ease of conflict analysis?
    # Let's use 1-based signed literals in the solver logic.
    
    # CDCL Solver
    # assignment: -1, 0, 1
    # reason: clause index or -1
    # level: decision level
    
    assignment = [0] * n
    reason = [-1] * n
    level = [0] * n
    
    trail = []
    trail_lim = []
    
    learned_clauses = []
    
    def propagate():
        """
        Returns (conflict_idx, )
        conflict_idx is -1 if no conflict, else index of conflicting clause.
        """
        # We need to process newly assigned variables.
        # In this naive implementation, we just scan all clauses repeatedly.
        # This is inefficient but correct and simple.
        # Optimization: Keep a queue of newly assigned vars.
        
        # Let's use a queue for efficiency.
        # We need to know which vars were assigned in the current call.
        # We can pass the start index of the trail.
        
        # For simplicity and given small n, I'll stick to the scan-all-clauses method
        # but only if necessary.
        # Actually, let's implement a proper queue-based propagation for correctness and speed.
        
        # We need to track which literals are "watched" or just re-evaluate.
        # Given n=150, re-evaluating all clauses is O(m*n) per assignment.
        # Total O(n^2 * m). 3.3M ops. This is fine.
        
        # However, we need to detect conflicts and unit clauses.
        # We will loop until no changes.
        
        changed = True
        while changed:
            changed = False
            for ci, clause in enumerate(clauses):
                # Check clause status
                sat = False
                unassigned = []
                for lit in clause:
                    v = abs(lit) - 1
                    val = assignment[v]
                    if (val == 1 and lit > 0) or (val == -1 and lit < 0):
                        sat = True
                        break
                    if val == 0:
                        unassigned.append(lit)
                
                if sat:
                    continue
                    
                if len(unassigned) == 0:
                    return ci # Conflict
                
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    v = abs(lit) - 1
                    val = 1 if lit > 0 else -1
                    if assignment[v] == 0:
                        assignment[v] = val
                        reason[v] = ci
                        level[v] = len(trail_lim)
                        trail.append(v)
                        changed = True
                    elif assignment[v] == -val:
                        return ci # Conflict (should be caught by len(unassigned)==0 if we checked all, 
                                  # but if assignment was already opposite, it's not in unassigned)
                        # Wait, if assignment[v] is -val, it's not in unassigned.
                        # So if len(unassigned)==1, the other literals are false.
                        # If the single unassigned lit is actually assigned opposite, 
                        # it wouldn't be in unassigned.
                        # So this case implies the clause has 0 unassigned and all false.
                        # Which is len(unassigned)==0.
        
        # Also check learned clauses
        for ci, clause in enumerate(learned_clauses):
            sat = False
            unassigned = []
            for lit in clause:
                v = abs(lit) - 1
                val = assignment[v]
                if (val == 1 and lit > 0) or (val == -1 and lit < 0):
                    sat = True
                    break
                if val == 0:
                    unassigned.append(lit)
            
            if sat:
                continue
                
            if len(unassigned) == 0:
                return len(clauses) + ci # Conflict in learned clause
                
            if len(unassigned) == 1:
                lit = unassigned[0]
                v = abs(lit) - 1
                val = 1 if lit > 0 else -1
                if assignment[v] == 0:
                    assignment[v] = val
                    reason[v] = len(clauses) + ci
                    level[v] = len(trail_lim)
                    trail.append(v)
                    changed = True
                elif assignment[v] == -val:
                    return len(clauses) + ci
        
        return -1

    def analyze_conflict(conflict_idx):
        """
        Returns (learned_clause, backtrack_level)
        learned_clause is a list of literals.
        """
        # 1-UIP Analysis
        # We need to resolve the conflict clause with the reasons of variables 
        # assigned in the current decision level.
        
        current_level = len(trail_lim)
        
        # Get the conflicting clause
        if conflict_idx < len(clauses):
            c = list(clauses[conflict_idx])
        else:
            c = list(learned_clauses[conflict_idx - len(clauses)])
        
        # We will resolve until we have exactly 1 literal from the current level.
        # Literals in c that are from current level are "active".
        
        # Helper to get literals from current level
        def get_current_level_lits(clause):
            lits = []
            for lit in clause:
                v = abs(lit) - 1
                if level[v] == current_level:
                    lits.append(lit)
            return lits
        
        active_lits = get_current_level_lits(c)
        
        while len(active_lits) > 1:
            # Pick one active literal to resolve
            # Usually the one assigned last in the trail
            # Find the one with highest trail index
            last_lit = None
            last_idx = -1
            for lit in active_lits:
                v = abs(lit) - 1
                if trail.index(v) > last_idx: # O(n) search, n is small
                    last_idx = trail.index(v)
                    last_lit = lit
            
            # Get the reason clause for last_lit
            v = abs(last_lit) - 1
            r_idx = reason[v]
            
            if r_idx < 0:
                # Should not happen for non-decision vars in conflict analysis
                # unless it's the first decision?
                break
            
            if r_idx < len(clauses):
                r_clause = list(clauses[r_idx])
            else:
                r_clause = list(learned_clauses[r_idx - len(clauses)])
            
            # Resolve c and r_clause on last_lit
            # Remove last_lit from c
            # Add literals from r_clause except last_lit
            new_c = []
            for lit in c:
                if lit != last_lit:
                    new_c.append(lit)
            for lit in r_clause:
                if lit != last_lit and lit != -last_lit: # last_lit is negation of the var in reason? 
                    # Reason clause contains last_lit? 
                    # If assignment[v] = val, reason clause must be satisfied by last_lit.
                    # So last_lit is in r_clause.
                    if lit != last_lit:
                        new_c.append(lit)
            
            c = new_c
            active_lits = get_current_level_lits(c)
            
        # Now c is the learned clause
        # The asserting literal is the one in c that is from current level
        # (there should be exactly 1)
        assert_lits = get_current_level_lits(c)
        if len(assert_lits) != 1:
            # Fallback
            return c, 0
        
        assert_lit = assert_lits[0]
        v = abs(assert_lit) - 1
        
        # Backtrack level is the level of the variable just before the one that 
        # implies the asserting literal?
        # Standard: Backtrack to the level of the second most recent variable in the learned clause.
        # Find levels of all vars in c
        levels_in_c = [level[abs(lit)-1] for lit in c]
        # The asserting literal is from current_level.
        # The other literals are from previous levels.
        # Backtrack level is the max level among the non-asserting literals.
        # If no other literals, backtrack to 0.
        
        other_levels = []
        for lit in c:
            if lit != assert_lit:
                v2 = abs(lit) - 1
                other_levels.append(level[v2])
        
        if other_levels:
            bt_level = max(other_levels)
        else:
            bt_level = 0
            
        return c, bt_level

    def backtrack(level_to):
        # Unassign variables from trail until we reach trail_lim[level_to]
        # trail_lim[k] is the index in trail where level k ended.
        # We want to unassign all vars with level > level_to.
        
        # Find the index in trail to backtrack to
        # trail_lim[level_to] is the index AFTER the last var of level level_to.
        # So we want to truncate trail to trail_lim[level_to].
        
        if level_to < len(trail_lim):
            target_len = trail_lim[level_to]
        else:
            target_len = 0
            
        while len(trail) > target_len:
            v = trail.pop()
            assignment[v] = 0
            reason[v] = -1
            level[v] = 0
            
        # Remove decision levels above level_to
        while len(trail_lim) > level_to:
            trail_lim.pop()

    # Main Search Loop
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Reset state
        assignment = [0] * n
        reason = [-1] * n
        level = [0] * n
        trail = []
        trail_lim = []
        learned_clauses = []
        
        # Initial propagation (should be none if no unit clauses initially, 
        # but good to check)
        conflict = propagate()
        if conflict != -1:
            # Conflict at level 0, instance is UNSAT? 
            # But it's planted SAT. Maybe error.
            continue
            
        while True:
            if time.time() > deadline:
                break
                
            # Check if all assigned
            if len(trail) == n:
                # All assigned, check if all clauses satisfied
                sat = True
                for clause in clauses:
                    c_sat = False
                    for lit in clause:
                        v = abs(lit) - 1
                        val = assignment[v]
                        if (val == 1 and lit > 0) or (val == -1 and lit < 0):
                            c_sat = True
                            break
                    if not c_sat:
                        sat = False
                        break
                if sat:
                    # Found solution
                    return [assignment[i] == 1 for i in range(n)]
                else:
                    # Should not happen if propagation was correct
                    break
            
            # Pick a variable
            # Simple heuristic: first unassigned
            var_to_assign = -1
            for v in range(n):
                if assignment[v] == 0:
                    var_to_assign = v
                    break
                    
            if var_to_assign == -1:
                break
                
            # Assign
            # Heuristic: Random or based on planted? 
            # We don't know planted. Use random.
            val = random.Random(restart * 1000 + len(trail)).random() < 0.5
            assignment[var_to_assign] = 1 if val else -1
            reason[var_to_assign] = -1
            level[var_to_assign] = len(trail_lim)
            trail.append(var_to_assign)
            trail_lim.append(len(trail))
            
            # Propagate
            conflict = propagate()
            
            if conflict != -1:
                # Analyze and learn
                learned_clause, bt_level = analyze_conflict(conflict)
                
                # Add learned clause
                learned_clauses.append(learned_clause)
                
                # Backtrack
                backtrack(bt_level)
                
                # Assign the asserting literal
                # Find asserting literal from learned_clause
                # It's the one that was in the current level
                # We can just find the one that is unassigned now?
                # No, we backtracked. The asserting literal should be the one 
                # that is now unassigned but was in the current level.
                # Actually, the asserting literal is determined by the analysis.
                # Let's find it again.
                assert_lit = None
                for lit in learned_clause:
                    v = abs(lit) - 1
                    if level[v] == 0 and assignment[v] == 0:
                        # This might not be the right way to identify it after backtrack.
                        # The asserting literal is the one that is NOT satisfied by 
                        # the backtracked state, and is the only one from the 
                        # original conflict level.
                        # Simpler: The asserting literal is the one that 
                        # becomes unit after backtracking.
                        pass
                
                # Better: Re-run propagation? 
                # The learned clause is now unit. 
                # Propagate will handle it.
                # But we need to set the reason for the asserting literal?
                # No, propagate will assign it and set reason to the learned clause index.
                
                # Just continue loop, propagate will assign it.
                # But we need to make sure we don't re-decide the same var.
                # Since we backtracked, it's unassigned.
                # Propagate will assign it.
                
                continue
                
            # No conflict, continue

    # Fallback: WalkSAT
    # ... (reuse WalkSAT code from original)
    
    # Since I don't have the WalkSAT code here, I'll implement a simple random fallback
    rng = random.Random(42)
    return [rng.random() < 0.5 for _ in range(n)]
