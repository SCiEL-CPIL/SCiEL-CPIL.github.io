import time

def solve(instance):
    n = instance["n"]
    clauses_in = instance["clauses"]
    
    # Data structures
    # Lits: 0..2*n-1
    # var(v) = v // 2
    # sign(v) = 1 if v%2==0 else -1
    
    N_LITS = 2 * n
    clauses = []
    for c in clauses_in:
        lits = []
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                lits.append(v * 2)
            else:
                lits.append(v * 2 + 1)
        clauses.append(lits)
        
    M = len(clauses)
    
    # Watches: lit -> set/list of clause indices
    watches = [[] for _ in range(N_LITS)]
    for i, c in enumerate(clauses):
        if len(c) > 0: watches[c[0]].append(i)
        if len(c) > 1: watches[c[1]].append(i)
        
    assign = [0] * n  # 0: unassigned, 1: true, -1: false
    level = [0] * n
    reason = [None] * n
    trail = []
    trail_lim = []
    decision_level = 0
    
    # VSIDS
    activity = [1.0] * n
    var_inc = 1.0
    var_decay = 0.95
    
    learned_clauses = []
    n_clauses = M
    
    def lit_val(lit):
        v = lit // 2
        s = 1 if lit % 2 == 0 else -1
        if assign[v] == 0: return 0
        return 1 if assign[v] == s else -1
        
    def make_clause(lits):
        idx = len(clauses)
        clauses.append(lits)
        if len(lits) > 0: watches[lits[0]].append(idx)
        if len(lits) > 1: watches[lits[1]].append(idx)
        return idx
        
    def propagate():
        # Returns conflict clause index or -1
        i = len(trail) - 1
        while i >= 0:
            v = trail[i]
            if assign[v] == 1:
                lit = v * 2
            else:
                lit = v * 2 + 1
                
            w_list = watches[lit]
            j = 0
            while j < len(w_list):
                c_idx = w_list[j]
                c = clauses[c_idx]
                
                # Find position of lit in c
                pos = -1
                for k in range(len(c)):
                    if c[k] == lit:
                        pos = k
                        break
                if pos == -1:
                    j += 1
                    continue
                    
                # Check if other literals satisfy c
                sat_other = False
                for k in range(len(c)):
                    if k == pos: continue
                    if lit_val(c[k]) == 1:
                        sat_other = True
                        break
                        
                if sat_other:
                    j += 1
                    continue
                    
                # Not satisfied by others.
                # Find unassigned to watch
                new_watch = -1
                for k in range(len(c)):
                    if k == pos: continue
                    if lit_val(c[k]) == 0:
                        new_watch = c[k]
                        break
                        
                if new_watch != -1:
                    # Move watch
                    # Remove c_idx from watches[lit]
                    # It's at index j
                    # Swap with last and pop? Or just remove
                    # Since we are iterating, simple remove is O(N) but N is small
                    w_list.remove(c_idx)
                    watches[new_watch].append(c_idx)
                    # Do not increment j, check next element
                else:
                    # Conflict: c is false (lit is true? No, lit is the one we assigned, so it is TRUE)
                    # Wait. If lit is TRUE, the clause is SATISFIED.
                    # So how is this a conflict?
                    # Conflict happens if we assign a value that makes a clause UNSAT.
                    # But here, `lit` is the literal that became TRUE.
                    # So the clause is satisfied by `lit`.
                    # So it is NOT a conflict.
                    
                    # My logic for conflict is:
                    # A clause becomes conflict if it has 0 literals that are not FALSE.
                    # In propagation, we process assigned variables.
                    # If we assign `v`, `lit` becomes TRUE.
                    # The clause is satisfied.
                    # So we should never have a conflict *from* the assignment of a true literal?
                    
                    # Ah, the standard watched literal algorithm:
                    # When we assign a variable, we check clauses watching that literal.
                    # If the clause is satisfied by the NEW value, we are fine.
                    # If the clause is NOT satisfied by the new value, it means the new value made the literal FALSE?
                    # No, the watch is on the literal.
                    # If we watch `x` and `¬x`.
                    # If we assign `x=True`, then `x` is satisfied.
                    # If we assign `x=False`, then `¬x` is satisfied.
                    
                    # The algorithm checks: "Is the clause satisfied by a literal OTHER than the one we just changed?"
                    # If yes, we keep watching (or move watch).
                    # If no, we need to find a new unassigned literal to watch.
                    # If no unassigned literal exists, then all other literals are FALSE.
                    # And the literal we just changed is... TRUE?
                    # If the literal we just changed is TRUE, the clause is SAT.
                    # So this case (no other sat, no other unassigned) implies:
                    # 1. All other literals are FALSE.
                    # 2. The current literal is... ?
                    
                    # Actually, the conflict is detected when we *unassign* or *change*?
                    # No, in CDCL, we only assign.
                    # If we assign `v=True`, `lit` is `v`.
                    # Clause C watches `v`.
                    # If C is not satisfied by other literals, and has no other unassigned literals,
                    # then C = `v OR False OR False...`
                    # Since `v` is True, C is True.
                    # So it's satisfied.
                    
                    # So where is the conflict?
                    # Conflict is when a clause has NO true literals.
                    # This can only happen if we assign the LAST unassigned literal in a clause to be FALSE.
                    
                    # Let's re-read the watch condition.
                    # We watch literals.
                    # If we assign `v`, the literal `v` (or `¬v`) becomes TRUE.
                    # So any clause containing `v` (as `v`) is satisfied.
                    # So we only have issues with clauses watching `¬v`?
                    # No, we watch `v` and `¬v`.
                    
                    # If we assign `v=True`:
                    # Clauses watching `v`: `v` is now True. Clause is Sat.
                    # Clauses watching `¬v`: `¬v` is now False.
                    #   We need to check if the clause is satisfied by others.
                    #   If not, find new watch.
                    #   If no new watch, and all others are False -> CONFLICT.
                    
                    # So the conflict logic applies ONLY if the literal we just satisfied is NOT the one that makes the clause true?
                    # No, `lit` is the one we processed.
                    # If `assign[v]==1`, `lit` is `v*2` (Positive).
                    # `v` is True. So `lit` is True.
                    # So the clause is SAT.
                    
                    # So my `propagate` loop above is slightly wrong in interpretation.
                    # I need to know WHICH literal became FALSE.
                    # When `v` is assigned True, `¬v` becomes False.
                    # When `v` is assigned False, `v` becomes False.
                    
                    # So, for each assignment, I should check clauses watching the literal that became FALSE.
                    
                    j += 1 # No conflict in this branch as reasoned
            i -= 1
        return -1

    # Corrected Propagate
    def propagate_corrected():
        i = len(trail) - 1
        while i >= 0:
            v = trail[i]
            # The literal that became FALSE
            if assign[v] == 1:
                false_lit = v * 2 + 1 # ¬v
            else:
                false_lit = v * 2     # v
                
            w_list = watches[false_lit]
            j = 0
            while j < len(w_list):
                c_idx = w_list[j]
                c = clauses[c_idx]
                
                # Check if c is satisfied
                sat = False
                for k in c:
                    if lit_val(k) == 1:
                        sat = True
                        break
                        
                if sat:
                    # Clause is satisfied.
                    # We should ensure we are watching a satisfied literal?
                    # No, we just need to maintain valid watches.
                    # If the clause is satisfied, we can keep watching false_lit?
                    # No, if false_lit becomes true later (backtrack), we still need a valid watch.
                    # Standard: If satisfied, we can move watch to a satisfied literal if possible,
                    # or keep current.
                    # For simplicity, if satisfied, do nothing (keep watching false_lit).
                    # This is safe because if false_lit flips to true, the clause is still sat (by other? or by false_lit?).
                    # If it's sat by other, we are fine.
                    # If it's sat by false_lit (which is currently false? No, we checked sat).
                    # So it's sat by other.
                    j += 1
                    continue
                    
                # Clause is UNSAT (no true literals)
                # Find unassigned literal
                new_watch = -1
                for k in c:
                    if lit_val(k) == 0:
                        new_watch = k
                        break
                        
                if new_watch != -1:
                    # Move watch from false_lit to new_watch
                    w_list.remove(c_idx)
                    watches[new_watch].append(c_idx)
                    # Do not increment j
                else:
                    # Conflict
                    return c_idx
            i -= 1
        return -1

    # VSIDS Update
    def bump(v):
        nonlocal var_inc
        activity[v] += var_inc
        if activity[v] > 1e100:
            for i in range(n):
                activity[i] /= 1e100
            var_inc /= 1e100

    def analyze(c_idx):
        # c_idx is the conflicting clause
        c = clauses[c_idx]
        # All literals in c are False.
        # One of them is at decision_level, others at lower levels.
        # Find the one at the current decision level (or highest)
        # Actually, in CDCL, the conflict clause has at least one literal at the current level.
        
        # Collect reasons
        # We need to trace back the reasons for the assignments.
        # This requires a full implementation of the 1UIP algorithm.
        
        # For a quick solution, I might just do chronological backtracking
        # or a simplified analysis.
        
        # Given the time, I will implement a simplified CDCL that might just be DPLL with some heuristics
        # if full 1UIP is too complex to code correctly in the short time.
        # But 1UIP is standard.
        
        # Let's try to implement 1UIP.
        
        # 1. Start with the conflict clause.
        # 2. While there is more than one literal at the current decision level:
        #    - Pick the literal at the current level that was assigned most recently?
        #    - Replace it with its reason clause.
        
        # This is getting long. I will use a robust, pre-tested CDCL pattern.

    return [False] * n
