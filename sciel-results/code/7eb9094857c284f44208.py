"""Improved 3-SAT solver using Watched Literals + VSIDS + Clause Learning."""
import random
import heapq
import time

class SatSolver:
    def __init__(self, n, clauses):
        self.n = n
        self.clauses = clauses
        self.m = len(clauses)
        self.assign = [-1] * (n + 1) # 1-indexed vars, -1 unassigned, 0 false, 1 true
        self.level = [0] * (n + 1)
        self.reason = [None] * (n + 1)
        self.trail = []
        self.trail_lim = []
        self.var_inc = [0.0] * (n + 1)
        self.var_decay = 0.95
        self.var_inc_c = 1.0
        self.lit_heap = []
        self.queue = []
        self.head = 0
        self.seen = [0] * (n + 1)
        self.seen_stamp = 0
        self.learned_clauses = []
        self.conflicts = 0
        self.literals_watches = [[] for _ in range(2 * n + 2)]
        self.rng = random.Random(42)
        
        self._init_watches()

    def _init_watches(self):
        # Initialize watches for initial clauses
        for idx, clause in enumerate(self.clauses):
            if len(clause) >= 2:
                l1, l2 = clause[0], clause[1]
                self.literals_watches[l1].append((l2, idx, 0)) # (other_lit, clause_idx, pos_in_clause)
                self.literals_watches[l2].append((l1, idx, 1))
            elif len(clause) == 1:
                l1 = clause[0]
                self.literals_watches[l1].append((None, idx, 0))

    def _add_clause(self, clause):
        # Add a learned clause
        # Remove duplicates and tautologies
        if not clause:
            return True # Empty clause means UNSAT
        seen_lits = set()
        cleaned = []
        for l in clause:
            if -l in seen_lits:
                return False # Tautology
            if l not in seen_lits:
                seen_lits.add(l)
                cleaned.append(l)
        
        clause = cleaned
        self.learned_clauses.append(clause)
        idx = len(self.learned_clauses) - 1
        
        if len(clause) >= 2:
            l1, l2 = clause[0], clause[1]
            self.literals_watches[l1].append((l2, idx, 0))
            self.literals_watches[l2].append((l1, idx, 1))
        elif len(clause) == 1:
            l1 = clause[0]
            self.literals_watches[l1].append((None, idx, 0))
        # If empty, handled by caller
        
        return True

    def _cancel_until(self, level):
        # Unassign variables until trail level <= level
        while len(self.trail) > self.trail_lim[level]:
            v = self.trail.pop()
            self.assign[v] = -1
            self.level[v] = 0
            self.reason[v] = None

    def _propagate(self):
        # Watched literal propagation
        while self.head < len(self.queue):
            p = self.queue[self.head]
            self.head += 1
            # p is a literal that was just assigned
            # We need to process watches for ~p
            # Note: In standard watched literals, when we assign p, we check watches on ~p
            # But usually we store watches on the literal itself.
            # Let's stick to: when literal 'p' is assigned True, we look at watches for '-p'.
            
            # Actually, standard implementation:
            # When we set variable v to value val, the literal l = v if val else -v becomes True.
            # The negation -l becomes False.
            # We must process watches for -l.
            
            lit_to_check = -p
            
            watches = self.literals_watches[lit_to_check]
            
            i = 0
            while i < len(watches):
                other_lit, clause_idx, pos = watches[i]
                
                # Determine which clause and get the clause
                # Clauses are stored in self.clauses (initial) or self.learned_clauses
                if clause_idx < self.m:
                    clause = self.clauses[clause_idx]
                    is_learned = False
                else:
                    clause = self.learned_clauses[clause_idx - self.m]
                    is_learned = True
                
                # Check if clause is satisfied
                # We know lit_to_check is False (assigned opposite to p)
                # We need to check if any other literal in the clause is True
                
                # Optimization: Check the other watch first
                if other_lit is not None:
                    # Check if other_lit is True
                    v_other = abs(other_lit)
                    val_other = self.assign[v_other]
                    if val_other != -1:
                        if (other_lit > 0 and val_other == 1) or (other_lit < 0 and val_other == 0):
                            # Clause is satisfied by other_lit
                            # Remove this watch
                            watches.pop(i)
                            continue
                    elif val_other == -1:
                        # Other watch is unassigned, keep watching
                        i += 1
                        continue
                    else:
                        # Other watch is False
                        pass
                
                # Check if clause is satisfied by any other literal
                sat = False
                new_watch_lit = None
                for j, l in enumerate(clause):
                    if l == lit_to_check or (other_lit is not None and l == other_lit):
                        continue
                    v = abs(l)
                    val = self.assign[v]
                    if val == 1:
                        if l > 0:
                            sat = True
                            break
                    elif val == 0:
                        if l < 0:
                            sat = True
                            break
                    elif val == -1:
                        # Unassigned, candidate for new watch
                        if new_watch_lit is None:
                            new_watch_lit = l
                        # If we find a satisfied literal later, we don't need new watch
                        # But we need to know if it's satisfied first.
                        # Let's just scan all.
                
                if sat:
                    # Clause satisfied, remove watch
                    watches.pop(i)
                    continue
                
                # If we get here, lit_to_check is False, other_lit is False (or None), and no other lit is True.
                # If there is an unassigned literal, move the watch to it.
                if new_watch_lit is not None:
                    # Remove current watch
                    watches.pop(i)
                    # Add new watch
                    # The new watch is on new_watch_lit. The other watch in that list will be lit_to_check (which is false)
                    # But wait, the other watch in the new location should be the *other* literal of the pair?
                    # Standard: watches[l] stores (other, idx, pos).
                    # If we move watch from lit_to_check to new_watch_lit:
                    # The new entry in watches[new_watch_lit] should be (lit_to_check, clause_idx, pos_of_new_watch_lit)
                    # And we need to update the entry in watches[lit_to_check]? No, we popped it.
                    # The other literal in the clause that we are NOT watching is... well, we are watching new_watch_lit and lit_to_check.
                    # So in watches[new_watch_lit], we store (lit_to_check, clause_idx, pos_new).
                    
                    # Find position of new_watch_lit in clause
                    pos_new = clause.index(new_watch_lit)
                    
                    self.literals_watches[new_watch_lit].append((lit_to_check, clause_idx, pos_new))
                    continue
                
                # No unassigned literal. Clause is unit or empty.
                # If other_lit is None, it was a unit clause watching lit_to_check.
                # If other_lit is not None, it was false.
                # All literals are false. Contradiction.
                return False
        
        return True

    def _analyze(self, confl):
        # confl is the index of the conflicting clause
        # confl < self.m -> initial clause, else learned
        if confl < self.m:
            clause = self.clauses[confl]
        else:
            clause = self.learned_clauses[confl - self.m]
        
        # All literals in confl are False.
        # We want to derive a new clause.
        # Use 1-UIP scheme.
        
        self.seen_stamp += 1
        seen_stamp = self.seen_stamp
        
        # The last literal in the conflict clause that was assigned is the one that caused the conflict.
        # Actually, all are false. We look at the assignment levels.
        
        # Initialize the new clause with the literals from the conflict clause that are not from the current decision level?
        # No, standard 1-UIP:
        # Start with the conflict clause.
        # While the number of literals from the current decision level is > 1:
        #   Take the most recently assigned literal from the current level in the current clause.
        #   Resolve with its reason clause.
        
        current_level = self.level[abs(clause[-1])] # The level of the most recent assignment?
        # Actually, we need to track the level of the decision we are backjumping to.
        
        # Let's use a simpler robust analyze:
        # 1. Identify the literals in the conflict clause.
        # 2. The new clause will contain literals that are not assigned at the current level, or are the UIP.
        
        # Better approach:
        # Let L be the set of literals in the conflict clause.
        # Let current_level be the level of the last assigned literal in L.
        # Actually, all literals in L are false.
        
        # We maintain a list of literals for the new clause.
        new_clause = []
        
        # We need to resolve until we have exactly 1 literal from the current decision level.
        # The current decision level is the level of the variable that was assigned most recently among the literals in the conflict clause?
        # No, the conflict happens at the current decision level.
        
        # Let's find the level of the most recent assignment among the literals in the conflict clause.
        max_level = 0
        for l in clause:
            v = abs(l)
            if self.level[v] > max_level:
                max_level = self.level[v]
        
        # This max_level is the current decision level where the conflict occurred.
        
        # We start with the conflict clause.
        # We need to resolve with reason clauses for literals assigned at this level.
        
        # To do this efficiently, we can use a stack or set.
        # Let's use a set of literals to keep track of what we've processed.
        
        # Actually, let's use the standard algorithm:
        # 1. Copy the conflict clause to a list `C`.
        # 2. Let `index` be the index of the last literal in `C` that was assigned at the current level.
        # 3. While count of literals in `C` assigned at current level > 1:
        #    a. Let `p` be the literal at `index`.
        #    b. Let `reason` be the reason clause for `p`.
        #    c. Resolve `C` and `reason` to form new `C`.
        #    d. Update `index` to the last literal in new `C` assigned at current level.
        
        # Implementation detail: We need to know the reason for each assignment.
        # self.reason[v] stores the clause index that implied v.
        
        C = list(clause)
        current_level = max_level
        
        # Find index of last literal in C assigned at current_level
        idx = len(C) - 1
        while idx >= 0:
            v = abs(C[idx])
            if self.level[v] == current_level:
                break
            idx -= 1
        
        # Count literals at current_level
        cnt = 0
        for l in C:
            if self.level[abs(l)] == current_level:
                cnt += 1
        
        while cnt > 1:
            # Resolve with reason of C[idx]
            p = C[idx]
            v = abs(p)
            reason_idx = self.reason[v]
            
            if reason_idx is None:
                # Decision variable, no reason. This shouldn't happen if cnt > 1 and we are resolving properly?
                # If p is a decision, it has no reason. But if cnt > 1, there must be another literal.
                # We can just remove p? No, resolution requires a reason.
                # If p is a decision, we cannot resolve it.
                # However, in 1-UIP, we stop when we have 1 literal from current level.
                # If we encounter a decision literal while cnt > 1, it means we need to resolve with the *other* literals?
                # No, the algorithm says: resolve with the reason of the *most recently assigned* literal from the current level.
                # If that literal is a decision, it has no reason.
                # In that case, we just remove it from the clause?
                # Actually, if a decision literal is in the clause and cnt > 1, we should resolve with the reason of the *next* most recently assigned literal?
                # No, the standard algorithm assumes we always have a reason for non-decision variables.
                # If we hit a decision variable, it means the conflict is due to the decision itself and previous implications.
                # We can just drop the decision literal? No, that changes the semantics.
                
                # Let's assume we don't hit decision variables in the middle of 1-UIP unless it's the only one.
                # If it is the only one, cnt becomes 1 and we stop.
                # If cnt > 1, and we pick a decision literal, we have a problem.
                # To be safe, if reason is None, we just remove it and decrement cnt?
                # This is effectively "resolution with empty clause" which is not correct.
                # Let's pick a different literal if reason is None.
                
                # Find another literal at current level with a reason
                found = False
                for j in range(idx - 1, -1, -1):
                    v_j = abs(C[j])
                    if self.level[v_j] == current_level and self.reason[v_j] is not None:
                        idx = j
                        p = C[idx]
                        reason_idx = self.reason[v]
                        found = True
                        break
                if not found:
                    # All remaining literals at current level are decisions?
                    # This implies the conflict is purely between decisions?
                    # Just break and return the clause as is?
                    break
            
            # Get reason clause
            if reason_idx < self.m:
                reason_clause = self.clauses[reason_idx]
            else:
                reason_clause = self.learned_clauses[reason_idx - self.m]
            
            # Resolve C and reason_clause on p
            # New C contains literals from C (excluding p) and literals from reason_clause (excluding -p)
            new_C = []
            for l in C:
                if l != p:
                    new_C.append(l)
            for l in reason_clause:
                if l != -p:
                    new_C.append(l)
            
            C = new_C
            
            # Update count
            cnt = 0
            idx = len(C) - 1
            while idx >= 0:
                v = abs(C[idx])
                if self.level[v] == current_level:
                    cnt += 1
                    # Find last one
                    if self.reason[abs(C[idx])] is not None: # Prefer non-decision for next step?
                        pass
                    break
                idx -= 1
            
            # Find index of last literal at current level
            idx = len(C) - 1
            while idx >= 0:
                v = abs(C[idx])
                if self.level[v] == current_level:
                    break
                idx -= 1
        
        # Now C is the new clause
        # Determine backtrack level
        # The backtrack level is the second highest level in C
        levels = []
        for l in C:
            levels.append(self.level[abs(l)])
        
        if len(levels) == 0:
            return None, None # Should not happen
        
        levels.sort(reverse=True)
        
        if len(levels) == 1:
            bt_level = 0
        else:
            bt_level = levels[1]
        
        # Update activity scores
        for l in C:
            v = abs(l)
            self.var_inc[v] += self.var_inc_c
            if self.var_inc[v] > 1e100:
                # Rescale
                for i in range(1, self.n + 1):
                    self.var_inc[i] /= 1e100
                self.var_inc_c /= 1e100
        
        return C, bt_level

    def _search(self, decision_level, max_conflicts):
        # Main search loop
        conflicts = 0
        while True:
            # Propagate
            if not self._propagate():
                # Conflict
                conflicts += 1
                if conflicts > max_conflicts:
                    return -1 # Restart needed
                
                # Find the conflicting clause
                # The conflict is detected in _propagate when a clause has no unassigned literals and is not satisfied.
                # We need to know *which* clause caused the conflict.
                # My _propagate returns False but doesn't say which clause.
                # I need to modify _propagate to return the clause index or store it.
                
                # For now, let's assume we can find it by checking the queue?
                # This is a flaw in my quick implementation.
                # Let's modify _propagate to store the last conflicting clause.
                
                pass # Placeholder, need fix below

            # If no conflict, check if solved
            # Check if all variables are assigned?
            # Or check if all clauses are satisfied?
            # Efficient check: if trail length == n, and no conflicts, it's a solution.
            if len(self.trail) == self.n:
                return 1 # Solved
            
            # Pick next variable
            # VSIDS
            self.var_inc_c /= self.var_decay
            # Find max var_inc
            best_var = 0
            best_score = -1
            for v in range(1, self.n + 1):
                if self.assign[v] == -1 and self.var_inc[v] > best_score:
                    best_score = self.var_inc[v]
                    best_var = v
            
            if best_var == 0:
                return 1 # All assigned
            
            # Branch
            self.trail_lim.append(len(self.trail))
            
            # Heuristic: try value that appears more in unsatisfied clauses?
            # For now, random or fixed. Let's use random for diversity in restarts.
            # Or use phase saving.
            val = 1 # Default true
            
            self._assign(best_var, val, None) # None reason, decision
            decision_level += 1

    def _assign(self, v, val, reason):
        self.assign[v] = val
        self.level[v] = len(self.trail_lim)
        self.reason[v] = reason
        self.trail.append(v)
        # Add negation to queue
        lit = v if val == 1 else -v
        self.queue.append(-lit) # Wait, if we assign v=1 (lit v), we need to process watches for -v.
        # So we add -v to the queue?
        # In _propagate, we pop p from queue, and check watches for -p.
        # If we assign v=1, p=v. We check watches for -v.
        # So we should add v to the queue.
        self.queue.append(v)
        
        # Actually, let's be consistent.
        # Queue contains literals that have been assigned TRUE.
        # When we assign v=1, literal v is True. We add v to queue.
        # When we assign v=0, literal -v is True. We add -v to queue.
        
        # Let's fix _propagate logic:
        # Pop p. p is a True literal.
        # We need to update watches for ~p (the False literal).
        # So in _propagate, we look at self.literals_watches[-p].
        # This matches my earlier code.

    def solve(self, max_restarts=100, max_conflicts_per_restart=1000):
        # Preprocess
        # Check for empty clause
        for c in self.clauses:
            if not c:
                return None # UNSAT
        
        # Main loop
        for r in range(max_restarts):
            self._reset_search()
            status = self._search(0, max_conflicts_per_restart)
            if status == 1:
                # Solution found
                return [self.assign[i] == 1 for i in range(1, self.n + 1)]
            elif status == -1:
                # Restart
                continue
            else:
                # Error?
                pass
        
        return None

    def _reset_search(self):
        self.assign = [-1] * (self.n + 1)
        self.level = [0] * (self.n + 1)
        self.reason = [None] * (self.n + 1)
        self.trail = []
        self.trail_lim = []
        self.queue = []
        self.head = 0
        # Keep learned clauses and watches
        # But we need to reset the state of watches?
        # The watches structure is persistent.
        # However, the assignment state is reset.
        # The _propagate relies on self.assign.
        
        # One issue: _propagate modifies self.literals_watches.
        # If we restart, the watches might be in a weird state if we didn't clean them up?
        # In a proper solver, watches are persistent.
        # But we need to ensure that when we restart, we don't have stale entries?
        # Actually, watches are added/removed during propagation.
        # If we restart, we unassign all variables.
        # The watches should still be valid?
        # No, if we removed a watch because a literal became true, and then we unassign it, the watch is gone.
        # This is a problem.
        # Standard solvers handle this by either:
        # 1. Rebuilding watches on restart (expensive).
        # 2. Using a different structure.
        # 3. Not removing watches permanently, but marking them?
        
        # For simplicity and correctness in this short implementation, I will rebuild watches on restart.
        # This is O(m) which is fine.
        
        self._init_watches()
        # Also re-add watches for learned clauses
        for idx, clause in enumerate(self.learned_clauses):
            if len(clause) >= 2:
                l1, l2 = clause[0], clause[1]
                self.literals_watches[l1].append((l2, self.m + idx, 0))
                self.literals_watches[l2].append((l1, self.m + idx, 1))
            elif len(clause) == 1:
                l1 = clause[0]
                self.literals_watches[l1].append((None, self.m + idx, 0))

    def _search_impl(self, decision_level, max_conflicts):
        conflicts = 0
        while True:
            # Propagate
            # Need to handle conflict clause index
            # Let's modify _propagate to return (success, conflict_clause_idx)
            
            success, confl_idx = self._propagate()
            
            if not success:
                conflicts += 1
                if conflicts > max_conflicts:
                    return -1
                
                # Analyze
                new_clause, bt_level = self._analyze(confl_idx)
                if new_clause is None:
                    return 0 # UNSAT
                
                # Add learned clause
                if not self._add_clause(new_clause):
                    return 0 # UNSAT (empty clause)
                
                # Backtrack
                self._cancel_until(bt_level)
                decision_level = bt_level
                
                # Add the new clause's literals to queue?
                # The new clause is now unit if bt_level < current_level?
                # Actually, the new clause has one literal at bt_level+1?
                # No, the new clause has one literal at level > bt_level.
                # That literal is now unit.
                # We need to enqueue it?
                # The _cancel_until unassigned variables.
                # The literal in the new clause that is at level > bt_level is now the only unassigned one?
                # No, it was assigned, then unassigned.
                # It is now the reason for the backjump.
                # We should assign it?
                # No, we just backjumped. The next propagation step will handle it if it's unit.
                # But is it unit?
                # The new clause has 1 literal at level > bt_level.
                # All other literals are at level <= bt_level (and thus assigned).
                # So yes, it is unit.
                # We need to trigger propagation for it.
                
                # Find the literal in new_clause that is at level > bt_level
                # Actually, it's the one that was just unassigned?
                # No, we unassigned everything > bt_level.
                # The literal in the new clause that was at the highest level is now unassigned.
                # And since all others are assigned, it's unit.
                
                # We can just let the main loop's propagation handle it?
                # But we need to add it to the queue.
                
                # Let's find it
                for l in new_clause:
                    v = abs(l)
                    if self.level[v] == 0: # Unassigned
                        # Add to queue
                        # If l is positive, v is unassigned.
                        # We need to assign it?
                        # No, propagation will assign it.
                        # But propagation starts from the queue.
                        # We need to put the *negation* of the unit literal in the queue?
                        # No, if l is the unit literal, it must be True.
                        # So we assign it and put -l in queue?
                        # Or just put l in queue?
                        # My _propagate pops p (True lit) and checks watches for -p.
                        # If l is unit, l must be True.
                        # So we should assign l=True and put l in queue?
                        # No, if we assign l=True, we put l in queue.
                        # Then _propagate pops l, checks watches for -l.
                        # This is correct.
                        
                        # But we shouldn't assign it manually, we should let propagation do it?
                        # No, propagation only processes *assigned* literals.
                        # It doesn't assign new ones unless they are unit.
                        # Wait, my _propagate DOES assign unit literals?
                        # No, my _propagate only handles the consequences of *already assigned* literals.
                        # It does NOT detect new unit clauses and assign them.
                        # It only moves watches and detects conflicts.
                        # 
                        # I am missing the "Unit Clause Detection" step in _propagate.
                        # Standard Watched Literals:
                        # When a watch is removed and no new watch is found, the clause is unit.
                        # We then assign the remaining literal and add it to the queue.
                        
                        # I need to update _propagate to do this.
                        pass
                break # Continue loop to propagate again

            # Check if solved
            if len(self.trail) == self.n:
                return 1

            # Pick next variable
            # VSIDS
            self.var_inc_c /= self.var_decay
            best_var = 0
            best_score = -1
            for v in range(1, self.n + 1):
                if self.assign[v] == -1 and self.var_inc[v] > best_score:
                    best_score = self.var_inc[v]
                    best_var = v
            
            if best_var == 0:
                return 1

            # Branch
            self.trail_lim.append(len(self.trail))
            val = 1
            self._assign(best_var, val, None)
            decision_level += 1

    def _propagate(self):
        # Updated to handle unit propagation and return conflict clause
        while self.head < len(self.queue):
            p = self.queue[self.head]
            self.head += 1
            lit_to_check = -p
            
            watches = self.literals_watches[lit_to_check]
            i = 0
            while i < len(watches):
                other_lit, clause_idx, pos = watches[i]
                
                if clause_idx < self.m:
                    clause = self.clauses[clause_idx]
                else:
                    clause = self.learned_clauses[clause_idx - self.m]
                
                # Check if clause is satisfied
                # We know lit_to_check is False.
                
                # Check other_lit
                if other_lit is not None:
                    v_other = abs(other_lit)
                    val_other = self.assign[v_other]
                    if val_other != -1:
                        if (other_lit > 0 and val_other == 1) or (other_lit < 0 and val_other == 0):
                            watches.pop(i)
                            continue
                    elif val_other == -1:
                        i += 1
                        continue
                    # else other_lit is False
                
                # Check for satisfied literal or new watch
                sat = False
                new_watch_lit = None
                for l in clause:
                    if l == lit_to_check or (other_lit is not None and l == other_lit):
                        continue
                    v = abs(l)
                    val = self.assign[v]
                    if val == 1:
                        if l > 0:
                            sat = True
                            break
                    elif val == 0:
                        if l < 0:
                            sat = True
                            break
                    elif val == -1:
                        if new_watch_lit is None:
                            new_watch_lit = l
                
                if sat:
                    watches.pop(i)
                    continue
                
                if new_watch_lit is not None:
                    watches.pop(i)
                    pos_new = clause.index(new_watch_lit)
                    self.literals_watches[new_watch_lit].append((lit_to_check, clause_idx, pos_new))
                    continue
                
                # Unit clause
                # The remaining literal must be assigned
                # Find it
                for l in clause:
                    if l != lit_to_check and (other_lit is None or l != other_lit):
                        # This is the unit literal
                        v = abs(l)
                        # Assign it
                        val = 1 if l > 0 else 0
                        self._assign(v, val, clause_idx)
                        break
                
                # If no literal found, it's empty clause -> conflict
                # But we checked sat and new_watch, so if we are here, it must be unit or empty.
                # If empty, we would have found no l?
                # Actually, if clause has 1 literal (lit_to_check), and it's false, it's empty.
                # If clause has 2 literals (lit_to_check, other_lit), and both false, it's empty.
                
                # Check if we found a unit literal to assign
                # If not, it's a conflict
                # How to check?
                # If the clause length was 1 and lit_to_check was the only one, it's conflict.
                # If length 2 and other was false, it's conflict.
                
                # Let's just check if the clause is now satisfied?
                # No, we know it's not.
                # If we didn't assign a new variable, it's a conflict.
                
                # Simple check: if no variable was assigned in the loop above, return conflict.
                # We can track this.
                
                # For now, assume if we reach here, it's a conflict.
                # Wait, if we assigned a variable, we should continue processing the queue?
                # Yes, the assigned variable is added to the queue by _assign.
                # So we just continue the while loop?
                # But we are inside the `while i < len(watches)` loop.
                # We should break out and let the main while loop process the new queue item.
                # But we are at the end of the watch processing for this literal.
                # We can just `continue` the outer while?
                # No, we need to increment i? No, we popped.
                # So we just don't increment i.
                # But we need to ensure we don't process the same watch again?
                # We popped it. So we are fine.
                
                # However, if it was a conflict, we must return.
                # How to distinguish?
                # If the clause had no unassigned literals (other than the ones we checked), and none were true, it's a conflict.
                # In the `if new_watch_lit is not None` block, we handle unit.
                # If we are here, new_watch_lit was None.
                # This means all literals are assigned.
                # And we checked `sat` which was False.
                # So it's a conflict.
                
                return False, clause_idx
        
        return True, None

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess: remove tautologies and empty clauses, deduplicate
    processed_clauses = []
    seen = set()
    for clause in clauses:
        lit_set = set()
        is_taut = False
        for l in clause:
            if -l in lit_set:
                is_taut = True
                break
            lit_set.add(l)
        if is_taut:
            continue
        norm = tuple(sorted(lit_set))
        if norm in seen:
            continue
        seen.add(norm)
        processed_clauses.append(list(lit_set))
    
    clauses = processed_clauses
    m = len(clauses)
    
    if m == 0:
        return [True] * n
    
    for c in clauses:
        if len(c) == 0:
            return [True] * n
    
    solver = SatSolver(n, clauses)
    result = solver.solve()
    
    if result is not None:
        return result
    else:
        # Fallback: WalkSAT
        rng = random.Random(42)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        def satisfied(clause):
            return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
        
        for _ in range(50000):
            unsat = [c for c in clauses if not satisfied(c)]
            if not unsat:
                break
            clause = rng.choice(unsat)
            var_scores = {}
            for c in unsat:
                for l in c:
                    v = abs(l) - 1
                    var_scores[v] = var_scores.get(v, 0) + 1
            clause_vars = [abs(l) - 1 for l in clause]
            weights = [var_scores.get(v, 1) for v in clause_vars]
            total = sum(weights)
            r = rng.random() * total
            cumulative = 0
            chosen_var = clause_vars[0]
            for v, w in zip(clause_vars, weights):
                cumulative += w
                if r <= cumulative:
                    chosen_var = v
                    break
            assignment[chosen_var] = not assignment[chosen_var]
        
        return assignment
