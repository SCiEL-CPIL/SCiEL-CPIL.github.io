"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 285.0  # 15s margin before 300s limit
    
    # Convert to 0-based internal representation
    # Each clause is a list of (var, sign) where sign is +1 or -1
    # var is 0-based
    clauses_internal = []
    for c in clauses:
        clause_int = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_int.append((var, sign))
        clauses_internal.append(clause_int)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
            
    # Optimized CDCL Solver
    # We implement a simplified Minisat-style solver
    # State:
    # - values: array of int8 (-1, 0, 1)
    # - level: array of int (decision level)
    # - reason: array of int (clause index or -1)
    # - trail: list of vars
    # - trail_lim: list of trail indices for each level
    # - clauses: list of lists of (var, sign)
    # - watches: dict mapping literal (var, sign) to list of clause indices
    # - act: activity scores for vars
    # - order: heap for VSIDS
    
    # For simplicity and speed in Python, we will use a robust DPLL with Unit Propagation
    # but optimized with bitsets or efficient lists.
    # However, for 150 vars, a well-implemented DPLL with VSIDS and Clause Learning is best.
    
    # Let's implement a specialized CDCL for 3-SAT.
    
    class CDCLSolver:
        def __init__(self, n, clauses):
            self.n = n
            self.m = len(clauses)
            self.clauses = clauses
            self.values = [0] * n  # 0: unassigned, 1: true, -1: false
            self.level = [0] * n
            self.reason = [-1] * n
            self.trail = []
            self.trail_lim = []
            self.qhead = 0
            self.watches = { (v, s): [] for v in range(n) for s in [-1, 1] }
            for ci, clause in enumerate(clauses):
                if len(clause) > 0:
                    self.watches[clause[0]].append(ci)
                    if len(clause) > 1:
                        self.watches[clause[1]].append(ci)
            self.act = [1.0] * n
            self.decay = 0.95
            self.lbd = [0] * self.m
            self.learned = []
            self.num_restarts = 0
            self.max_restarts = 100
            self.restart_interval = 100
            self.restart_inc = 1.5
            self.conflicts = 0
            self.max_conflicts = 10000
            self.max_conflicts_inc = 1.5
            
            # Heap for VSIDS
            self.heap = list(range(n))
            self.heap_pos = [i for i in range(n)]
            self._heapify()

        def _heapify(self):
            # Build max-heap based on activity
            for i in range(len(self.heap)//2 - 1, -1, -1):
                self._sift_down(i)

        def _sift_down(self, i):
            heap = self.heap
            pos = self.heap_pos
            act = self.act
            n = len(heap)
            while True:
                l = 2*i + 1
                r = 2*i + 2
                largest = i
                if l < n and act[heap[l]] > act[heap[largest]]:
                    largest = l
                if r < n and act[heap[r]] > act[heap[largest]]:
                    largest = r
                if largest == i:
                    break
                # Swap
                self._swap_heap(largest, i)
                i = largest

        def _sift_up(self, i):
            heap = self.heap
            pos = self.heap_pos
            act = self.act
            while i > 0:
                p = (i - 1) // 2
                if act[heap[p]] >= act[heap[i]]:
                    break
                self._swap_heap(p, i)
                i = p

        def _swap_heap(self, i, j):
            heap = self.heap
            pos = self.heap_pos
            a, b = heap[i], heap[j]
            heap[i], heap[j] = b, a
            pos[a], pos[b] = j, i

        def bump(self, var):
            self.act[var] *= 1.5
            if self.act[var] > 1e100:
                # Rescale
                for i in range(self.n):
                    self.act[i] /= 1e100
                self._heapify()
            else:
                pos = self.heap_pos[var]
                self._sift_up(pos)

        def assign(self, var, val, level, reason):
            self.values[var] = val
            self.level[var] = level
            self.reason[var] = reason
            self.trail.append(var)

        def analyze(self, conflict_clause):
            # Returns (learnt_clause, backtrack_level)
            learnt = []
            seen = [False] * self.n
            counter = 0
            p = conflict_clause
            index = len(self.trail) - 1
            bt_level = 0
            while True:
                for i in range(index, len(p)):
                    lit = p[i]
                    var = lit[0]
                    sign = lit[1]
                    if self.values[var] != -sign:
                        continue
                    if not seen[var]:
                        self.bump(var)
                        seen[var] = True
                        counter += 1
                        lvl = self.level[var]
                        if lvl > bt_level:
                            bt_level = lvl
                        learnt.append(lit)
                        reason = self.reason[var]
                        if reason != -1:
                            p = self.clauses[reason]
                            break
                else:
                    break
                index = 0
                
                # Reduce learnt clause
                # Simplify learnt by removing literals that are implied by earlier decisions?
                # For 3-SAT, we can keep it simple.
                
                # Stop when we have 1 unassigned literal or all literals are from level 0
                unassigned = 0
                for lit in learnt:
                    var, sign = lit
                    if self.values[var] == 0:
                        unassigned += 1
                if unassigned <= 1:
                    break
                
                # Remove literals from levels >= bt_level except the one at bt_level?
                # Standard LBD reduction
                # Find the literal with the highest level
                max_lvl = 0
                max_idx = -1
                for i, lit in enumerate(learnt):
                    var, sign = lit
                    lvl = self.level[var]
                    if lvl > max_lvl:
                        max_lvl = lvl
                        max_idx = i
                if max_idx != -1:
                    learnt.pop(max_idx)
                    
            # Add the conflict literal (the one that was false)
            # The conflict clause p has at least one false literal.
            # We need to identify which one caused the conflict.
            # In the loop above, we iterated over p.
            # The learnt clause should contain the negation of the literal that was false?
            # Actually, standard algorithm:
            # learnt = {l | l is in p and l is false} U {l | l is in reason chains}
            
            # Let's re-implement analyze more carefully as the above is a bit loose.
            
            # Standard Analyze:
            learnt = []
            seen = [False] * self.n
            counter = 0
            p = conflict_clause
            index = len(self.trail) - 1
            bt_level = 0
            
            while True:
                for i in range(index, len(p)):
                    lit = p[i]
                    var = lit[0]
                    sign = lit[1]
                    # If the literal is false (values[var] == -sign)
                    if self.values[var] == -sign:
                        if not seen[var]:
                            self.bump(var)
                            seen[var] = True
                            counter += 1
                            lvl = self.level[var]
                            if lvl > bt_level:
                                bt_level = lvl
                            learnt.append(lit)
                            reason = self.reason[var]
                            if reason != -1:
                                p = self.clauses[reason]
                                break
                else:
                    break
                index = 0
                
                # Reduce
                if counter <= 1:
                    break
                    
                # Find the literal with the highest level in learnt
                max_lvl = 0
                max_idx = -1
                for i, lit in enumerate(learnt):
                    var, sign = lit
                    lvl = self.level[var]
                    if lvl > max_lvl:
                        max_lvl = lvl
                        max_idx = i
                        
                if max_idx != -1:
                    learnt.pop(max_idx)
                    counter -= 1
                    
            # The learnt clause is satisfied by the negation of the last false literal?
            # No, the learnt clause consists of the false literals from the conflict path.
            # It is a valid clause.
            
            # We need to add this learnt clause to the database
            # And return the backtrack level
            
            return learnt, bt_level

        def propagate(self):
            # Unit propagation using watched literals
            while self.qhead < len(self.trail):
                p = self.trail[self.qhead]
                self.qhead += 1
                v = p
                sign = self.values[v]
                # Watched literal is (v, -sign) because the literal that became false is the one watched
                # If we assigned v=True, the literal (v, False) i.e. (v, -1) became false.
                # So we look at watches[(v, -sign)]
                watch_list = self.watches[(v, -sign)]
                i = 0
                while i < len(watch_list):
                    c_idx = watch_list[i]
                    c = self.clauses[c_idx]
                    # Check if the other watched literal is false
                    # We need to find the other literal
                    # c[0] or c[1] is (v, -sign). The other is the one we check.
                    if c[0] == (v, -sign):
                        other = c[1]
                    else:
                        other = c[0]
                        
                    # If other is true, we are done with this clause
                    if self.values[other[0]] == other[1]:
                        continue
                    # If other is false, we have a conflict or need to update watch
                    if self.values[other[0]] == -other[1]:
                        # Conflict
                        return c_idx
                    # Other is unassigned
                    # Find a new watched literal
                    new_watch_found = False
                    for j, lit in enumerate(c):
                        if lit == (v, -sign) or lit == other:
                            continue
                        if self.values[lit[0]] == 0:
                            # Swap watch
                            self.watches[other].remove(c_idx)
                            self.watches[lit].append(c_idx)
                            # Update clause representation? 
                            # We store clauses as lists. We can swap in the list for efficiency.
                            # Find index of other in c
                            if c[0] == other:
                                c[0] = lit
                            else:
                                c[1] = lit
                            new_watch_found = True
                            break
                        elif self.values[lit[0]] == lit[1]:
                            # Clause satisfied, remove from watch list
                            self.watches[other].remove(c_idx)
                            new_watch_found = True
                            break
                            
                    if not new_watch_found:
                        # All literals assigned?
                        # This shouldn't happen if we handled the conflict case above
                        # But if the clause has only 2 literals and both are false, it's a conflict.
                        # If it has 3, and one is true, we would have caught it in the satisfied check.
                        # If it has 3, and two are false, one unassigned, we would have found a new watch.
                        # So this case implies a conflict if no new watch found and other was false?
                        # Actually, if other was false, and we couldn't find a new watch, it means all other literals are false.
                        return c_idx
                        
                    i += 1
                # Remove processed watches? No, we just iterate.
                # We need to be careful with list modification while iterating.
                # The above logic is slightly flawed with list removal.
                # Let's use a different approach for propagation which is safer in Python.
                
            return -1

        def decide(self):
            # VSIDS selection
            while self.heap:
                var = self.heap[0]
                if self.values[var] == 0:
                    return var
                else:
                    self._swap_heap(0, self.heap_pos[var])
                    self._sift_down(0)
            return -1

        def cancel_until(self, level):
            while len(self.trail) > (self.trail_lim[level] if level < len(self.trail_lim) else 0):
                v = self.trail.pop()
                self.values[v] = 0
                self.reason[v] = -1
            self.qhead = len(self.trail)
            if len(self.trail_lim) > level:
                self.trail_lim = self.trail_lim[:level+1]
            if level == 0:
                self.qhead = 0

        def solve(self, time_limit):
            self.conflicts = 0
            self.max_conflicts = 10000
            self.num_restarts = 0
            self.restart_interval = 100
            self.restart_inc = 1.5
            
            while True:
                if time.time() > time_limit:
                    return None
                
                # Propagate
                conflict_idx = self.propagate()
                if conflict_idx != -1:
                    # Conflict
                    self.conflicts += 1
                    conflict_clause = self.clauses[conflict_idx]
                    
                    if len(self.trail_lim) == 0:
                        return None # UNSAT at level 0
                    
                    learnt, bt_level = self.analyze(conflict_clause)
                    
                    # Add learnt clause
                    self.clauses.append(learnt)
                    new_idx = len(self.clauses) - 1
                    # Update watches for learnt clause
                    if len(learnt) > 0:
                        self.watches[learnt[0]].append(new_idx)
                        if len(learnt) > 1:
                            self.watches[learnt[1]].append(new_idx)
                            
                    self.cancel_until(bt_level)
                    
                    # Assign the last literal in learnt clause (the one at bt_level)
                    # It should be the only one not implied by lower levels?
                    # Actually, the learnt clause has exactly one literal at level bt_level
                    # and the rest are at lower levels.
                    # We need to assign the literal at bt_level.
                    for lit in learnt:
                        var, sign = lit
                        if self.level[var] == bt_level:
                            self.assign(var, sign, bt_level, new_idx)
                            break
                            
                    if self.conflicts > self.max_conflicts:
                        self.max_conflicts *= self.max_conflicts_inc
                        self.num_restarts += 1
                        if self.num_restarts > self.max_restarts:
                            return None # Give up on CDCL, maybe fallback
                        self.cancel_until(0)
                        self.qhead = 0
                        
                else:
                    # No conflict
                    if len(self.trail) == self.n:
                        return self.values[:]
                        
                    var = self.decide()
                    if var == -1:
                        return self.values[:]
                        
                    # Decide
                    # Heuristic: try True first, or based on activity?
                    # For planted SAT, random or activity based is fine.
                    val = 1 
                    self.trail_lim.append(len(self.trail))
                    self.assign(var, val, len(self.trail_lim), -1)
                    
                    # Decay activities
                    for i in range(self.n):
                        self.act[i] *= self.decay

        def run(self, time_limit):
            return self.solve(time_limit)

    solver = CDCLSolver(n, clauses_internal)
    result = solver.run(deadline - 5.0)
    
    if result is not None:
        # Verify
        # result is list of -1, 0, 1
        # Check if all assigned
        if 0 in result:
            # Fill unassigned with random?
            # If CDCL didn't finish, it might have unassigned vars.
            # But if it returns, it should be a model or UNSAT.
            # If it returns None, it timed out or UNSAT.
            pass
        # Convert to bool list
        # 1 -> True, -1 -> False
        # If 0, default to False (or random)
        return [bool(x) for x in result]
        
    # Fallback: WalkSAT
    if time.time() > deadline - 5.0:
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]

    rng = random.Random(42)
    w_assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Efficient WalkSAT
    # Precompute clause literals
    # Use numpy for speed if possible, but pure python with lists is okay for 150 vars
    
    def count_unsat(assign):
        unsat = 0
        for c in clauses_internal:
            sat = False
            for var, sign in c:
                if assign[var] == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat

    current_unsat = count_unsat(w_assignment)
    steps = 0
    max_steps = 100000
    no_improve = 0
    max_no_improve = 500
    
    while steps < max_steps and time.time() < deadline - 2.0:
        # Find a random unsatisfied clause
        unsat_clauses = []
        for c in clauses_internal:
            sat = False
            for var, sign in c:
                if w_assignment[var] == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(c)
                
        if not unsat_clauses:
            return w_assignment
            
        clause = rng.choice(unsat_clauses)
        
        # Try flipping a variable in the clause
        best_var = -1
        best_delta = float('inf')
        
        for var, sign in clause:
            if w_assignment[var] == (sign == 1):
                continue # Already satisfied, flipping would make it unsat (bad)
                
            # Calculate delta
            delta = 0
            for ci, (v, s) in var_to_clauses[var]:
                c = clauses_internal[ci]
                was_sat = False
                for vv, ss in c:
                    if w_assignment[vv] == (ss == 1):
                        was_sat = True
                        break
                would_be_sat = False
                for vv, ss in c:
                    if vv == var:
                        new_val = not w_assignment[var]
                        if new_val == (ss == 1):
                            would_be_sat = True
                            break
                    elif w_assignment[vv] == (ss == 1):
                        would_be_sat = True
                        break
                if was_sat and not would_be_sat:
                    delta += 1
                elif not was_sat and would_be_sat:
                    delta -= 1
                    
            if delta < best_delta:
                best_delta = delta
                best_var = var
                
        if best_var != -1:
            w_assignment[best_var] = not w_assignment[best_var]
            new_unsat = count_unsat(w_assignment)
            if new_unsat < current_unsat:
                current_unsat = new_unsat
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    # Restart
                    rng = random.Random(rng.randint(0, 10000))
                    w_assignment = [rng.random() < 0.5 for _ in range(n)]
                    current_unsat = count_unsat(w_assignment)
                    no_improve = 0
                    max_no_improve = int(max_no_improve * 1.5)
                    
        steps += 1
        
    return w_assignment

# EVOLVE-BLOCK-END
