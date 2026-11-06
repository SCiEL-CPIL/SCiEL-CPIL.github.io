import time
import random
import sys

class CDCLSolver:
    def __init__(self, n, clauses):
        self.n = n
        self.clauses = clauses
        self.m = len(clauses)
        
        # assignment: 0 for False, 1 for True, -1 for Unassigned
        self.assignment = [-1] * (n + 1)  # 1-based indexing for literals
        self.level = [0] * (n + 1)
        self.reason = [-1] * (n + 1)
        
        # Watches: watches[literal] = list of (clause_idx, literal_index_in_clause)
        # We need watches for both positive and negative literals.
        # Literal l is satisfied if assignment[abs(l)] == (1 if l>0 else 0)
        # We will store watches in a list of lists, indexed by literal (1..2n)
        # Map literal to index: 1..n are positive, n+1..2n are negative?
        # Let's use 1..n for var 1..n. 
        # Literal value: +v or -v.
        # watches will be a dict or list of size 2*n+1.
        # Index 1..n: positive literal v.
        # Index n+1..2n: negative literal v (corresponds to -v).
        
        self.watches = [[] for _ in range(2 * n + 1)]
        
        # Build watches
        for i, clause in enumerate(self.clauses):
            for j, lit in enumerate(clause):
                # Determine watch index
                if lit > 0:
                    watch_idx = lit
                else:
                    watch_idx = n - lit  # -(-v) = v? No.
                    # If lit = -v, we want to watch -v.
                    # Let's define watch index for literal L:
                    # If L > 0, idx = L
                    # If L < 0, idx = n - L  (e.g., -1 -> n+1)
                    # Wait, -(-1) = 1. n - (-1) is wrong.
                    # Let's use: idx = lit if lit > 0 else n - lit.
                    # If lit = -1, n - (-1) = n+1. Correct.
                    # If lit = -n, n - (-n) = 2n. Correct.
                    pass 
                if lit > 0:
                    watch_idx = lit
                else:
                    watch_idx = n - lit
                self.watches[watch_idx].append((i, j))
        
        # VSIDS
        self.activity = [0.0] * (n + 1)
        self.var_inc = 1.0
        self.var_decay = 0.95
        self.trail = []
        self.trail_lim = [0]
        self.learnt_clauses = []
        self.learnt_idx = 0
        self.rng = random.Random(42)
        
        # Conflict analysis
        self.conflict_clause = None
        
    def is_satisfied(self, lit):
        v = abs(lit)
        if self.assignment[v] == -1:
            return False
        if lit > 0:
            return self.assignment[v] == 1
        else:
            return self.assignment[v] == 0

    def is_false(self, lit):
        v = abs(lit)
        if self.assignment[v] == -1:
            return False
        if lit > 0:
            return self.assignment[v] == 0
        else:
            return self.assignment[v] == 1

    def assign(self, v, val, level, reason_idx):
        self.assignment[v] = val
        self.level[v] = level
        self.reason[v] = reason_idx
        self.trail.append(v)

    def unassign(self, v):
        self.assignment[v] = -1
        self.level[v] = 0
        self.reason[v] = -1

    def propagate(self):
        # Watched literals propagation
        # We need a queue of literals to check?
        # Standard MiniSAT uses a queue of literals that became false.
        # When a literal becomes false, we check its watches.
        
        # We will implement a simple loop that scans the trail?
        # No, that's slow. We need a queue.
        
        # Let's use a list as a queue for false literals.
        # Initially, we don't have any false literals.
        # When we decide, we assign a value. The opposite literal becomes false.
        # We push the opposite literal to the queue.
        
        # This implementation will be a bit different.
        # We'll maintain a queue of "false literals" that need their watches updated.
        
        # For simplicity in this Python implementation, we might just
        # check all clauses? No, that's O(m) per step.
        
        # Let's try a simpler approach: 
        # After each assignment, we don't immediately propagate.
        # We call a `propagate_all` that iterates through all clauses?
        # With n=150, m~450, O(m) is fine.
        
        # Let's go with the simple O(m) check for unit clauses.
        # It's less efficient but easier to get right in Python.
        
        while True:
            unit_found = False
            for i, clause in enumerate(self.clauses + self.learnt_clauses):
                # Check if clause is unit
                num_unassigned = 0
                num_false = 0
                last_unassigned = None
                last_false = None
                
                for lit in clause:
                    if self.assignment[abs(lit)] == -1:
                        num_unassigned += 1
                        last_unassigned = lit
                    elif self.is_false(lit):
                        num_false += 1
                        last_false = lit
                    else:
                        # Satisfied
                        break
                
                if num_false == len(clause):
                    # Conflict
                    return i  # Return clause index of conflict
                elif num_unassigned == 1 and num_false == 0:
                    # Unit clause
                    # Assign last_unassigned to satisfy
                    v = abs(last_unassigned)
                    val = 1 if last_unassigned > 0 else 0
                    current_level = len(self.trail_lim) - 1
                    self.assign(v, val, current_level, i)
                    unit_found = True
                    # Restart loop to check for new units
                    break
                elif num_unassigned == 0 and num_false == 0:
                    # Empty clause? Or all satisfied?
                    # If all satisfied, do nothing.
                    pass
            
            if not unit_found:
                break
                
        return -1  # No conflict

    def pick_branch(self):
        # VSIDS
        # Decay activities
        self.var_inc /= self.var_decay
        for v in range(1, self.n + 1):
            self.activity[v] /= self.var_decay
            
        # Pick unassigned var with highest activity
        best_v = -1
        best_act = -1.0
        for v in range(1, self.n + 1):
            if self.assignment[v] == -1:
                if self.activity[v] > best_act:
                    best_act = self.activity[v]
                    best_v = v
                    
        if best_v == -1:
            return None
            
        # Randomize polarity slightly? Or use phase saving?
        # Let's use random polarity for simplicity, or majority phase.
        # For planted instances, random is often fine.
        val = 1 if self.rng.random() < 0.5 else 0
        
        self.activity[best_v] = self.var_inc
        
        return best_v, val

    def analyze_conflict(self, conflict_clause_idx):
        # Simplified conflict analysis
        # We need to find the asserting literal.
        # This is complex to implement correctly in Python quickly.
        # Let's use a simpler heuristic: Backtrack to the level of the second highest variable in the conflict clause.
        
        clause = self.clauses + self.learnt_clauses
        c = clause[conflict_clause_idx]
        
        # Find the variable with the highest level in the conflict clause
        # and the second highest.
        levels = []
        for lit in c:
            v = abs(lit)
            if self.assignment[v] != -1:
                levels.append((self.level[v], v))
        
        if not levels:
            return 0, None
            
        levels.sort(reverse=True)
        
        # If the highest level is the current level, we need to backtrack.
        # The asserting literal is the one with the highest level?
        # No, the asserting literal is the one that remains after backtracking.
        
        # Let's just backtrack to the level of the second highest variable.
        if len(levels) >= 2:
            backtrack_level = levels[1][0]
            # The asserting literal is the one with the highest level?
            # Actually, we need to identify which literal is the "reason" for the conflict at that level.
            # For now, let's just backtrack and try the opposite value.
            # This is "Backjumping" without learning.
            
            # Find the literal corresponding to the highest level variable
            assert_lit = None
            for lit in c:
                if abs(lit) == levels[0][1]:
                    assert_lit = lit
                    break
            
            return backtrack_level, assert_lit
        else:
            # Only one variable in conflict?
            backtrack_level = levels[0][0] - 1
            if backtrack_level < 0:
                backtrack_level = 0
            return backtrack_level, None

    def solve(self, timeout=325.0):
        start_time = time.time()
        
        # Initial unit propagation
        conflict_idx = self.propagate()
        if conflict_idx != -1:
            return None # UNSAT? But task says satisfiable.
            
        while True:
            if time.time() - start_time > timeout - 5.0:
                break
                
            # Check if all variables assigned
            if len(self.trail) == self.n:
                # Check if all clauses satisfied
                all_sat = True
                for clause in self.clauses:
                    sat = False
                    for lit in clause:
                        if self.is_satisfied(lit):
                            sat = True
                            break
                    if not sat:
                        all_sat = False
                        break
                if all_sat:
                    # Return assignment
                    result = [False] * self.n
                    for v in range(1, self.n + 1):
                        result[v-1] = (self.assignment[v] == 1)
                    return result
                else:
                    # Conflict at root?
                    return None

            # Decide
            decision = self.pick_branch()
            if decision is None:
                break
                
            v, val = decision
            current_level = len(self.trail_lim)
            self.trail_lim.append(len(self.trail))
            self.assign(v, val, current_level, -1)
            
            # Propagate
            conflict_idx = self.propagate()
            
            if conflict_idx != -1:
                # Conflict!
                backtrack_level, assert_lit = self.analyze_conflict(conflict_idx)
                
                # Backtrack
                while len(self.trail) > (self.trail_lim[backtrack_level] if backtrack_level < len(self.trail_lim) else 0):
                    v_to_unassign = self.trail.pop()
                    self.unassign(v_to_unassign)
                
                # Remove trail limits above backtrack level
                while len(self.trail_lim) > backtrack_level + 1:
                    self.trail_lim.pop()
                
                # If we have an asserting literal, assign it
                if assert_lit is not None:
                    v_assert = abs(assert_lit)
                    val_assert = 1 if assert_lit > 0 else 0
                    # Assign at current level (which is now backtrack_level)
                    self.assign(v_assert, val_assert, backtrack_level, conflict_idx)
                    
                # If backtrack_level is 0 and no assert, we might be stuck.
                # In a real solver, we'd learn a clause. Here we just restart or fail.
                if backtrack_level == 0 and assert_lit is None:
                    # Try restarting with random assignment?
                    # For now, break and return best effort?
                    # Since instances are satisfiable, this shouldn't happen often.
                    pass

        # If we time out, return the current assignment if it looks good?
        # Or just return None?
        # The task requires returning a list of booleans.
        # If we didn't find a solution, we should return a heuristic assignment.
        # Let's return the current trail assignment.
        result = [False] * self.n
        for v in range(1, self.n + 1):
            if self.assignment[v] != -1:
                result[v-1] = (self.assignment[v] == 1)
            else:
                result[v-1] = self.rng.random() < 0.5
                
        # Verify
        all_sat = True
        for clause in self.clauses:
            sat = False
            for lit in clause:
                if self.is_satisfied(lit):
                    sat = True
                    break
            if not sat:
                all_sat = False
                break
                
        if all_sat:
            return result
        else:
            # Fallback to WalkSAT for the remaining time?
            # Let's just return the best effort.
            return result

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 325.0
    
    if n == 0:
        return []
    if not clauses:
        return [False] * n

    # Try CDCL
    solver = CDCLSolver(n, clauses)
    result = solver.solve(timeout=timeout - 10.0)
    
    if result is not None:
        # Verify result
        valid = True
        for clause in clauses:
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                if (lit > 0 and result[v]) or (lit < 0 and not result[v]):
                    sat = True
                    break
            if not sat:
                valid = False
                break
        if valid:
            return result

    # Fallback to WalkSAT if CDCL fails or times out
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time < 5.0:
        return [False] * n

    # Simple WalkSAT
    rng = random.Random(42)
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_list = []
        signs_list = []
        for lit in clause:
            var_idx = abs(lit) - 1
            vars_list.append(var_idx)
            signs_list.append(lit > 0)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
        
    var_to_clauses = [[] for _ in range(n)]
    for i in range(len(clauses)):
        for j, v in enumerate(clause_vars[i]):
            var_to_clauses[v].append((i, clause_signs[i][j]))
            
    def init_state(assignment):
        clause_sat_count = [0] * len(clauses)
        for i in range(len(clauses)):
            count = 0
            for j, v in enumerate(clause_vars[i]):
                if assignment[v] == clause_signs[i][j]:
                    count += 1
            clause_sat_count[i] = count
        return clause_sat_count
        
    def is_satisfied(clause_sat_count):
        return all(c > 0 for c in clause_sat_count)
        
    def flip_var(v, assignment, clause_sat_count):
        old_val = assignment[v]
        assignment[v] = not old_val
        new_val = assignment[v]
        for ci, sign in var_to_clauses[v]:
            if new_val == sign:
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1
                
    best_assignment = None
    best_unsat_count = len(clauses) + 1
    
    weight = 0.5
    while time.time() - start_time < timeout - 2.0:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count = init_state(assignment)
        
        if is_satisfied(clause_sat_count):
            return assignment
            
        current_unsat = sum(1 for c in clause_sat_count if c == 0)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
            
        max_flips = 5000
        for _ in range(max_flips):
            if time.time() - start_time > timeout - 2.0:
                break
                
            unsat_indices = [i for i, c in enumerate(clause_sat_count) if c == 0]
            if not unsat_indices:
                return assignment
                
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            vars_in_clause = clause_vars[ci]
            
            if rng.random() < weight:
                v = vars_in_clause[rng.randrange(len(vars_in_clause))]
            else:
                best_v = vars_in_clause[0]
                best_damage = float('inf')
                for v in vars_in_clause:
                    damage = 0
                    for other_ci, sign in var_to_clauses[v]:
                        if other_ci != ci and clause_sat_count[other_ci] > 0:
                            damage += 1
                    if damage < best_damage:
                        best_damage = damage
                        best_v = v
                v = best_v
                
            flip_var(v, assignment, clause_sat_count)
            
            current_unsat = sum(1 for c in clause_sat_count if c == 0)
            if current_unsat == 0:
                return assignment
                
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
                
    if best_assignment:
        return best_assignment
    return [False] * n

