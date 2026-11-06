# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import heapq
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin

    def lit_to_idx(lit):
        var = abs(lit) - 1
        if lit > 0:
            return 2 * var
        else:
            return 2 * var + 1

    def idx_to_var(idx):
        return idx // 2

    def idx_to_sign(idx):
        return 1 if idx % 2 == 0 else -1

    def neg_idx(idx):
        return idx ^ 1

    internal_clauses = []
    for c in clauses:
        ic = [lit_to_idx(l) for l in c]
        internal_clauses.append(ic)
    
    m = len(internal_clauses)

    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(internal_clauses):
        for lit in clause:
            var = idx_to_var(lit)
            var_clauses[var].append((ci, lit))

    def count_unsat(assignment):
        unsat = 0
        for clause in internal_clauses:
            sat = False
            for lit in clause:
                var = idx_to_var(lit)
                sign = idx_to_sign(lit)
                if assignment[var] == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat

    # CDCL Solver
    class CDCLSolver:
        def __init__(self, n, clauses, var_clauses):
            self.n = n
            self.clauses = clauses
            self.m = len(clauses)
            self.var_clauses = var_clauses
            
            self.values = [-1] * n
            self.trail = []
            self.trail_pos = [-1] * n
            self.reason = [-1] * n
            
            self.watched = defaultdict(list)
            self.watched_lits = [None] * self.m
            
            for ci, clause in enumerate(self.clauses):
                if len(clause) >= 2:
                    l1, l2 = clause[0], clause[1]
                    self.watched[l1].append(ci)
                    self.watched[l2].append(ci)
                    self.watched_lits[ci] = [l1, l2]
                elif len(clause) == 1:
                    self.watched[clause[0]].append(ci)
                    self.watched_lits[ci] = [clause[0]]
                    
            self.activity = [1.0] * n
            self.decay = 0.95
            self.bump = 1.0
            self.heap = [(-self.activity[i], i) for i in range(n)]
            heapq.heapify(self.heap)
            
            self.conflicts = 0
            self.restart_interval = 100
            self.next_restart = self.restart_interval
            
            self.learned_clauses = []
            self.max_learned = 3000
            
            self.decision_level = 0
            self.trail_lim = []
            
            self.prop_queue = []
            
        def is_true(self, lit_idx):
            v = idx_to_var(lit_idx)
            s = idx_to_sign(lit_idx)
            return self.values[v] == s
            
        def is_false(self, lit_idx):
            v = idx_to_var(lit_idx)
            s = idx_to_sign(lit_idx)
            return self.values[v] == -s
            
        def is_unassigned(self, lit_idx):
            v = idx_to_var(lit_idx)
            return self.values[v] == -1
            
        def assign(self, var, val, reason_clause_idx):
            self.values[var] = val
            self.trail.append(var)
            self.trail_pos[var] = len(self.trail) - 1
            self.reason[var] = reason_clause_idx
            false_lit = 2 * var + (0 if val == 1 else 1)
            self.prop_queue.append(false_lit)
            
        def unassign(self, var):
            self.values[var] = -1
            self.trail_pos[var] = -1
            self.reason[var] = -1
            
        def backjump(self, level):
            lim = self.trail_lim[level] if level < len(self.trail_lim) else 0
            while len(self.trail) > lim:
                var = self.trail.pop()
                self.unassign(var)
            self.decision_level = level
            while len(self.trail_lim) > level:
                self.trail_lim.pop()
                
        def bump_activity(self, clause):
            for lit in clause:
                var = idx_to_var(lit)
                self.activity[var] += self.bump
            self.bump *= self.decay
            for i in range(self.n):
                self.activity[i] *= self.decay
                
        def select_variable(self):
            while self.heap:
                _, var = heapq.heappop(self.heap)
                if self.values[var] == -1:
                    return var
            return -1
            
        def get_var_level(self, var):
            pos = self.trail_pos[var]
            if pos < 0:
                return 0
            level = 0
            for i in range(len(self.trail_lim)):
                if pos >= self.trail_lim[i]:
                    level = i + 1
            return level
            
        def propagate(self):
            self.conflict_clause = None
            
            while self.prop_queue:
                false_lit = self.prop_queue.pop()
                
                watch_list = self.watched[false_lit]
                i = 0
                while i < len(watch_list):
                    ci = watch_list[i]
                    clause = self.clauses[ci] if ci < self.m else self.learned_clauses[ci - self.m]
                    wl = self.watched_lits[ci]
                    
                    if wl[0] == false_lit:
                        other_pos = 1
                    else:
                        other_pos = 0
                    other_lit = wl[other_pos]
                    
                    if self.is_true(other_lit):
                        i += 1
                        continue
                    
                    new_lit = None
                    for lit in clause:
                        if lit == false_lit or lit == other_lit:
                            continue
                        if self.is_unassigned(lit):
                            new_lit = lit
                            break
                    
                    if new_lit is not None:
                        self.watched[new_lit].append(ci)
                        wl[other_pos] = new_lit
                        i += 1
                        continue
                    
                    if self.is_true(other_lit):
                        i += 1
                        continue
                    elif self.is_false(other_lit):
                        self.conflict_clause = ci
                        return False
                    else:
                        self.assign(idx_to_var(other_lit), idx_to_sign(other_lit), ci)
                        i += 1
                        continue
                    
                    i += 1
                
            return True
            
        def analyze_conflict(self, conflict_clause_idx):
            learnt = []
            seen = set()
            current_level = self.decision_level
            backjump_level = 0
            counter = 0
            
            clause = self.clauses[conflict_clause_idx] if conflict_clause_idx < self.m else self.learned_clauses[conflict_clause_idx - self.m]
            
            queue = []
            for lit in clause:
                var = idx_to_var(lit)
                if self.trail_pos[var] >= 0:
                    level = self.get_var_level(var)
                    if level == current_level:
                        queue.append(lit)
                        seen.add(var)
                    elif level > backjump_level:
                        backjump_level = level
                        
            while queue:
                lit = queue.pop()
                var = idx_to_var(lit)
                
                if len(seen) == 1 and var == self.trail[-1]:
                    break
                    
                reason_idx = self.reason[var]
                if reason_idx < 0:
                    continue
                    
                reason_clause = self.clauses[reason_idx] if reason_idx < self.m else self.learned_clauses[reason_idx - self.m]
                
                for lit2 in reason_clause:
                    var2 = idx_to_var(lit2)
                    if var2 in seen:
                        continue
                    seen.add(var2)
                    
                    level2 = self.get_var_level(var2)
                    if level2 == current_level:
                        queue.append(lit2)
                    elif level2 > backjump_level:
                        backjump_level = level2
                        
                learnt.append(lit)
                
            return learnt, max(backjump_level, 0)
            
        def add_learned_clause(self, clause, level):
            if len(clause) <= 1:
                if len(clause) == 1:
                    var = idx_to_var(clause[0])
                    if self.values[var] == -1:
                        self.assign(var, idx_to_sign(clause[0]), -1)
                return
                
            if len(self.learned_clauses) >= self.max_learned:
                return
                
            idx = self.m + len(self.learned_clauses)
            self.learned_clauses.append(clause)
            
            l1, l2 = clause[0], clause[1]
            self.watched[l1].append(idx)
            self.watched[l2].append(idx)
            
            while len(self.watched_lits) <= idx:
                self.watched_lits.append(None)
            self.watched_lits[idx] = [l1, l2]
            
            for lit in clause:
                if self.is_true(lit):
                    return
                if self.is_false(lit):
                    self.conflict_clause = idx
                    return
                if self.is_unassigned(lit):
                    var = idx_to_var(lit)
                    if self.get_var_level(var) == level:
                        self.assign(var, idx_to_sign(lit), idx)
                    return
                    
        def search(self, time_limit):
            self.start = time.time()
            self.time_limit = time_limit
            
            while True:
                if time.time() - self.start > self.time_limit:
                    return False
                
                if not self.propagate():
                    self.conflicts += 1
                    if self.decision_level == 0:
                        return False
                    
                    learnt, backjump_level = self.analyze_conflict(self.conflict_clause)
                    
                    self.bump_activity(learnt)
                    
                    self.backjump(backjump_level)
                    
                    if len(learnt) > 0:
                        self.add_learned_clause(learnt, backjump_level)
                        
                    if self.conflicts >= self.next_restart:
                        self.backjump(0)
                        self.restart_interval = int(self.restart_interval * 1.5)
                        self.next_restart = self.conflicts + self.restart_interval
                    continue
                
                var = self.select_variable()
                if var == -1:
                    return True
                
                self.decision_level += 1
                self.trail_lim.append(len(self.trail))
                
                val = 1 if random.random() < 0.5 else 0
                self.assign(var, val, -1)
                
                if self.conflicts % 200 == 0:
                    self.heap = [(-self.activity[i], i) for i in range(self.n)]
                    heapq.heapify(self.heap)
                    
            return False

    def walksat(rng, max_restarts=1000, max_steps=10000):
        best_assignment = None
        best_unsat = m + 1
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            clause_unsat = np.zeros(m, dtype=bool)
            for ci, clause in enumerate(internal_clauses):
                sat = False
                for lit in clause:
                    var = idx_to_var(lit)
                    sign = idx_to_sign(lit)
                    if assignment[var] == (sign == 1):
                        sat = True
                        break
                clause_unsat[ci] = not sat
                
            unsat_count = int(np.sum(clause_unsat))
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                
            noise = 0.5
            for step in range(max_steps):
                if time.time() > deadline:
                    return best_assignment
                    
                if unsat_count == 0:
                    return assignment
                    
                unsat_indices = np.where(clause_unsat)[0]
                ci = int(rng.choice(unsat_indices))
                clause = internal_clauses[ci]
                
                best_var = -1
                best_score = -float('inf')
                
                for lit in clause:
                    var = idx_to_var(lit)
                    score = 0
                    
                    for other_ci, other_lit in var_clauses[var]:
                        if other_ci == ci:
                            continue
                        if not clause_unsat[other_ci]:
                            other_var = idx_to_var(other_lit)
                            other_sign = idx_to_sign(other_lit)
                            
                            if var == other_var:
                                new_val = not assignment[var]
                                if new_val != (other_sign == 1):
                                    only_sat = True
                                    for lit2 in internal_clauses[other_ci]:
                                        if lit2 == other_lit:
                                            continue
                                        v2 = idx_to_var(lit2)
                                        s2 = idx_to_sign(lit2)
                                        if assignment[v2] == (s2 == 1):
                                            only_sat = False
                                            break
                                    if only_sat:
                                        score += 1
                                        
                    score += rng.random() * noise
                    if score > best_score:
                        best_score = score
                        best_var = var
                        
                if best_var == -1:
                    lit = clause[rng.randrange(len(clause))]
                    best_var = idx_to_var(lit)
                    
                old_val = assignment[best_var]
                assignment[best_var] = not old_val
                
                for other_ci, other_lit in var_clauses[best_var]:
                    sat = False
                    for lit2 in internal_clauses[other_ci]:
                        v2 = idx_to_var(lit2)
                        s2 = idx_to_sign(lit2)
                        if assignment[v2] == (s2 == 1):
                            sat = True
                            break
                    was_unsat = clause_unsat[other_ci]
                    is_unsat = not sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    clause_unsat[other_ci] = is_unsat
                    
                if unsat_count == 0:
                    return assignment
                    
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
                    
                if step % 500 == 0 and step > 0:
                    noise = max(0.1, noise * 0.95)
                    
        return best_assignment

    # Strategy 1: Try CDCL first
    if time.time() < deadline - 15:
        cdcl = CDCLSolver(n, internal_clauses, var_clauses)
        if cdcl.search(time_limit=120.0):
            assignment = [bool(v) for v in cdcl.values]
            return assignment

    # Strategy 2: WalkSAT with multiple seeds
    best_result = None
    best_unsat = m + 1
    
    rng1 = random.Random(42)
    result1 = walksat(rng1, max_restarts=50, max_steps=5000)
    if result1 is not None:
        u = count_unsat(result1)
        if u < best_unsat:
            best_unsat = u
            best_result = result1
        if u == 0:
            return result1

    if time.time() < deadline - 10:
        rng2 = random.Random(123)
        result2 = walksat(rng2, max_restarts=200, max_steps=20000)
        if result2 is not None:
            u = count_unsat(result2)
            if u < best_unsat:
                best_unsat = u
                best_result = result2
            if u == 0:
                return result2

    if time.time() < deadline - 5:
        rng3 = random.Random(777)
        result3 = walksat(rng3, max_restarts=100, max_steps=10000)
        if result3 is not None:
            u = count_unsat(result3)
            if u < best_unsat:
                best_unsat = u
                best_result = result3
            if u == 0:
                return result3

    if time.time() < deadline - 2 and best_result is not None and best_unsat > 0:
        rng4 = random.Random(999)
        result4 = walksat(rng4, max_restarts=500, max_steps=5000)
        if result4 is not None:
            u = count_unsat(result4)
            if u < best_unsat:
                best_unsat = u
                best_result = result4
            if u == 0:
                return result4

    return best_result

# EVOLVE-BLOCK-END
