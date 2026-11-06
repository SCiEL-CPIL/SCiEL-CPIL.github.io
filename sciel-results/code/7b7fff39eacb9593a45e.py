import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 285.0
    
    # --- Preprocessing ---
    pos_counts = [0] * n
    neg_counts = [0] * n
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_counts[v] += 1
            else:
                neg_counts[v] += 1
                
    assignment = [None] * n
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
                    continue
            else:
                new_c.append(lit)
        if not satisfied:
            if len(new_c) == 0:
                return [random.random() < 0.5 for _ in range(n)]
            new_clauses.append(new_c)
            
    clauses = new_clauses
    if not clauses:
        final_assignment = [False] * n
        for i in range(n):
            if assignment[i] is not None:
                final_assignment[i] = assignment[i]
            else:
                final_assignment[i] = random.random() < 0.5
        return final_assignment

    var_map = {}
    new_n = 0
    for v in range(n):
        if assignment[v] is None:
            var_map[v] = new_n
            new_n += 1
            
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
        
    # --- CDCL Solver ---
    assign = [None] * new_n
    level = [0] * new_n
    reason = [None] * new_n
    trail = []
    trail_lim = [0]
    
    watches = {i: [] for i in range(1, new_n + 1)}
    watches.update({-i: [] for i in range(1, new_n + 1)})
    
    clauses_all = [c[:] for c in remapped_clauses]
    
    for i, c in enumerate(clauses_all):
        if len(c) >= 2:
            watches[c[0]].append(i)
            watches[c[1]].append(i)
            
    activity = [1.0] * new_n
    var_inc = 1.0
    decay = 0.95
    phase = [False] * new_n
    conflicts = 0
    restart_base = 100
    luby_i = 0
    luby_seq = []
    
    def gen_luby(n):
        seq = []
        for i in range(1, n+1):
            p = 1
            while p * 2 <= i:
                p *= 2
            seq.append(p)
        return seq
    luby_seq = gen_luby(50)
    
    def luby_restart_limit():
        nonlocal luby_i
        if luby_i >= len(luby_seq):
            luby_i = len(luby_seq) - 1
        limit = restart_base * luby_seq[luby_i]
        luby_i += 1
        return limit

    restart_limit = luby_restart_limit()
    
    def var(lit):
        return abs(lit) - 1
        
    def sign(lit):
        return lit > 0
        
    def value(lit):
        v = var(lit)
        if assign[v] is None:
            return None
        return sign(lit) == assign[v]
        
    def set_value(v, val, lev, cl_idx):
        assign[v] = val
        level[v] = lev
        reason[v] = cl_idx
        lit = v + 1
        if val:
            trail.append(lit)
        else:
            trail.append(-lit)
            
    def propagate():
        nonlocal trail, watches, clauses_all, assign, level, reason
        trail_index = [-1] * new_n
        for i, lit in enumerate(trail):
            trail_index[var(lit)] = i
            
        while len(trail) > trail_lim[-1]:
            p = trail[-1]
            w_list = watches[-p]
            i = 0
            while i < len(w_list):
                c_idx = w_list[i]
                c = clauses_all[c_idx]
                
                if c[0] == -p:
                    other = c[1]
                    other_idx = 1
                else:
                    other = c[0]
                    other_idx = 0
                    
                val_other = value(other)
                
                if val_other is True:
                    i += 1
                    continue
                    
                elif val_other is False:
                    new_watcher = None
                    for k in range(2, len(c)):
                        if value(c[k]) is None:
                            new_watcher = c[k]
                            break
                            
                    if new_watcher is None:
                        return c_idx
                    else:
                        c[other_idx] = new_watcher
                        watches[new_watcher].append(c_idx)
                        w_list.pop(i)
                        continue
                else:
                    v = var(other)
                    set_value(v, sign(other), level[p], c_idx)
                    i += 1
                    continue
                    
        return -1
        
    def analyze(conflict_idx):
        nonlocal clauses_all, watches, activity, var_inc, assign, level, reason, trail, trail_lim
        
        trail_index = [-1] * new_n
        for i, lit in enumerate(trail):
            trail_index[var(lit)] = i
            
        current_level = level[var(clauses_all[conflict_idx][0])]
        
        learnt_set = set(clauses_all[conflict_idx])
        
        def count_at_level(lev):
            return sum(1 for lit in learnt_set if level[var(lit)] == lev)
            
        while count_at_level(current_level) > 1:
            best_lit = None
            best_idx = -1
            for lit in learnt_set:
                if level[var(lit)] == current_level:
                    v = var(lit)
                    if reason[v] is not None:
                        idx = trail_index[v]
                        if idx > best_idx:
                            best_idx = idx
                            best_lit = lit
                            
            if best_lit is None:
                break
                
            v = var(best_lit)
            r_idx = reason[v]
            for lit in clauses_all[r_idx]:
                if lit not in learnt_set:
                    learnt_set.add(lit)
            learnt_set.remove(best_lit)
            
        new_clause = list(learnt_set)
        
        levels_in_learnt = [level[var(lit)] for lit in learnt_set]
        levels_in_learnt.sort(reverse=True)
        
        if len(levels_in_learnt) >= 2:
            bt_level = levels_in_learnt[1]
        else:
            bt_level = 0
            
        clauses_all.append(new_clause)
        new_clause_idx = len(clauses_all) - 1
        
        if len(new_clause) >= 2:
            watches[new_clause[0]].append(new_clause_idx)
            watches[new_clause[1]].append(new_clause_idx)
            
        for lit in new_clause:
            v = var(lit)
            activity[v] += var_inc
            if activity[v] > 1e100:
                for i in range(new_n):
                    activity[i] /= 1e100
                var_inc /= 1e100
                
        return new_clause_idx, bt_level
        
    def backtrack(lev):
        nonlocal trail, trail_lim, assign, level, reason
        
        while len(trail) > trail_lim[lev]:
            lit = trail.pop()
            v = var(lit)
            assign[v] = None
            level[v] = 0
            reason[v] = None
            
        if len(trail_lim) > lev + 1:
            trail_lim = trail_lim[:lev+1]
            
    while time.time() < deadline:
        conflict = propagate()
        
        if conflict == -1:
            if len(trail) == new_n:
                break
            else:
                best_v = -1
                best_act = -1
                for v in range(new_n):
                    if assign[v] is None:
                        if activity[v] > best_act:
                            best_act = activity[v]
                            best_v = v
                            
                if best_v == -1:
                    break
                    
                val = phase[best_v]
                if random.random() < 0.5:
                    val = not val
                phase[best_v] = val
                
                trail_lim.append(len(trail))
                set_value(best_v, val, len(trail_lim)-1, -1)
                
                for v in range(new_n):
                    activity[v] *= decay
                var_inc /= decay
        else:
            conflicts += 1
            new_clause_idx, bt_level = analyze(conflict)
            backtrack(bt_level)
            
            if conflicts >= restart_limit:
                trail = []
                trail_lim = [0]
                assign = [None] * new_n
                level = [0] * new_n
                reason = [None] * new_n
                conflicts = 0
                restart_limit = luby_restart_limit()
                
    rev_map = {v: k for k, v in var_map.items()}
    final_assignment = [False] * n
    for i in range(n):
        if assignment[i] is not None:
            final_assignment[i] = assignment[i]
        else:
            new_v = rev_map[i]
            if assign[new_v] is not None:
                final_assignment[i] = assign[new_v]
            else:
                final_assignment[i] = random.random() < 0.5
                
    return final_assignment
