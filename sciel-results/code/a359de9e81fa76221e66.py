# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def _build_masks(clauses, n):
    pos = []
    neg = []
    for c in clauses:
        p = 0
        q = 0
        for l in c:
            if l > 0:
                p |= (1 << (l - 1))
            else:
                q |= (1 << (-l - 1))
        pos.append(p)
        neg.append(q)
    return pos, neg


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0
    
    # Precompute clause structures
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])
    
    # Build variable occurrence lists: var -> list of (clause_idx, is_pos_in_clause)
    var_occ = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var, is_pos in cv:
            var_occ[var].append((ci, is_pos))
    
    # For fast satisfaction checking: precompute for each clause its literal masks
    # and also a list of (var, is_pos) for each clause
    
    # Build numpy arrays for vectorized satisfaction checking
    pos_arr = np.array([0] * m, dtype=np.uint64)
    neg_arr = np.array([0] * m, dtype=np.uint64)
    for ci, cv in enumerate(clause_vars):
        p = 0
        q = 0
        for var, is_pos in cv:
            if is_pos:
                p |= (1 << var)
            else:
                q |= (1 << var)
        pos_arr[ci] = p
        neg_arr[ci] = q
    
    full_mask = (1 << n) - 1
    
    def assign_to_int(assign):
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)
        return v
    
    def int_to_assign(v):
        return [bool(v & (1 << i)) for i in range(n)]
    
    def get_unsat_set(v):
        """Return set of unsatisfied clause indices using numpy."""
        assign_np = np.uint64(v)
        pos_satisfied = (assign_np & pos_arr) != 0
        neg_satisfied = ((~assign_np) & full_mask & neg_arr) != 0
        satisfied = pos_satisfied | neg_satisfied
        return set(np.where(~satisfied)[0].tolist())
    
    def update_unsat_after_flip(v, unsat_set, var):
        """After flipping var, update unsat_set incrementally."""
        bit = 1 << var
        for ci, is_pos_in_clause in var_occ[var]:
            if is_pos_in_clause:
                now_satisfied = (v & bit) != 0
            else:
                now_satisfied = (v & bit) == 0
            
            if now_satisfied:
                unsat_set.discard(ci)
            else:
                # Check if any other literal in the clause is satisfied
                clause = clauses[ci]
                other_satisfied = False
                for l in clause:
                    if abs(l) == var + 1:
                        continue
                    if l > 0:
                        if (v & (1 << (l - 1))) != 0:
                            other_satisfied = True
                            break
                    else:
                        if (v & (1 << (-l - 1))) == 0:
                            other_satisfied = True
                            break
                if not other_satisfied:
                    unsat_set.add(ci)
                else:
                    unsat_set.discard(ci)
        return unsat_set
    
    # Precompute for each variable and each clause containing it,
    # whether flipping the variable would satisfy that clause given current assignment
    # This is dynamic, so we compute on the fly but cache the polarity match counts
    
    # Build a lookup: for each var, for each clause it appears in, store the polarity
    # var_polarity[var][ci] = True if var appears positively in clause ci
    
    var_polarity = {}
    for var in range(n):
        var_polarity[var] = {}
        for ci, is_pos in var_occ[var]:
            var_polarity[var][ci] = is_pos
    
    rng = random.Random(42)
    best_assign = None
    best_sat = -1
    
    # Phase 0: Quick random initializations to find a good starting point
    for attempt in range(16):
        if time.time() > deadline:
            break
        r = random.Random(attempt * 7919)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        unsat_set = get_unsat_set(v)
        s = m - len(unsat_set)
        if s > best_sat:
            best_sat = s
            best_assign = assign
        if s == m:
            return assign
    
    # Phase 1: Multi-restart WalkSAT with improved heuristics
    # Key improvements:
    # 1. Better variable selection: count how many unsat clauses would be satisfied
    #    by flipping each candidate variable (using polarity matching)
    # 2. Adaptive noise
    # 3. More aggressive restarts
    
    for restart in range(20000):
        if time.time() > deadline:
            break
        
        r = random.Random(restart * 104729 + 1)
        
        # Start from best assignment found so far, or random
        if best_assign is not None and r.random() < 0.3:
            v = assign_to_int(best_assign)
        else:
            assign = [r.random() < 0.5 for _ in range(n)]
            v = assign_to_int(assign)
        
        unsat_set = get_unsat_set(v)
        cur_sat = m - len(unsat_set)
        
        # Adaptive step budget: more steps when closer to solution
        base_steps = 5000
        if cur_sat > m * 0.9:
            steps_budget = 15000
        elif cur_sat > m * 0.8:
            steps_budget = 10000
        else:
            steps_budget = base_steps
        
        # Adaptive noise parameter
        noise_base = 0.3
        
        for step in range(steps_budget):
            if time.time() > deadline:
                break
            
            if len(unsat_set) == 0:
                return int_to_assign(v)
            
            # Select an unsatisfied clause
            # Weight by "difficulty": clauses with fewer satisfied other literals are harder
            # But all unsat clauses have 0 satisfied literals by definition
            # So just pick randomly, or weight by clause frequency
            unsat_list = list(unsat_set)
            ci = unsat_list[r.randrange(len(unsat_list))]
            cv = clause_vars[ci]
            
            # Variable Selection: For each variable in the clause, score based on
            # how many OTHER unsatisfied clauses it appears in with matching polarity
            best_var = None
            best_score = -1
            
            for var, is_pos in cv:
                score = 0.0
                # Count occurrences in other unsat clauses with same polarity
                for uci, uis_pos in var_occ[var]:
                    if uci in unsat_set and uci != ci:
                        if is_pos == uis_pos:
                            score += 1.0
                        else:
                            # Opposite polarity: flipping might hurt
                            score -= 0.3
                # Add noise
                score += r.random() * noise_base
                if score > best_score:
                    best_score = score
                    best_var = var
            
            if best_var is None:
                best_var = cv[r.randrange(len(cv))][0]
            
            # Flip the variable
            bit = 1 << best_var
            v ^= bit
            unsat_set = update_unsat_after_flip(v, unsat_set, best_var)
            cur_sat = m - len(unsat_set)
            
            # Track best
            if cur_sat > best_sat:
                best_sat = cur_sat
                best_assign = int_to_assign(v)
            
            if best_sat == m:
                return best_assign
        
        # Update best after restart
        if cur_sat > best_sat:
            best_sat = cur_sat
            best_assign = int_to_assign(v)
        
        if best_sat == m:
            return best_assign
    
    # Phase 2: Final polish with very tight local search
    if best_assign is not None and best_sat < m:
        v = assign_to_int(best_assign)
        unsat_set = get_unsat_set(v)
        polish_deadline = min(time.time() + 30.0, deadline)
        
        for polish_restart in range(500):
            if time.time() > polish_deadline:
                break
            
            v_cur = v
            unsat_cur = set(unsat_set)
            
            for pstep in range(10000):
                if time.time() > polish_deadline:
                    break
                
                if len(unsat_cur) == 0:
                    return int_to_assign(v_cur)
                
                unsat_list = list(unsat_cur)
                ci = unsat_list[r.randrange(len(unsat_list))]
                cv = clause_vars[ci]
                
                # Greedy with minimal noise
                best_var = None
                best_score = -1
                for var, is_pos in cv:
                    score = 0.0
                    for uci, uis_pos in var_occ[var]:
                        if uci in unsat_cur and uci != ci:
                            if is_pos == uis_pos:
                                score += 1.0
                            else:
                                score -= 0.3
                    score += r.random() * 0.05
                    if score > best_score:
                        best_score = score
                        best_var = var
                
                if best_var is None:
                    best_var = cv[r.randrange(len(cv))][0]
                
                v_cur ^= (1 << best_var)
                unsat_cur = update_unsat_after_flip(v_cur, unsat_cur, best_var)
            
            s = m - len(unsat_cur)
            if s > best_sat:
                best_sat = s
                best_assign = int_to_assign(v_cur)
                v = v_cur
                unsat_set = unsat_cur
            if best_sat == m:
                return best_assign
    
    if best_assign is None:
        return [False] * n
    return best_assign
# EVOLVE-BLOCK-END
