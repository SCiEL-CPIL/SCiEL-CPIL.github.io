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
    deadline = time.time() + 295.0
    pos, neg = _build_masks(clauses, n)
    pos = tuple(pos)
    neg = tuple(neg)

    # Build numpy arrays for vectorized satisfaction checking
    pos_arr = np.array(pos, dtype=np.uint64)
    neg_arr = np.array(neg, dtype=np.uint64)
    full_mask = (1 << n) - 1

    # Precompute clause literal index lists
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])

    # Build variable occurrence: for each var, list of (clause_idx, is_pos_in_clause)
    var_occ = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var, is_pos in cv:
            var_occ[var].append((ci, is_pos))

    def sat_count_numpy(assign_int):
        assign_np = np.uint64(assign_int)
        pos_satisfied = (assign_np & pos_arr) != 0
        neg_satisfied = ((~assign_np) & full_mask & neg_arr) != 0
        satisfied = pos_satisfied | neg_satisfied
        return int(np.sum(satisfied))

    def assign_to_int(assign):
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)
        return v

    def int_to_assign(v):
        return [bool(v & (1 << i)) for i in range(n)]

    def get_unsat_set(v):
        """Return set of unsatisfied clause indices."""
        assign_np = np.uint64(v)
        pos_satisfied = (assign_np & pos_arr) != 0
        neg_satisfied = ((~assign_np) & full_mask & neg_arr) != 0
        satisfied = pos_satisfied | neg_satisfied
        return set(np.where(~satisfied)[0].tolist())

    def update_unsat_after_flip(v, unsat_set, var):
        """After flipping var, update unsat_set incrementally."""
        # Check clauses containing var
        for ci, is_pos_in_clause in var_occ[var]:
            # Determine if this clause is now satisfied
            bit = 1 << var
            if is_pos_in_clause:
                # Clause has +var: satisfied if var is True (bit set in v)
                now_satisfied = (v & bit) != 0
            else:
                # Clause has -var: satisfied if var is False (bit not set in v)
                now_satisfied = (v & bit) == 0
            
            if now_satisfied:
                unsat_set.discard(ci)
            else:
                # Need to check if other literals in clause satisfy it
                # Quick check: is any other literal satisfied?
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

    def is_literal_satisfied(v, var, is_pos):
        """Check if a specific literal is satisfied by assignment v."""
        bit = 1 << var
        if is_pos:
            return (v & bit) != 0
        else:
            return (v & bit) == 0

    rng = random.Random(12345)
    best_assign = None
    best_sat = -1

    # Phase 0: quick random initializations
    for attempt_seed in range(8):
        r = random.Random(attempt_seed)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        s = sat_count_numpy(v)
        if s > best_sat:
            best_sat = s
            best_assign = assign
        if s == m:
            return assign
        if time.time() > deadline:
            return best_assign if best_assign else [False] * n

    # Phase 1: Multi-restart WalkSAT with Min-Conflicts (Max-Help) heuristic
    for restart in range(5000):
        if time.time() > deadline:
            break
        r = random.Random(restart)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        unsat_set = get_unsat_set(v)
        cur_sat = m - len(unsat_set)
        
        steps_budget = 8000
        for step in range(steps_budget):
            if time.time() > deadline:
                break
            
            if len(unsat_set) == 0:
                best_assign = int_to_assign(v)
                best_sat = m
                return best_assign
            
            # Select an unsatisfied clause
            # Heuristic: Prefer clauses with more "helping" literals (min-conflicts)
            # A literal is "helping" if flipping its variable would satisfy the clause.
            # Since the clause is unsatisfied, all its literals are currently falsified.
            # Wait, if clause is unsatisfied, NO literal is satisfied.
            # So flipping ANY literal in the clause will satisfy it.
            # The "Min-Conflicts" idea from Insp 1 was: pick clause where max literals are falsified?
            # In 3-SAT, if clause is unsat, all 3 literals are falsified. So all unsat clauses are "equal" in this sense.
            # However, Insp 1 sampled clauses and checked `help_cnt`.
            # Let's stick to a simpler, robust variant: Random clause, but smart variable selection.
            
            unsat_list = list(unsat_set)
            ci = unsat_list[r.randrange(len(unsat_list))]
            cv = clause_vars[ci]
            
            # Variable Selection Heuristic:
            # 1. Count how many unsatisfied clauses each variable appears in (with correct polarity to satisfy?)
            # Actually, standard WalkSAT: pick variable from clause.
            # Improved: Pick variable that appears in many OTHER unsatisfied clauses.
            # This is the "smooth surrogate" idea but simplified to just count occurrences in unsat set.
            
            best_var = None
            best_score = -1
            
            for var, is_pos in cv:
                score = 0.0
                # Count occurrences in unsat clauses
                for uci, uis_pos in var_occ[var]:
                    if uci in unsat_set:
                        # If this variable is in another unsat clause, flipping it might help or hurt.
                        # If it has the SAME polarity in the other clause, and that clause is unsat,
                        # then the variable is currently falsifying that literal too.
                        # Flipping it will satisfy the current clause, and potentially satisfy the other one.
                        if is_pos == uis_pos:
                            score += 1.0
                        else:
                            # Opposite polarity. If we flip to satisfy current (is_pos), 
                            # we might falsify the other one if it was satisfied? 
                            # But the other one is UNSAT, so its literal is already falsified.
                            # If we flip var, and polarities are opposite:
                            # Current wants var=True (is_pos=True). Other has var=False (uis_pos=False).
                            # If we set var=True, Current is sat. Other's literal (False) becomes False (still falsified).
                            # So it doesn't help the other one.
                            # Does it hurt? No, it was already unsat.
                            pass
                # Add noise
                score += r.random() * 0.5
                if score > best_score:
                    best_score = score
                    best_var = var
            
            if best_var is None:
                best_var = cv[r.randrange(len(cv))][0]
            
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
        if time.time() > deadline:
            break

    # Phase 2: Polish from best assignment with tighter local search
    if best_assign is not None and best_sat < m:
        v = assign_to_int(best_assign)
        unsat_set = get_unsat_set(v)
        polish_deadline = min(time.time() + 20.0, deadline)
        
        for polish_restart in range(200):
            if time.time() > polish_deadline:
                break
            
            v_cur = v
            unsat_cur = set(unsat_set)
            
            for pstep in range(3000):
                if time.time() > polish_deadline:
                    break
                
                if len(unsat_cur) == 0:
                    best_assign = int_to_assign(v_cur)
                    best_sat = m
                    return best_assign
                
                unsat_list = list(unsat_cur)
                ci = unsat_list[r.randrange(len(unsat_list))]
                cv = clause_vars[ci]
                
                # Greedy: pick variable that maximally reduces unsat count
                # We use the same heuristic as Phase 1 but with less noise
                best_var = None
                best_score = -1
                for var, is_pos in cv:
                    score = 0.0
                    for uci, uis_pos in var_occ[var]:
                        if uci in unsat_cur:
                            if is_pos == uis_pos:
                                score += 1.0
                    score += r.random() * 0.1 # Very low noise for polish
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
