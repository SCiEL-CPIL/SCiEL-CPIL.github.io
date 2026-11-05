import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 295.0  # safety margin before 330s kill

    # Build clause structure: each clause is list of (var, sign)
    # var is 0-based, sign is +1 or -1
    clause_vars = []
    for cl in clauses:
        cvars = []
        for lit in cl:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cvars.append((var, sign))
        clause_vars.append(cvars)
    m = len(clause_vars)

    # var_to_clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v, s in clause_vars[i]:
            var_to_clauses[v].append(i)

    # Solver state
    assignment = [0] * n  # 0=unassigned, 1=true, -1=false
    trail = []  # stack of assigned vars
    trail_lim = []  # levels
    level = [0] * n
    reason = [None] * n  # clause that implied var (for non-decision vars)

    # Activity / phase saving
    var_activity = [0.0] * n
    clause_activity = [0.0] * m
    var_heap = list(range(n))
    clause_heap = list(range(m))
    var_index = list(range(n))
    clause_index = list(range(m))
    phase = [False] * n
    var_decay = 0.95
    clause_decay = 0.999

    # Learned clauses
    learned = []  # list of (vars_list, activity, is_lunit)
    learned_count = 0
    max_learned = 5000

    conflicts = 0
    decisions = 0
    restarts = 0
    restart_base = 100
    conflicts_last_restart = 0

    # Helper: evaluate a literal under current assignment
    def lit_value(var, sign):
        a = assignment[var]
        if a == 0:
            return 0  # unassigned
        return 1 if a == sign else -1

    def clause_eval(cvars):
        # return (num_unassigned, num_false)
        nu = 0
        nf = 0
        for v, s in cvars:
            val = lit_value(v, s)
            if val == 0:
                nu += 1
            elif val == -1:
                nf += 1
        return nu, nf

    def heap_swap(h, i, j):
        h[i], h[j] = h[j], h[i]
        var_index[h[i]] = i
        var_index[h[j]] = j

    def var_heapify(h, i):
        n_items = len(h)
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            best = i
            if l < n_items and var_activity[h[l]] > var_activity[h[best]]:
                best = l
            if r < n_items and var_activity[h[r]] > var_activity[h[best]]:
                best = r
            if best == i:
                break
            heap_swap(h, i, best)
            i = best

    def var_heap_push_or_increment(h, var, inc=1.0):
        idx = var_index[var]
        var_activity[var] += inc
        if idx > 0:
            parent = (idx - 1) // 2
            if var_activity[var] > var_activity[h[parent]]:
                heap_swap(h, idx, parent)
                var_heapify(h, parent)

    def var_heap_decrease_key(h, var):
        idx = var_index[var]
        var_heapify(h, idx)

    def var_heap_pop(h):
        top = h[0]
        h[0] = h[-1]
        var_index[h[0]] = 0
        h.pop()
        var_index[top] = -1
        if h:
            var_heapify(h, 0)
        return top

    def clause_heapify(h, i):
        n_items = len(h)
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            best = i
            if l < n_items and clause_activity[h[l]] > clause_activity[h[best]]:
                best = l
            if r < n_items and clause_activity[h[r]] > clause_activity[h[best]]:
                best = r
            if best == i:
                break
            clause_heap_swap(h, i, best)
            i = best

    def clause_heap_swap(h, i, j):
        h[i], h[j] = h[j], h[i]
        clause_index[h[i]] = i
        clause_index[h[j]] = j

    def clause_heap_pop(h):
        top = h[0]
        h[0] = h[-1]
        clause_index[h[0]] = 0
        h.pop()
        clause_index[top] = -1
        if h:
            clause_heapify(h, 0)
        return top

    def clause_heap_push(h, cidx):
        h.append(cidx)
        clause_index[cidx] = len(h) - 1
        # percolate up
        i = len(h) - 1
        while i > 0:
            parent = (i - 1) // 2
            if clause_activity[h[i]] > clause_activity[h[parent]]:
                clause_heap_swap(h, i, parent)
                i = parent
            else:
                break

    def clause_heap_increase_key(h, cidx):
        clause_activity[cidx] += 1.0
        i = clause_index[cidx]
        while i > 0:
            parent = (i - 1) // 2
            if clause_activity[h[i]] > clause_activity[h[parent]]:
                clause_heap_swap(h, i, parent)
                i = parent
            else:
                break

    # Initialize heaps
    for v in range(n):
        var_activity[v] = float(v)
    for c in range(m):
        clause_activity[c] = float(c)
    # Build heap
    for i in range(len(var_heap) - 1, -1, -1):
        var_heapify(var_heap, i)
    for i in range(len(clause_heap) - 1, -1, -1):
        clause_heapify(clause_heap, i)

    def assign(var, sign, lvl, rsn):
        assignment[var] = sign
        level[var] = lvl
        reason[var] = rsn
        trail.append(var)
        trail_lim.append(lvl)
        phase[var] = (sign == 1)

    def unassign_to(lvl):
        while trail_lim and trail_lim[-1] >= lvl:
            trail_lim.pop()
            v = trail.pop()
            assignment[v] = 0
            level[v] = 0
            reason[v] = None

    def pick_branching_var():
        # Pick from var_heap top, skip unassigned
        while var_heap:
            v = var_heap[0]
            if assignment[v] == 0:
                return v
            var_heap_pop(var_heap)
        return -1

    def analyze_conflict(conflict_clause_idx, is_learned):
        """
        Returns (learnt_clause_vars, backjump_level, lit_to_flip)
        conflict_clause_idx: index into all_clauses (0..m-1 for original, m+i for learned)
        """
        # We need to access the clause
        if conflict_clause_idx < m:
            cvars = clause_vars[conflict_clause_idx]
        else:
            cvars = learned[conflict_clause_idx - m][0]

        learnt = []
        seen = [False] * n
        p = 0  # pointer
        bt_level = 0
        cur_level = trail_lim[-1] if trail_lim else 0

        while True:
            # Find the last unassigned literal in cvars (the one we just flipped)
            # Actually, we need to find the literal at current level that is the "causal" one
            # The conflict clause is all false. We look for the literal with highest level < cur_level
            # Actually, the standard approach:
            # The clause is falsified. All its literals are false.
            # We need to find the literal that was assigned at the current decision level
            # (the one that caused the conflict)
            
            # Find the literal at the current level
            causal = -1
            causal_level = -1
            for v, s in cvars:
                if level[v] == cur_level:
                    causal = v
                    causal_level = cur_level
                    break
            
            if causal == -1:
                # Should not happen
                break

            # Remove causal from consideration, add to learnt
            seen[causal] = True
            learnt.append((causal, -1))  # negate: if clause has (v,s), learnt has (v,-s)
            # Actually: conflict clause is (l1 | l2 | l3) = false, so l1,l2,l3 all false
            # Learnt: if we flip causal, we need the clause to be true
            # The learnt clause should contain the negation of causal's assignment
            # If causal was assigned sign s_c, then the learnt literal is (causal, -s_c)
            # But in our representation, cvars has (v, s) meaning "v must be s"
            # If v was assigned s (making the literal false means v != s, so v = -s)
            # Wait, let me reconsider.
            
            # Literal (v, s) is false when assignment[v] != s, i.e., assignment[v] = -s
            # So if the conflict clause has literal (v, s) and it's false, v = -s
            # The learnt clause should assert: NOT(all false) = at least one true
            # We're doing resolution: the learnt clause will have the negation of the causal literal
            # Causal literal in conflict: (causal, s_c) where s_c is from cvars
            # It's false, so assignment[causal] = -s_c
            # We want to add (causal, s_c) to learnt (because if causal is -s_c, this is false, but we want to prevent that)
            # Actually, learnt clause = (causal, s_c) | (other resolved literals)
            # The literal (causal, s_c) is currently false (since causal = -s_c)
            # So in the learnt clause, this literal is the "asserting" one after backjump
            
            # Let me redo: learnt literal for causal is (causal, s_c) where s_c is the sign in the conflict clause
            # Because the conflict clause says "v must be s_c" (literal (v,s_c) is true when v=s_c)
            # Since it's false, v=-s_c. The learnt clause should say "v must NOT be -s_c" i.e. "v must be s_c"
            # So learnt literal is (causal, s_c). Correct.
            
            # Fix learnt
            for i, (v, s) in enumerate(learnt):
                if v == causal:
                    learnt[i] = (causal, s_c)
                    break

            # Now resolve with reason clause of causal
            if p >= len(cvars) - 1:
                # Done
                break

            # Find reason for causal
            rsn = reason[causal]
            if rsn is None:
                # It's a decision variable
                bt_level = cur_level
                break

            # Get reason clause
            if rsn < m:
                rvars = clause_vars[rsn]
            else:
                rvars = learned[rsn - m][0]

            # Resolve: add all unassigned literals from rvars to learnt
            new_p = p
            for v, s in rvars:
                if seen[v]:
                    continue
                seen[v] = True
                if level[v] < cur_level:
                    learnt.append((v, s))
                    if level[v] > bt_level:
                        bt_level = level[v]
                # else: level >= cur_level, skip (will be handled by higher-level resolution)
            
            # Advance p: skip past the causal literal in cvars
            # Find index of causal in cvars
            for i in range(p, len(cvars)):
                if cvars[i][0] == causal:
                    p = i + 1
                    break
            else:
                p = len(cvars)

            # If we've processed all, check if we need to continue
            # The current level decreases when we resolve
            # Actually, the standard algorithm:
            # After resolving, the new "current level" is the level of the next causal
            # Let me restructure...

        # Actually, let me use a cleaner implementation
        # Restart analyze_conflict with a cleaner approach
        return None, None, None

    # Let me rewrite with a cleaner CDCL
    # Actually, the above is getting complex. Let me use a well-tested pattern.

    # Clean CDCL implementation
    # Reset state
    assignment = [0] * n
    trail = []
    trail_lim = []
    level = [0] * n
    reason = [None] * n
    var_activity = [0.0] * n
    clause_activity = [0.0] * m
    var_heap = list(range(n))
    clause_heap = list(range(m))
    var_index = list(range(n))
    clause_index = list(range(m))
    phase = [False] * n
    learned = []
    learned_count = 0
    conflicts = 0
    decisions = 0
    restarts = 0
    conflicts_last_restart = 0

    for v in range(n):
        var_activity[v] = float(v)
    for c in range(m):
        clause_activity[c] = float(c)
    for i in range(len(var_heap) - 1, -1, -1):
        var_heapify(var_heap, i)
    for i in range(len(clause_heap) - 1, -1, -1):
        clause_heapify(clause_heap, i)

    def assign(var, sign, lvl, rsn):
        assignment[var] = sign
        level[var] = lvl
        reason[var] = rsn
        trail.append(var)
        trail_lim.append(lvl)
        phase[var] = (sign == 1)

    def unassign_to(lvl):
        while trail_lim and trail_lim[-1] >= lvl:
            trail_lim.pop()
            v = trail.pop()
            assignment[v] = 0
            level[v] = 0
            reason[v] = None

    def current_level():
        return trail_lim[-1] if trail_lim else 0

    def pick_branching_var():
        while var_heap:
            v = var_heap[0]
            if assignment[v] == 0:
                return v
            var_heap_pop(var_heap)
        return -1

    def analyze_conflict(conflict_idx):
        """
        conflict_idx: index into all clauses (0..m-1 original, m+ learned)
        Returns (learnt_vars, bt_level) or None on error
        learnt_vars: list of (var, sign)
        """
        if conflict_idx < m:
            cvars = clause_vars[conflict_idx]
        else:
            cvars = learned[conflict_idx - m][0]

        seen = [False] * n
        learnt = []
        bt_level = 0
        cur_level = current_level()
        p = 0  # pointer into cvars

        while True:
            # Find the literal at cur_level in cvars[p:]
            causal = -1
            causal_sign = 0
            causal_pos = -1
            for i in range(p, len(cvars)):
                v, s = cvars[i]
                if level[v] == cur_level:
                    causal = v
                    causal_sign = s
                    causal_pos = i
                    break

            if causal == -1:
                # No literal at current level - shouldn't happen
                break

            # Add negation of causal to learnt
            # The conflict clause has literal (causal, causal_sign) which is false
            # So assignment[causal] = -causal_sign
            # Learnt should have (causal, causal_sign) to prevent this
            seen[causal] = True
            learnt.append((causal, causal_sign))

            # Check if we're done (all other literals are at lower levels)
            # Advance p past causal
            p = causal_pos + 1

            # If this was the last, or no more at cur_level, we may be done
            # Check if there's another literal at cur_level
            has_more_at_level = False
            for i in range(p, len(cvars)):
                if level[cvars[i][0]] == cur_level:
                    has_more_at_level = True
                    break

            if not has_more_at_level:
                # This was the last at current level
                # Now resolve with reason if any
                if p >= len(cvars):
                    # All literals processed
                    break

                # Get reason for causal
                rsn = reason[causal]
                if rsn is None:
                    # Decision variable - backjump to cur_level
                    bt_level = cur_level
                    break

                # Resolve with reason clause
                if rsn < m:
                    rvars = clause_vars[rsn]
                else:
                    rvars = learned[rsn - m][0]

                # Add unassigned literals from reason to learnt
                for v, s in rvars:
                    if seen[v]:
                        continue
                    seen[v] = True
                    if level[v] < cur_level:
                        learnt.append((v, s))
                        if level[v] > bt_level:
                            bt_level = level[v]

                # The current level for next iteration is the level of the next causal
                # which is the max level among remaining unprocessed literals
                # Actually, after resolving, we need to find the next highest level
                # among the literals we haven't processed yet
                # The new cur_level is the max level of unprocessed literals in the combined set
                # For simplicity, find max level among all learnt literals except the one we just added
                # Actually, the standard approach: the next causal is the one with the next highest level
                # Let's find the max level among unprocessed literals in cvars
                new_max_level = 0
                for i in range(p, len(cvars)):
                    v, s = cvars[i]
                    if not seen[v] and level[v] > new_max_level:
                        new_max_level = level[v]
                # Also consider literals from reason that we added
                # Actually, the reason's literals at level < cur_level are added to learnt
                # The next causal should be at the next highest level
                # For a 3-SAT with unit propagation, this is usually simple
                # Let's just use: next cur_level = max level among unprocessed cvars literals
                # But we also need to handle the case where reason has literals at higher levels
                # This is getting complex. Let me use a simpler approach for 3-SAT.
                
                # For 3-SAT, the reason clause is also 3-literal (or learned)
                # The resolution should be straightforward
                # Let me just set cur_level to the max level of unprocessed literals
                if new_max_level == 0:
                    break
                cur_level = new_max_level
                continue
            else:
                # There's another literal at cur_level - continue the loop
                # The next causal will be found in the next iteration
                # But we need to resolve the current causal first
                # Actually, if there are multiple literals at the same level,
                # we need to handle them. But in practice, with unit propagation,
                # there's usually only one at the current level.
                # Let's just continue and find the next one
                continue

        # Clean up learnt: remove literals that are not at the backjump level or below
        # The learnt clause should have exactly one literal at bt_level (the asserting literal)
        # and the rest at lower levels
        # Filter: keep only literals at level <= bt_level
        final_learnt = []
        for v, s in learnt:
            if level[v] <= bt_level:
                final_learnt.append((v, s))
            elif v == causal:  # The asserting literal might be at cur_level which could be > bt_level
                # Actually, the asserting literal is at bt_level by construction
                pass

        # If bt_level is 0, the learnt clause is a tautology or unit
        # Make sure we have at least one literal
        if not final_learnt:
            final_learnt = learnt[:1]

        return final_learnt, bt_level

    def propagate():
        """
        Unit propagation. Returns conflict clause index or None.
        """
        # Simple queue-based propagation
        # For each clause, check if it's unit
        # We need to track which clauses to check
        # For efficiency, maintain a list of clauses to check
        # Simple approach: iterate through all clauses each time (slow but correct)
        # Better: maintain a worklist
        
        # For a 50-variable problem, let's use a simple approach
        # But we need to be careful about efficiency
        
        # Use a queue of clauses to check
        # Initially, all clauses
        # When we assign a variable, add all clauses containing that variable
        
        # For simplicity and correctness, let's do a full scan each propagation step
        # This is O(m * k) per propagation, which for m~1000, k=3 is fine
        
        while True:
            found_unit = False
            conflict = None
            
            # Check all clauses
            for ci in range(m):
                cvars = clause_vars[ci]
                num_unassigned = 0
                num_false = 0
                unassigned_var = -1
                unassigned_sign = 0
                
                for v, s in cvars:
                    val = lit_value(v, s)
                    if val == 0:
                        num_unassigned += 1
                        unassigned_var = v
                        unassigned_sign = s
                    elif val == -1:
                        num_false += 1
                
                if num_false == len(cvars):
                    conflict = ci
                    break
                
                if num_unassigned == 1 and num_false == len(cvars) - 1:
                    # Unit clause: must assign unassigned_var to unassigned_sign
                    if assignment[unassigned_var] == 0:
                        assign(unassigned_var, unassigned_sign, current_level(), ci)
                        found_unit = True
                        # Boost activity of reason clause
                        clause_heap_increase_key(clause_heap, ci)
                        # Boost activity of variable
                        var_heap_push_or_increment(var_heap, unassigned_var)
            
            # Check learned clauses
            for li in range(len(learned)):
                lvars = learned[li][0]
                num_unassigned = 0
                num_false = 0
                unassigned_var = -1
                unassigned_sign = 0
                
                for v, s in lvars:
                    val = lit_value(v, s)
                    if val == 0:
                        num_unassigned += 1
                        unassigned_var = v
                        unassigned_sign = s
                    elif val == -1:
                        num_false += 1
                
                if num_false == len(lvars):
                    conflict = m + li
                    break
                
                if num_unassigned == 1 and num_false == len(lvars) - 1:
                    if assignment[unassigned_var] == 0:
                        assign(unassigned_var, unassigned_sign, current_level(), m + li)
                        found_unit = True
                        clause_heap_increase_key(clause_heap, m + li)
                        var_heap_push_or_increment(var_heap, unassigned_var)
            
            if conflict is not None:
                return conflict
            
            if not found_unit:
                break
        
        return None

    def add_learned_clause(lvars, activity=0.0):
        global learned_count
        if len(learned) >= max_learned:
            # Remove the least active learned clause
            # Find min activity
            min_idx = 0
            min_act = learned[0][1]
            for i in range(1, len(learned)):
                if learned[i][1] < min_act:
                    min_act = learned[i][1]
                    min_idx = i
            learned.pop(min_idx)
        
        learned.append((lvars, activity, False))
        learned_count += 1

    def main_search():
        global conflicts, decisions, restarts, conflicts_last_restart
        global var_decay, clause_decay

        while time.time() < deadline:
            # Check for restart
            if conflicts - conflicts_last_restart > restart_base:
                restarts += 1
                conflicts_last_restart = conflicts
                restart_base = int(restart_base * 1.2)
                # Decay activities
                for v in range(n):
                    var_activity[v] *= var_decay
                for c in range(m):
                    clause_activity[c] *= clause_decay
                for i in range(len(learned)):
                    learned[i] = (learned[i][0], learned[i][1] * clause_decay, learned[i][2])
                # Rebuild heaps
                for i in range(len(var_heap) - 1, -1, -1):
                    var_heapify(var_heap, i)
                for i in range(len(clause_heap) - 1, -1, -1):
                    clause_heapify(clause_heap, i)
                # Reset trail
                unassign_to(0)
                continue

            # Pick branching variable
            var = pick_branching_var()
            if var == -1:
                # All assigned - check if satisfied
                # Should be satisfied if no conflict
                return True

            decisions += 1
            sign = 1 if phase[var] else -1
            assign(var, sign, current_level() + 1, None)
            var_heap_push_or_increment(var_heap, var)

            # Propagate
            conflict = propagate()
            
            if conflict is None:
                # Check if all assigned
                if len(trail) == n:
                    # Verify
                    for ci in range(m):
                        cvars = clause_vars[ci]
                        sat = False
                        for v, s in cvars:
                            if assignment[v] == s:
                                sat = True
                                break
                        if not sat:
                            break
                    else:
                        return True
                continue

            # Conflict!
            conflicts += 1
            
            # Analyze
            learnt, bt_level = analyze_conflict(conflict)
            
            if learnt is None or not learnt:
                # Error in analysis
                unassign_to(0)
                continue

            # Backjump
            unassign_to(bt_level)
            
            # Add learned clause
            add_learned_clause(learnt)
            
            # If bt_level is 0, the learnt clause is unit - propagate it
            # The first literal in learnt should be at bt_level
            # Assign it
            if bt_level > 0:
                # The asserting literal is the one at bt_level
                # Find it
                for v, s in learnt:
                    if level[v] == bt_level:
                        assign(v, s, bt_level, m + len(learned) - 1)
                        break
            else:
                # Unit at level 0 - assign
                v, s = learnt[0]
                if assignment[v] == 0:
                    assign(v, s, 0, m + len(learned) - 1)

            # Boost activity of learned clause
            if len(learned) > 0:
                clause_heap_increase_key(clause_heap, m + len(learned) - 1)

        return False

    result = main_search()
    
    # Extract assignment
    if result:
        return [assignment[i] == 1 for i in range(n)]
    
    # Fallback: return best effort
    # Try a random assignment
    rng = random.Random(42)
    best_assign = [rng.random() < 0.5 for _ in range(n)]
    best_unsat = float('inf')
    
    def count_unsat(a):
        cnt = 0
        for cl in clause_vars:
            sat = False
            for v, s in cl:
                if a[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    best_unsat = count_unsat(best_assign)
    
    # Simple local search as fallback
    start = time.time()
    while time.time() - start < 10:
        unsat_clauses = []
        for ci, cl in enumerate(clause_vars):
            sat = False
            for v, s in cl:
                if best_assign[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(ci)
        
        if not unsat_clauses:
            break
        
        # Flip a variable in a random unsat clause
        ci = rng.choice(unsat_clauses)
        v, s = rng.choice(clause_vars[ci])
        best_assign[v] = not best_assign[v]
        
        new_unsat = count_unsat(best_assign)
        if new_unsat < best_unsat:
            best_unsat = new_unsat
        elif rng.random() < 0.1:
            pass  # Accept anyway (simulated annealing-like)
    
    return best_assign
