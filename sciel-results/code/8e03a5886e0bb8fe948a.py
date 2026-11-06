import time
import random
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Time limit: 300s. We should stop well before that to be safe.
    # Let's use 280s as the hard limit for the solver logic.
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Data structures
    # assignment: 0 = UNDEF, 1 = True, -1 = False
    assignment = [0] * (n + 1)
    
    # Watches: watches[v][0] is list of clause indices watching positive literal v
    #           watches[v][1] is list of clause indices watching negative literal -v
    # We store clause indices, not the clauses themselves, to save memory and allow learning.
    watches = [[[], []] for _ in range(n + 1)]
    
    # Store clauses as lists of literals
    # We also need a way to handle learned clauses.
    # Let's maintain a list `all_clauses` where index 0..m-1 are original, m.. are learned.
    # But watches need to be updated when we add learned clauses.
    # Alternatively, we can just store clauses in a list and update watches for learned ones.
    
    # Initialize watches for original clauses
    # To speed up, we can just iterate through clauses and add to watches.
    # For each clause, pick two literals to watch. Usually the last two.
    # If a clause has only 1 literal, watch it. If 0, unsat.
    
    all_clauses = [list(c) for c in clauses]
    
    for i, clause in enumerate(all_clauses):
        if len(clause) == 0:
            # Unsat instance, but task says satisfiable. Just in case.
            return [False] * n
        # Pick two literals to watch. Let's pick the first and last for simplicity, 
        # or just last two. It doesn't matter much for correctness, only performance.
        # Let's pick the last two.
        if len(clause) >= 2:
            l1 = clause[-2]
            l2 = clause[-1]
            v1 = abs(l1)
            v2 = abs(l2)
            # Watch positive if literal > 0, negative if < 0
            # watches[v][0] for positive, watches[v][1] for negative
            watches[v1][0 if l1 > 0 else 1].append(i)
            watches[v2][0 if l2 > 0 else 1].append(i)
        elif len(clause) == 1:
            l1 = clause[0]
            v1 = abs(l1)
            watches[v1][0 if l1 > 0 else 1].append(i)

    # VSIDS Activity
    activity = [0.0] * (n + 1)
    var_inc = 1.0
    var_decay = 0.95
    
    # Trail: list of (variable, assignment)
    trail = []
    trail_lim = [] # list of trail lengths at each decision level
    
    # Reason: reason[v] = clause index that forced v, or -1 if decision
    reason = [-1] * (n + 1)
    
    # Clause analysis data
    # We need to be able to undo assignments.
    
    def assign_var(v, value, clause_id, level):
        assignment[v] = value
        trail.append(v)
        reason[v] = clause_id
        # Update watches? No, watches are updated during propagation.
        # But we need to check if this assignment satisfies any watched clauses.
        # Actually, the standard watched literals scheme handles this in the propagation loop.
        # When we assign a variable, we check the clauses watching the *negated* literal.
        # If the clause becomes satisfied (because we just assigned the positive literal), 
        # we can remove the clause from the watch list of the *other* literal if it's still unassigned.
        
    def unassign_var(v):
        assignment[v] = 0
        reason[v] = -1
        # Remove from trail? We pop from trail in backtracking.

    def propagate():
        # Returns conflict clause index or -1 if no conflict
        while trail_lim and len(trail) > trail_lim[-1]:
            # Get the most recent assignment
            v = trail[-1]
            trail.pop()
            # v was just assigned. 
            # Check clauses watching the opposite literal.
            # If v was assigned True (1), check clauses watching -v (negative).
            # If v was assigned False (-1), check clauses watching v (positive).
            
            # Wait, the standard algorithm is:
            # When v is assigned, we look at the list of clauses watching the literal that is now FALSE.
            # Let lit = v if assignment[v] == -1 else -v.
            # These are the clauses that might be in trouble.
            
            if assignment[v] == 1:
                # v is True. Clauses watching -v (negative) are in trouble.
                watch_list = watches[v][1]
            else:
                # v is False. Clauses watching v (positive) are in trouble.
                watch_list = watches[v][0]
            
            # We need to iterate through the watch list and potentially modify it.
            # We'll build a new list or modify in place.
            # To be safe and efficient, we can use a loop with index.
            
            i = 0
            while i < len(watch_list):
                c_idx = watch_list[i]
                clause = all_clauses[c_idx]
                
                # Check if the clause is already satisfied by some other literal.
                # We know the watched literal (the one corresponding to this watch list) is now false.
                # Let's find the other watched literal.
                # The clause has at least 2 literals (or 1, but if 1 and it's the one assigned, it's conflict).
                
                # Identify the two watched literals in the clause.
                # We don't store which literals are watched explicitly. 
                # We can just check if any other literal in the clause is true.
                # If yes, the clause is satisfied. We can swap the watched literal to this true one? 
                # No, we just keep the watch if the other literal is still unassigned.
                # If the other literal is assigned false, we need to find a new watch.
                
                # Let's check if the clause is satisfied.
                sat = False
                other_watch = None
                for l in clause:
                    if l == v or l == -v:
                        # This is the literal that became false. Skip.
                        continue
                    val = assignment[abs(l)]
                    if val == 1 and l > 0:
                        sat = True
                        break
                    elif val == -1 and l < 0:
                        sat = True
                        break
                    elif val == 0:
                        other_watch = l
                        # We found an unassigned literal. This can be the new watch.
                        # But we should check if there's another assigned one first?
                        # Actually, if we find an unassigned one, we can use it as a watch.
                        # We don't need to check further if we just want to find *a* new watch.
                        # But we need to make sure the clause isn't already satisfied.
                        # If we find a true literal, sat=True.
                        # If we find an unassigned literal, we can watch it.
                        # If we find a false literal, we continue.
                
                if sat:
                    # Clause is satisfied. Remove from this watch list? 
                    # No, we can leave it, but it's better to remove it to keep lists small.
                    # Actually, if it's satisfied, we don't need to watch it anymore.
                    # But we need to add it to the watch list of the satisfied literal.
                    # For simplicity, let's just remove it from the current watch list.
                    # We can add it back later if needed, but for SAT, once satisfied, it stays satisfied.
                    # So we can just remove it.
                    # However, we must be careful: if we remove it, we need to add it to the watch list of the satisfied literal.
                    # Let's find the satisfied literal.
                    # We broke out of the loop without storing it. Let's re-find or store.
                    # To avoid re-finding, let's modify the loop above to store the satisfied literal.
                    pass # This logic is getting complicated. Let's use a simpler approach.
                
                # Simpler Approach for Propagation:
                # 1. Get watch list for the literal that became false.
                # 2. For each clause in the list:
                #    a. If clause is satisfied by any other literal, remove from this list. (Optional, but good)
                #    b. Else, if clause has another unassigned literal, move watch to that literal.
                #    c. Else, if clause has only one unassigned literal, it becomes unit. Assign it.
                #    d. Else (all other literals are false), conflict.
                
                # Let's implement this cleanly.
                
                # Check if satisfied
                is_sat = False
                new_watch_lit = 0
                unassigned_count = 0
                for l in clause:
                    if l == v or l == -v:
                        continue
                    val = assignment[abs(l)]
                    if val == 1 and l > 0:
                        is_sat = True
                        break
                    elif val == -1 and l < 0:
                        continue
                    else: # val == 0
                        unassigned_count += 1
                        if new_watch_lit == 0:
                            new_watch_lit = l
                
                if is_sat:
                    # Remove from current watch list
                    # We can swap with last element and pop
                    watch_list[i] = watch_list[-1]
                    watch_list.pop()
                    # Do not increment i
                    continue
                
                if unassigned_count > 1:
                    # Move watch to new_watch_lit
                    # Remove from current list (swap with last)
                    watch_list[i] = watch_list[-1]
                    watch_list.pop()
                    # Add to new watch list
                    new_v = abs(new_watch_lit)
                    watches[new_v][0 if new_watch_lit > 0 else 1].append(c_idx)
                    # Do not increment i
                    continue
                
                elif unassigned_count == 1:
                    # Unit clause. Assign the unassigned literal.
                    # The literal is new_watch_lit.
                    # We need to assign it to make the clause true.
                    # So if new_watch_lit > 0, assign True. If < 0, assign False.
                    new_v = abs(new_watch_lit)
                    new_val = 1 if new_watch_lit > 0 else -1
                    # Assign
                    assignment[new_v] = new_val
                    trail.append(new_v)
                    reason[new_v] = c_idx
                    # We don't increment i, because we modified the list (swap with last)
                    # But wait, we didn't remove it from the list yet?
                    # We should remove it from the current watch list.
                    watch_list[i] = watch_list[-1]
                    watch_list.pop()
                    # The new assignment will trigger propagation of new_v in the next iteration of the while loop?
                    # No, the while loop pops from `trail`. We just pushed new_v to trail.
                    # So it will be processed in the next iteration of the `while trail_lim...` loop.
                    # This is correct.
                    continue
                
                else: # unassigned_count == 0
                    # Conflict
                    return c_idx
            
            # If we finished the while loop without conflict, we are done with this variable.
            # But we might have pushed new variables to trail.
            # The outer while loop handles that.
        
        return -1 # No conflict

    # The above propagate function is a bit messy with the in-place modification and logic.
    # Let's rewrite it more carefully.
    
    def propagate_clean():
        # Returns conflict clause index or -1
        while trail_lim and len(trail) > trail_lim[-1]:
            v = trail[-1]
            trail.pop()
            
            # Determine which watch list to check
            if assignment[v] == 1:
                watch_list = watches[v][1] # Watching -v
            else:
                watch_list = watches[v][0] # Watching v
            
            i = 0
            while i < len(watch_list):
                c_idx = watch_list[i]
                clause = all_clauses[c_idx]
                
                # Check if clause is satisfied
                is_sat = False
                new_watch_lit = 0
                unassigned_count = 0
                for l in clause:
                    if l == v or l == -v:
                        continue
                    val = assignment[abs(l)]
                    if val == 1 and l > 0:
                        is_sat = True
                        break
                    elif val == -1 and l < 0:
                        continue
                    else: # val == 0
                        unassigned_count += 1
                        if new_watch_lit == 0:
                            new_watch_lit = l
                
                if is_sat:
                    # Clause is satisfied. Remove from this watch list.
                    # We can leave it in the other watch list if it was there, but we only have one watch list per literal.
                    # Actually, a clause is in exactly two watch lists (or one if unit).
                    # When we remove it from this one, we don't add it to the satisfied literal's list because it's already satisfied.
                    # It will stay satisfied.
                    # So we just remove it from the current list.
                    watch_list[i] = watch_list[-1]
                    watch_list.pop()
                    continue # Do not increment i
                
                if unassigned_count > 1:
                    # Move watch to new_watch_lit
                    watch_list[i] = watch_list[-1]
                    watch_list.pop()
                    new_v = abs(new_watch_lit)
                    watches[new_v][0 if new_watch_lit > 0 else 1].append(c_idx)
                    continue # Do not increment i
                
                elif unassigned_count == 1:
                    # Unit propagation
                    new_v = abs(new_watch_lit)
                    new_val = 1 if new_watch_lit > 0 else -1
                    assignment[new_v] = new_val
                    trail.append(new_v)
                    reason[new_v] = c_idx
                    # Remove from current watch list
                    watch_list[i] = watch_list[-1]
                    watch_list.pop()
                    continue # Do not increment i
                
                else:
                    # Conflict
                    return c_idx
            
            # End of while i < len(watch_list)
        
        return -1

    # VSIDS Update
    def bump_activity(v):
        activity[v] += var_inc
        if activity[v] > 1e100:
            # Rescale
            for i in range(1, n + 1):
                activity[i] *= 1e-100
            var_inc *= 1e-100

    def decay_activity():
        for i in range(1, n + 1):
            activity[i] *= var_decay
        var_inc /= var_decay

    # Choose variable for decision
    def pick_variable():
        best_v = 0
        best_act = -1.0
        for v in range(1, n + 1):
            if assignment[v] == 0:
                if activity[v] > best_act:
                    best_act = activity[v]
                    best_v = v
        return best_v

    # Conflict Analysis
    # Returns (learnt_clause, backtrack_level)
    def analyze_conflict(conflict_clause_idx):
        # 1-UIP Analysis
        # We need to find the unique implication point.
        # We traverse the trail backwards from the conflict.
        
        learnt_clause = []
        seen = set()
        counter = 0
        p = 0 # Pointer in trail
        
        # Start from the last assigned variable in the trail
        # The conflict clause is all false.
        
        # We need to handle the fact that the conflict clause might be a learned clause or original.
        # The algorithm is the same.
        
        # Initialize with literals of the conflict clause
        # Actually, we start with the last literal of the conflict clause that was assigned last?
        # No, we start with the conflict clause itself.
        
        # Standard 1-UIP:
        # 1. Start with the conflict clause.
        # 2. While the clause has more than 1 literal at the current decision level:
        #    a. Pick the literal that was assigned last (highest index in trail).
        #    b. Find the clause that implied that literal (reason).
        #    c. Add the literals of the reason clause to the current clause (resolving).
        #    d. Remove the resolved literal from the current clause.
        # 3. The resulting clause is the learnt clause.
        # 4. The backtrack level is the second highest decision level in the learnt clause.
        
        # Implementation details:
        # We need to know the decision level of each variable.
        # We can store this in a `level` array.
        
        # Let's add `level` array.
        # level[v] = decision level when v was assigned.
        
        # We need to modify assign_var to set level.
        # And backtrack to reset level.
        
        # Since we are in the middle of implementation, let's add `level` array now.
        # It's a bit late, but necessary.
        
        # Actually, we can determine the level by looking at trail_lim.
        # But it's easier to store it.
        
        # Let's restart the data structure setup to include `level`.
        
        # For now, let's assume we have `level`.
        # I will rewrite the whole function with `level` included.

    # Given the complexity and the risk of bugs in a from-scratch CDCL in Python within this response,
    # and the fact that the baseline is WalkSAT, a very optimized WalkSAT with restarts and 
    # incremental clause tracking (as hinted in the "Knowledge from this run") might be safer 
    # and potentially faster to implement correctly than a buggy CDCL.
    
    # However, the directive explicitly asks for CDCL. "Strategy: Implement a CDCL solver...".
    # I will try to implement a correct CDCL. If it's too buggy, I'll fall back to a very good WalkSAT.
    # But let's try CDCL.

    # Let's refine the CDCL implementation with `level` array.
    
    # Re-initialize data structures with `level`
    assignment = [0] * (n + 1)
    level = [0] * (n + 1)
    watches = [[[], []] for _ in range(n + 1)]
    all_clauses = [list(c) for c in clauses]
    
    for i, clause in enumerate(all_clauses):
        if len(clause) == 0:
            return [False] * n
        if len(clause) >= 2:
            l1 = clause[-2]
            l2 = clause[-1]
            v1 = abs(l1)
            v2 = abs(l2)
            watches[v1][0 if l1 > 0 else 1].append(i)
            watches[v2][0 if l2 > 0 else 1].append(i)
        elif len(clause) == 1:
            l1 = clause[0]
            v1 = abs(l1)
            watches[v1][0 if l1 > 0 else 1].append(i)

    activity = [0.0] * (n + 1)
    var_inc = 1.0
    var_decay = 0.95
    
    trail = []
    trail_lim = []
    reason = [-1] * (n + 1)
    
    def assign_var(v, value, clause_id, lvl):
        assignment[v] = value
        level[v] = lvl
        trail.append(v)
        reason[v] = clause_id

    def propagate_clean():
        while trail_lim and len(trail) > trail_lim[-1]:
            v = trail[-1]
            trail.pop()
            
            if assignment[v] == 1:
                watch_list = watches[v][1]
            else:
                watch_list = watches[v][0]
            
            i = 0
            while i < len(watch_list):
                c_idx = watch_list[i]
                clause = all_clauses[c_idx]
                
                is_sat = False
                new_watch_lit = 0
                unassigned_count = 0
                for l in clause:
                    if l == v or l == -v:
                        continue
                    val = assignment[abs(l)]
                    if val == 1 and l > 0:
                        is_sat = True
                        break
                    elif val == -1 and l < 0:
                        continue
                    else:
                        unassigned_count += 1
                        if new_watch_lit == 0:
                            new_watch_lit = l
                
                if is_sat:
                    watch_list[i] = watch_list[-1]
                    watch_list.pop()
                    continue
                
                if unassigned_count > 1:
                    watch_list[i] = watch_list[-1]
                    watch_list.pop()
                    new_v = abs(new_watch_lit)
                    watches[new_v][0 if new_watch_lit > 0 else 1].append(c_idx)
                    continue
                
                elif unassigned_count == 1:
                    new_v = abs(new_watch_lit)
                    new_val = 1 if new_watch_lit > 0 else -1
                    assign_var(new_v, new_val, c_idx, trail_lim[-1] if trail_lim else 0)
                    watch_list[i] = watch_list[-1]
                    watch_list.pop()
                    continue
                
                else:
                    return c_idx
        return -1

    def bump_activity(v):
        activity[v] += var_inc
        if activity[v] > 1e100:
            for i in range(1, n + 1):
                activity[i] *= 1e-100
            var_inc *= 1e-100

    def decay_activity():
        for i in range(1, n + 1):
            activity[i] *= var_decay
        var_inc /= var_decay

    def pick_variable():
        best_v = 0
        best_act = -1.0
        for v in range(1, n + 1):
            if assignment[v] == 0:
                if activity[v] > best_act:
                    best_act = activity[v]
                    best_v = v
        return best_v

    def analyze_conflict(conflict_clause_idx):
        # 1-UIP
        learnt_clause = []
        seen = set()
        counter = 0
        p = len(trail) - 1
        
        # Start with the conflict clause
        # We need to add the literals of the conflict clause to learnt_clause
        # But we only care about literals at the current decision level?
        # No, we resolve until we have 1 literal at the current level.
        
        # Initialize learnt_clause with literals of the conflict clause
        # But we need to handle the "seen" set to avoid duplicates.
        
        # Actually, the standard algorithm starts with the conflict clause and resolves.
        
        # Let's use a queue or stack of clauses to resolve.
        # Start with the conflict clause.
        
        current_clause = list(all_clauses[conflict_clause_idx])
        
        while True:
            # Find the literal in current_clause that was assigned last
            # and is at the current decision level (or highest level).
            # We want to resolve the literal that was assigned last.
            
            last_v = 0
            last_idx = -1
            for idx, l in enumerate(current_clause):
                v = abs(l)
                if level[v] > level[last_v]:
                    last_v = v
                    last_idx = idx
            
            if last_v == 0:
                break # Should not happen
            
            # Check if we have reached the 1-UIP
            # The 1-UIP is when the current clause has only 1 literal at the current decision level.
            # Let's count literals at the current decision level.
            curr_level = trail_lim[-1] if trail_lim else 0
            count_at_level = 0
            for l in current_clause:
                v = abs(l)
                if level[v] == curr_level:
                    count_at_level += 1
            
            if count_at_level == 1:
                break
            
            # Resolve the last assigned literal
            # The literal is current_clause[last_idx]
            # It was assigned by reason[last_v]
            
            reason_clause_idx = reason[last_v]
            if reason_clause_idx == -1:
                # This is a decision variable. We shouldn't be resolving it?
                # Actually, if it's a decision variable, it's the first literal at this level.
                # If count_at_level > 1, we shouldn't have a decision variable in the middle?
                # No, decision variables are at the beginning of the trail for that level.
                # If we are resolving, we are looking at implied variables.
                # If we hit a decision variable, it means we have resolved all implied variables.
                # But then count_at_level should be 1 (the decision variable itself).
                # So if count_at_level > 1, we shouldn't hit a decision variable.
                
                # If we do, it's an error.
                break
            
            # Resolve
            reason_clause = all_clauses[reason_clause_idx]
            
            # New clause = (current_clause - {last_lit}) U reason_clause
            # But we need to remove the last_lit from current_clause.
            
            new_clause = []
            for l in current_clause:
                if l != current_clause[last_idx]:
                    new_clause.append(l)
            
            for l in reason_clause:
                if l != current_clause[last_idx] and l != -current_clause[last_idx]:
                    new_clause.append(l)
            
            current_clause = new_clause
            
            # Bump activity of variables in the new clause?
            # Usually we bump activity of variables in the learnt clause later.
            
        # current_clause is the learnt clause
        # Remove any literals that are not at the backtrack level or below?
        # No, the learnt clause is valid.
        
        # Determine backtrack level
        # The second highest level in the learnt clause.
        levels = []
        for l in current_clause:
            v = abs(l)
            levels.append(level[v])
        
        if len(levels) == 1:
            backtrack_level = 0
        else:
            levels.sort(reverse=True)
            backtrack_level = levels[1]
        
        # Add learnt clause to all_clauses
        # We need to update watches for the new clause.
        # Pick two literals to watch.
        # Let's pick the first two.
        
        learnt_clause_idx = len(all_clauses)
        all_clauses.append(current_clause)
        
        if len(current_clause) >= 2:
            l1 = current_clause[0]
            l2 = current_clause[1]
            v1 = abs(l1)
            v2 = abs(l2)
            watches[v1][0 if l1 > 0 else 1].append(learnt_clause_idx)
            watches[v2][0 if l2 > 0 else 1].append(learnt_clause_idx)
        elif len(current_clause) == 1:
            l1 = current_clause[0]
            v1 = abs(l1)
            watches[v1][0 if l1 > 0 else 1].append(learnt_clause_idx)
        
        # Bump activity of variables in the learnt clause
        for l in current_clause:
            v = abs(l)
            bump_activity(v)
        
        return learnt_clause_idx, backtrack_level

    # Main Loop
    rng = random.Random(0)
    
    # Initial random assignment? No, CDCL starts with empty assignment.
    # But for planted SAT, random start might help? No, CDCL is deterministic if VSIDS is deterministic.
    # We can use random for tie-breaking in pick_variable.
    
    for _ in range(100000): # Max iterations
        if time.time() > deadline:
            break
            
        # Propagate
        conflict_idx = propagate_clean()
        
        if conflict_idx != -1:
            # Conflict
            learnt_idx, backtrack_level = analyze_conflict(conflict_idx)
            
            # Backtrack
            while trail_lim and trail_lim[-1] > backtrack_level:
                trail_lim.pop()
            # Undo assignments above backtrack_level
            while len(trail) > (trail_lim[-1] if trail_lim else 0):
                v = trail.pop()
                assignment[v] = 0
                level[v] = 0
                reason[v] = -1
            
            decay_activity()
            
            # If backtrack_level is 0, we are at the root.
            # If the learnt clause is unit, we will propagate it in the next step.
            # If it's empty, UNSAT.
            
            # Check if learnt clause is empty
            if len(all_clauses[learnt_idx]) == 0:
                # UNSAT
                return [False] * n
        
        else:
            # No conflict
            # Check if all variables are assigned
            # We can check if len(trail) == n
            # But we might have unassigned variables if we haven't made decisions for them.
            # Actually, if propagate_clean returns -1, it means no more unit propagations.
            # If all variables are assigned, we are done.
            
            # How to check if all assigned?
            # We can keep a count of unassigned variables.
            # Or just check if len(trail) == n.
            
            if len(trail) == n:
                # All assigned. Check if all clauses are satisfied.
                # Since we propagated, they should be.
                # But let's verify just in case.
                sat = True
                for clause in all_clauses:
                    if not any(assignment[abs(l)] == (1 if l > 0 else -1) for l in clause):
                        sat = False
                        break
                if sat:
                    return [assignment[i] == 1 for i in range(1, n + 1)]
                else:
                    # Should not happen
                    break
            
            # Make a decision
            v = pick_variable()
            if v == 0:
                # No unassigned variables? Should have been caught above.
                break
            
            # Decide value
            # Random or based on activity? Just random for now.
            val = 1 if rng.random() < 0.5 else -1
            
            # Increment decision level
            trail_lim.append(len(trail))
            
            assign_var(v, val, -1, len(trail_lim) - 1)
            
            decay_activity()
    
    # If we run out of time or iterations, return the current assignment
    # It might not be satisfying, but it's the best we can do.
    # However, the task requires a satisfying assignment.
    # If we don't find one, we return a random assignment or the current one.
    # The score is the fraction solved. If we return a non-satisfying assignment, it scores 0 for that instance.
    # So it's better to return the current assignment if it happens to be satisfying, 
    # or just return the current assignment and hope for the best?
    # Actually, if we don't find a solution, we should return a valid format (list of booleans).
    # The evaluator will check if it's satisfying.
    
    # Let's return the current assignment.
    return [assignment[i] == 1 for i in range(1, n + 1)]

