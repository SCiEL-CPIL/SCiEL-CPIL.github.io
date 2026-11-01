import random
import time
from collections import Counter

def construct(n_rows, k):
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # 1. Greedy Initialization
    init_rows = greedy()
    init_gaps = [[init_rows[r][j+1] - init_rows[r][j] for j in range(k)] for r in range(n_rows)]
    
    def calc_marks(gaps_row):
        marks = [0]
        curr = 0
        for g in gaps_row:
            curr += g
            marks.append(curr)
        return marks

    def calc_diffs(marks):
        diffs = []
        n = len(marks)
        for i in range(n):
            mi = marks[i]
            for j in range(i+1, n):
                diffs.append(marks[j] - mi)
        return diffs

    # Initialize State
    curr_gaps = [list(g) for g in init_gaps]
    curr_marks = [calc_marks(g) for g in curr_gaps]
    curr_diffs = [calc_diffs(m) for m in curr_marks]
    
    global_counts = Counter()
    for d_list in curr_diffs:
        global_counts.update(d_list)
        
    curr_conflicts = sum(1 for v in global_counts.values() if v > 1)
    curr_max = max(max(m) for m in curr_marks)
    
    best_gaps = [list(g) for g in curr_gaps]
    best_max = curr_max
    best_conflicts = curr_conflicts
    
    start_time = time.time()
    time_limit = 115.0
    
    temp = 100.0
    cooling = 0.9995
    penalty = 10000
    
    while time.time() - start_time < time_limit:
        # Select move
        r = random.randint(0, n_rows - 1)
        j = random.randint(0, k - 1)
        delta = random.choice([-2, -1, 1, 1, 2])
        
        old_gap = curr_gaps[r][j]
        new_gap = old_gap + delta
        if new_gap < 1:
            continue
            
        # Save old state for rollback
        old_marks_r = curr_marks[r]
        old_diffs_r = curr_diffs[r]
        old_max_r = max(old_marks_r)
        
        # Apply change
        curr_gaps[r][j] = new_gap
        new_marks_r = calc_marks(curr_gaps[r])
        new_diffs_r = calc_diffs(new_marks_r)
        
        # Update global counts
        # Remove old
        for d in old_diffs_r:
            global_counts[d] -= 1
            if global_counts[d] == 0:
                del global_counts[d]
            elif global_counts[d] == 1:
                curr_conflicts -= 1
            # If count was >1, decrementing keeps it >=1. 
            # If it was 2, it becomes 1 (conflict resolved, handled above).
            # If it was 3, it becomes 2 (still conflict, no change).
            
        # Add new
        for d in new_diffs_r:
            if d in global_counts:
                global_counts[d] += 1
                if global_counts[d] == 2:
                    curr_conflicts += 1
            else:
                global_counts[d] = 1
                
        # Update current state
        curr_marks[r] = new_marks_r
        curr_diffs[r] = new_diffs_r
        new_max_r = max(new_marks_r)
        
        # Calculate new global max
        # This is O(n_rows * k). Small.
        new_global_max = 0
        for rr in range(n_rows):
            m = max(curr_marks[rr])
            if m > new_global_max:
                new_global_max = m
                
        old_global_max = curr_max
        
        # Scores
        old_score = old_global_max + curr_conflicts * penalty # Wait, curr_conflicts is NEW conflicts now.
        # I need to calculate score based on OLD state for comparison.
        # Let's track old_conflicts and old_max separately.
        
        # Actually, it's easier to calculate delta_score.
        # But conflicts change non-linearly.
        # Let's just store old_conflicts and old_max before modification.
        
        # ROLLBACK LOGIC
        # If we accept, we keep the changes.
        # If we reject, we revert.
        
        # Let's redo the loop with explicit old/new variables.
        
        # Break and rewrite the loop body cleanly.
        pass
        
    # The loop above was messy. Here is the clean version.
    
    # Re-init for the clean loop
    curr_gaps = [list(g) for g in init_gaps]
    curr_marks = [calc_marks(g) for g in curr_gaps]
    curr_diffs = [calc_diffs(m) for m in curr_marks]
    global_counts = Counter()
    for d_list in curr_diffs:
        global_counts.update(d_list)
    curr_conflicts = sum(1 for v in global_counts.values() if v > 1)
    curr_max = max(max(m) for m in curr_marks)
    
    best_gaps = [list(g) for g in curr_gaps]
    best_max = curr_max
    best_conflicts = curr_conflicts
    
    start_time = time.time()
    time_limit = 115.0
    temp = 50.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        r = random.randint(0, n_rows - 1)
        j = random.randint(0, k - 1)
        delta = random.choice([-2, -1, 1, 1, 2])
        
        old_gap = curr_gaps[r][j]
        new_gap = old_gap + delta
        if new_gap < 1:
            continue
            
        # Save old
        old_marks_r = curr_marks[r]
        old_diffs_r = curr_diffs[r]
        old_max_r = max(old_marks_r)
        old_conflicts = curr_conflicts
        old_max = curr_max
        
        # Apply
        curr_gaps[r][j] = new_gap
        new_marks_r = calc_marks(curr_gaps[r])
        new_diffs_r = calc_diffs(new_marks_r)
        
        # Update Global
        for d in old_diffs_r:
            global_counts[d] -= 1
            if global_counts[d] == 0:
                del global_counts[d]
            elif global_counts[d] == 1:
                curr_conflicts -= 1
                
        for d in new_diffs_r:
            if d in global_counts:
                global_counts[d] += 1
                if global_counts[d] == 2:
                    curr_conflicts += 1
            else:
                global_counts[d] = 1
                
        # Update Curr State
        curr_marks[r] = new_marks_r
        curr_diffs[r] = new_diffs_r
        new_max_r = max(new_marks_r)
        
        # Calculate New Global Max
        new_max = 0
        for rr in range(n_rows):
            m = max(curr_marks[rr])
            if m > new_max:
                new_max = m
        curr_max = new_max
        
        # Scores
        old_score = old_max + old_conflicts * 10000
        new_score = new_max + curr_conflicts * 10000
        
        if new_score < old_score or random.random() < pow(2.718, -(new_score - old_score) / temp):
            # Accept
            if curr_conflicts == 0 and new_max < best_max:
                best_max = new_max
                best_gaps = [list(g) for g in curr_gaps]
        else:
            # Reject and Rollback
            curr_gaps[r][j] = old_gap
            curr_marks[r] = old_marks_r
            curr_diffs[r] = old_diffs_r
            curr_max = old_max
            curr_conflicts = old_conflicts
            
            # Rollback Global Counts
            # Remove new
            for d in new_diffs_r:
                global_counts[d] -= 1
                if global_counts[d] == 0:
                    del global_counts[d]
                elif global_counts[d] == 1:
                    # This logic is tricky for rollback.
                    # If we added a diff that caused a conflict (1->2), removing it (2->1) resolves conflict.
                    # If we added a diff that didn't cause conflict (0->1 or 2->3), removing it doesn't change conflict count?
                    # Let's just recalculate conflicts from scratch on rollback?
                    # No, that's slow.
                    # Let's trace:
                    # Add:
                    #   0->1: no conflict change
                    #   1->2: conflict +1
                    #   2->3: no conflict change
                    # Remove:
                    #   1->0: no conflict change
                    #   2->1: conflict -1
                    #   3->2: no conflict change
                    # So:
                    # If we added and count became 2, conflict was +1. Removing makes it 1, conflict -1.
                    # If we added and count became 1 (from 0), conflict 0. Removing makes it 0, conflict 0.
                    # If we added and count became 3 (from 2), conflict 0. Removing makes it 2, conflict 0.
                    
                    # So on rollback (removing new_diffs):
                    if global_counts[d] == 2:
                        curr_conflicts -= 1
                    # If it was 1, removing makes 0. No conflict change.
                    # If it was >2, removing makes >1. No conflict change.
                    
            # Add old
            for d in old_diffs_r:
                if d in global_counts:
                    global_counts[d] += 1
                    if global_counts[d] == 2:
                        curr_conflicts += 1
                else:
                    global_counts[d] = 1
                    
        temp *= cooling

    # Convert best_gaps back to rows
    best_rows = []
    for g in best_gaps:
        row = [0]
        curr = 0
        for gap in g:
            curr += gap
            row.append(curr)
        best_rows.append(row)
        
    return best_rows
