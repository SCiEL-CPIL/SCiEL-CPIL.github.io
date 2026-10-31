import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    def time_left():
        return deadline - time.time()

    # 1. Greedy Init
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    # Check validity
                    # Diffs with existing elements in this row
                    # And global used set
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        break
                    x += 1
                    if x > 1000:
                        return None
            rows.append(row)
        return rows

    best_rows = greedy_init()
    if best_rows is None:
        # Fallback: construct a very sparse set
        # e.g., 0, 1, 3, 7, 15...
        # But for n=4, k=8, greedy should work.
        # If not, return a trivial valid set with large scope.
        rows = []
        offset = 0
        for _ in range(n_rows):
            row = [offset]
            val = offset
            for _ in range(k):
                val += 100
                row.append(val)
            rows.append(row)
            offset += 1000
        return rows

    best_scope = max(max(r) for r in best_rows)
    
    # State
    max_val = best_scope
    # Ensure buffer
    max_diff_limit = max_val + 100 
    counts = [0] * max_diff_limit
    
    row_maxes = [max(r) for r in best_rows]
    
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d < max_diff_limit:
                    counts[d] += 1
                else:
                    # Extend
                    counts.extend([0] * (d + 1 - max_diff_limit))
                    max_diff_limit = d + 1
                    counts[d] += 1

    def get_conflicts():
        return sum(1 for c in counts if c > 1)

    current_conflicts = get_conflicts()
    current_scope = best_scope
    
    best_conflicts = current_conflicts
    best_scope_recorded = current_scope
    best_rows_copy = [r[:] for r in best_rows]
    best_counts_copy = counts[:]

    # SA Loop
    t = 10.0
    t_min = 0.1
    cooling = 0.999
    
    # Pre-allocate list for affected diffs
    while time_left() > 5 and t > t_min:
        # Pick move
        r_idx = random.randrange(n_rows)
        i_idx = random.randrange(1, k + 1)
        
        # Delta distribution: small steps are more common
        # Use normal distribution clipped, or uniform with bias
        delta = int(random.gauss(0, 2))
        if delta == 0: continue
        
        row = best_rows[r_idx]
        old_val = row[i_idx]
        new_val = old_val + delta
        
        # Bounds check
        if new_val < 0: continue
        if i_idx > 0 and new_val <= row[i_idx - 1]: continue
        if i_idx < k and new_val >= row[i_idx + 1]: continue
        
        # Calculate affected diffs
        # Diffs involving i_idx
        old_diffs = []
        new_diffs = []
        for p in range(i_idx):
            old_diffs.append(old_val - row[p])
            new_diffs.append(new_val - row[p])
        for q in range(i_idx + 1, k + 1):
            old_diffs.append(row[q] - old_val)
            new_diffs.append(row[q] - new_val)
            
        affected = set(old_diffs + new_diffs)
        
        # Backup counts for affected
        backup = {}
        for d in affected:
            if d < len(counts):
                backup[d] = counts[d]
        
        # Apply changes
        for d in old_diffs:
            if d < len(counts):
                counts[d] -= 1
        for d in new_diffs:
            if d >= len(counts):
                counts.extend([0] * (d + 1 - len(counts)))
            counts[d] += 1
            
        # Calculate delta conflicts
        delta_conflicts = 0
        for d in affected:
            if d < len(counts):
                old_c = backup[d]
                new_c = counts[d]
                old_contrib = 1 if old_c > 1 else 0
                new_contrib = 1 if new_c > 1 else 0
                delta_conflicts += new_contrib - old_contrib
                
        new_conflicts = current_conflicts + delta_conflicts
        
        # Calculate new scope
        # Only row_maxes[r_idx] might change
        old_row_max = row_maxes[r_idx]
        new_row_max = max(new_val, old_row_max) 
        # Wait, if new_val < old_row_max, the max might be an element BEFORE i_idx.
        # We need to know the max of the rest of the row.
        # Since row is increasing, if i_idx == k, max is new_val.
        # If i_idx < k, max is row[k] (which is unchanged).
        # So:
        if i_idx == k:
            new_row_max = new_val
        else:
            # The max is still row[k] unless new_val > row[k] (impossible due to increasing check)
            # Actually, if we shift an element down, the max doesn't change (it's still the last element).
            # If we shift an element up, it can't exceed the next element.
            # So the row max is ALWAYS row[k] for i_idx < k.
            # And for i_idx == k, it is new_val.
            new_row_max = row[k]
            
        # Wait, what if we shift an element in the middle DOWN?
        # Row: 0, 5, 10. i_idx=1 (val 5). Shift to 3. Row: 0, 3, 10. Max is 10.
        # Shift to 15? Not allowed (15 > 10).
        # So yes, for i_idx < k, row max is invariant.
        # For i_idx == k, row max is new_val.
        
        new_row_maxes = row_maxes[:]
        new_row_maxes[r_idx] = new_row_max
        new_scope = max(new_row_maxes)
        
        # Energy
        # We want to minimize Conflicts, then Scope.
        # E = Conflicts * 1000 + Scope
        old_energy = current_conflicts * 1000 + current_scope
        new_energy = new_conflicts * 1000 + new_scope
        
        delta_energy = new_energy - old_energy
        
        accept = False
        if delta_energy < 0:
            accept = True
        else:
            if random.random() < pow(2.71828, -delta_energy / max(t, 1e-6)):
                accept = True
                
        if accept:
            # Commit
            best_rows[r_idx][i_idx] = new_val
            row_maxes[r_idx] = new_row_max
            current_conflicts = new_conflicts
            current_scope = new_scope
            
            if (current_conflicts, current_scope) < (best_conflicts, best_scope_recorded):
                best_conflicts = current_conflicts
                best_scope_recorded = current_scope
                best_rows_copy = [r[:] for r in best_rows]
                best_counts_copy = counts[:]
                
        else:
            # Revert
            for d in affected:
                if d in backup:
                    counts[d] = backup[d]
            
        t *= cooling

    # Return best valid
    if best_conflicts == 0:
        return best_rows_copy
    else:
        # Return best_rows (which might be invalid if we didn't find 0 conflicts)
        # But greedy was valid.
        # If we modified best_rows and it became invalid, we should return the last valid one?
        # No, best_rows_copy is the best VALID one found?
        # Wait, we update best_rows_copy ONLY if (conf, scope) is better.
        # If we start valid (conf=0), and then go to conf=1, we don't update best_rows_copy.
        # So best_rows_copy remains the initial valid solution or a better valid one.
        return best_rows_copy
