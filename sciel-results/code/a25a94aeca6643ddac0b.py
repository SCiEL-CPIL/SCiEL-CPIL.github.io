import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 135.0  # Leave 15s margin for the evaluator's 150s limit
    
    def _elapsed():
        return time.time() - start_time
    
    def _row_diffs(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _max_entry(rows):
        return max(max(r) for r in rows)
    
    def _is_valid(rows):
        """Check all rows have distinct internal diffs and no cross-row conflicts."""
        used = set()
        for row in rows:
            rd = _row_diffs(row)
            if len(rd) != len(row) * (len(row) - 1) // 2:
                return False
            if rd & used:
                return False
            used |= rd
        return True
    
    # 1. Greedy initial construction
    def _greedy_full(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # Check internal distinctness
                    all_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row) + 1):
                            if j == len(row):
                                d = x - row[i]
                            else:
                                d = row[j] - row[i]
                            if d in all_diffs:
                                break
                            all_diffs.add(d)
                        else:
                            continue
                        break
                    else:
                        row.append(x)
                        used |= new_diffs
        return rows
    
    rows = _greedy_full(n_rows, k)
    
    # Build used set from initial rows
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    
    best_rows = [r[:] for r in rows]
    best_scope = _max_entry(best_rows)
    
    # Precompute row diffs for fast delta checking
    row_diffs_list = [_row_diffs(r) for r in rows]
    
    # Simulated Annealing with delta-based conflict tracking
    T = 30.0
    cooling = 0.997
    no_improve = 0
    patience = 5000
    
    while _elapsed() < deadline - 10.0:
        if T < 0.001:
            break
        
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        rd = row_diffs_list[r_idx]
        
        # Pick a random position j (1..k) to modify
        j = random.randint(1, k)
        x_old = row[j]
        
        # Determine valid range
        low = row[j - 1] + 1
        high = row[j + 1] - 1 if j < k else best_scope + 30
        
        if low > high:
            continue
        
        # Try a few candidates
        accepted = False
        for attempt in range(8):
            if _elapsed() > deadline - 10.0:
                break
            
            # Generate candidate
            if random.random() < 0.6:
                # Bias towards smaller values (scope reduction)
                if x_old > low:
                    x_new = random.randint(low, x_old - 1)
                else:
                    x_new = x_old
            else:
                if high > x_old:
                    x_new = random.randint(x_old + 1, high)
                else:
                    x_new = x_old
            
            if x_new == x_old:
                continue
            
            # Compute delta: diffs removed and added
            # Old diffs involving position j:
            #   {x_old - row[i] for i in range(j)} U {row[m] - x_old for m in range(j+1, k+1)}
            # New diffs involving position j:
            #   {x_new - row[i] for i in range(j)} U {row[m] - x_new for m in range(j+1, k+1)}
            
            old_involving = set()
            for i in range(j):
                old_involving.add(x_old - row[i])
            for m in range(j + 1, k + 1):
                old_involving.add(row[m] - x_old)
            
            new_involving = set()
            for i in range(j):
                new_involving.add(x_new - row[i])
            for m in range(j + 1, k + 1):
                new_involving.add(row[m] - x_new)
            
            # Check internal distinctness of new row
            # The new row's diffs = (rd - old_involving) | new_involving
            # Need to verify all are distinct
            remaining = rd - old_involving
            if len(remaining | new_involving) != len(remaining) + len(new_involving):
                continue  # Conflict within the row
            
            # Check against global used (excluding this row's old diffs)
            # Conflict if any d in new_involving is in (used - rd)
            conflict = False
            for d in new_involving:
                if d in used and d not in rd:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Valid move! Compute cost delta
            new_row = row[:]
            new_row[j] = x_new
            new_max = max(max(r) for r in rows) if r_idx != 0 else max(max(rows[0]), max(max(r) for r in rows[1:]))
            # Actually just compute simply:
            new_scope_val = max(max(r) for r in rows)  # rows not updated yet, use max of all
            # More precisely:
            other_max = max(max(r) for idx, r in enumerate(rows) if idx != r_idx) if n_rows > 1 else 0
            new_scope_val = max(x_new, other_max, max(row[:j]), max(row[j+1:]) if j < k else 0)
            # Simpler: just use the row's max
            new_row_max = max(new_row)
            other_max = max(max(r) for idx, r in enumerate(rows) if idx != r_idx) if n_rows > 1 else 0
            new_scope_val = max(new_row_max, other_max)
            
            old_scope_val = best_scope  # approximate with current best
            # Actually track current scope:
            curr_scope = max(max(r) for r in rows)
            
            delta_cost = new_scope_val - curr_scope
            
            # Metropolis acceptance
            if delta_cost <= 0 or random.random() < pow(2.718281828, -delta_cost / max(T, 0.001)):
                # Accept
                rows[r_idx] = new_row
                used -= rd
                used |= (rd - old_involving) | new_involving
                row_diffs_list[r_idx] = (rd - old_involving) | new_involving
                
                new_scope = _max_entry(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
                
                accepted = True
                break
        
        T *= cooling
        
        if no_improve > patience:
            # Restart with current best
            rows = [r[:] for r in best_rows]
            used = set()
            for row in rows:
                used |= _row_diffs(row)
            row_diffs_list = [_row_diffs(r) for r in rows]
            no_improve = 0
            T = max(T, 10.0)
    
    # Final polish: try to reduce scope by moving entries
    polish_deadline = deadline - 2.0
    while _elapsed() < polish_deadline:
        improved = False
        for r_idx in range(n_rows):
            if _elapsed() > polish_deadline:
                break
            row = rows[r_idx]
            rd = row_diffs_list[r_idx]
            row_max = max(row)
            
            for j in range(k, 0, -1):
                if _elapsed() > polish_deadline:
                    break
                x_cur = row[j]
                if x_cur < 1:
                    continue
                
                # Try to reduce x_cur
                low = row[j - 1] + 1
                high = row[j + 1] - 1 if j < k else x_cur - 1
                
                if low > high:
                    continue
                
                # Try values from high down to low (prefer smallest valid)
                for x_new in range(high, low - 1, -1):
                    if x_new >= x_cur:
                        continue
                    
                    old_involving = set()
                    for i in range(j):
                        old_involving.add(x_cur - row[i])
                    for m in range(j + 1, k + 1):
                        old_involving.add(row[m] - x_cur)
                    
                    new_involving = set()
                    for i in range(j):
                        new_involving.add(x_new - row[i])
                    for m in range(j + 1, k + 1):
                        new_involving.add(row[m] - x_new)
                    
                    remaining = rd - old_involving
                    if len(remaining | new_involving) != len(remaining) + len(new_involving):
                        continue
                    
                    conflict = False
                    for d in new_involving:
                        if d in used and d not in rd:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    # Accept
                    new_row = row[:]
                    new_row[j] = x_new
                    rows[r_idx] = new_row
                    used -= rd
                    used |= (rd - old_involving) | new_involving
                    row_diffs_list[r_idx] = (rd - old_involving) | new_involving
                    rd = row_diffs_list[r_idx]
                    row = new_row
                    improved = True
                    break  # Move to next position
                
                if improved:
                    break  # Restart scan of this row
        
        if not improved:
            break
        
        new_scope = _max_entry(rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in rows]
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        return rows
