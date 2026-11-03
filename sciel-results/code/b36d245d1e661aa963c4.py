import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 110.0  # 110s safety margin within 120s limit
    k_plus_1 = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    best_rows = None
    best_max = float('inf')
    
    def validate(rows):
        """Check if all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def get_max(rows):
        return max(max(row) for row in rows)
    
    def greedy_complete(partial_rows, used_diffs, time_limit):
        """Complete partial rows greedily, returning None if stuck."""
        rows = [list(r) for r in partial_rows]
        used = set(used_diffs)
        
        for idx in range(len(rows)):
            row = rows[idx]
            while len(row) < k_plus_1:
                if time.time() > time_limit:
                    return None
                # Find smallest x > row[-1] such that all differences are unused
                x = row[-1] + 1
                found = False
                while not found:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                    else:
                        x += 1
                        if x > 200:  # Safety bound
                            return None
        return rows, used
    
    def backtrack_search(partial_rows, used_diffs, time_limit, depth=0):
        """Backtracking search to complete the rows."""
        if time.time() > time_limit:
            return None
        
        if len(partial_rows) == n_rows:
            return [list(r) for r in partial_rows]
        
        row = [0]
        used = set(used_diffs)
        
        def try_complete_row(row, used, time_limit):
            if len(row) == k_plus_1:
                return row, used
            if time.time() > time_limit:
                return None
            
            best_x = None
            best_used = None
            
            # Try candidates in order of increasing x
            x = row[-1] + 1
            while x <= 150:  # Reasonable upper bound for search
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    new_row = row + [x]
                    new_used = used | new_diffs
                    result = try_complete_row(new_row, new_used, time_limit)
                    if result is not None:
                        return result
                x += 1
            return None
        
        result = try_complete_row(row, used, time_limit)
        if result is not None:
            new_row, new_used = result
            new_partial = partial_rows + [new_row]
            sub_result = backtrack_search(new_partial, new_used, time_limit, depth + 1)
            if sub_result is not None:
                return sub_result
        
        return None
    
    def local_improve(rows, time_limit):
        """Try to reduce the maximum entry by local search."""
        if time.time() > time_limit:
            return rows
        
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                for pos in range(1, k_plus_1):
                    if time.time() > time_limit:
                        return rows
                    original_val = rows[r_idx][pos]
                    # Try to decrease this value
                    for new_val in range(rows[r_idx][pos - 1] + 1, original_val):
                        # Check if this position can be changed
                        row = rows[r_idx]
                        # Remove old differences involving position pos
                        old_diffs = set()
                        for i in range(k_plus_1):
                            if i != pos:
                                if row[pos] > row[i]:
                                    old_diffs.add(row[pos] - row[i])
                                else:
                                    old_diffs.add(row[i] - row[pos])
                        
                        # Temporarily set new value
                        row[pos] = new_val
                        # Check strict increase
                        valid = True
                        for i in range(pos - 1, pos + 1):
                            if i >= 0 and i < k_plus_1:
                                if i < pos and row[i] >= new_val:
                                    valid = False
                                    break
                                if i > pos and row[i] <= new_val:
                                    valid = False
                                    break
                        
                        if not valid:
                            continue
                        
                        # Check new differences
                        new_diffs = set()
                        conflict = False
                        for i in range(k_plus_1):
                            if i != pos:
                                d = abs(row[pos] - row[i])
                                if d == 0:
                                    conflict = True
                                    break
                                if d in used and d not in old_diffs:
                                    conflict = True
                                    break
                                new_diffs.add(d)
                        
                        if conflict:
                            continue
                        
                        # Accept the change
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        break
                    # Restore if no improvement
                    if rows[r_idx][pos] != original_val:
                        pass  # Already updated
                    else:
                        pass  # No change made
            
            # If no improvement in this pass, try random perturbations
            if not improved:
                break
        
        return rows
    
    # Strategy 1: Pure greedy (baseline)
    rows_greedy, _ = greedy_complete([], set(), deadline)
    if rows_greedy:
        best_rows = rows_greedy
        best_max = get_max(rows_greedy)
    
    # Strategy 2: Multiple greedy with different orderings
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Randomly permute which row we build first (but all must be built)
        # Actually, let's try greedy with different starting points
        # Try building rows with slight randomization in candidate selection
        
        used = set()
        rows = []
        success = True
        
        for r_idx in range(n_rows):
            row = [0]
            # Add some randomness: try candidates in slightly shuffled order
            for pos in range(1, k_plus_1):
                found = False
                # Try x from row[-1]+1 upward, but with some randomization
                candidates = []
                x = row[-1] + 1
                while x <= row[-1] + 20:  # Look ahead a bit
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    success = False
                    break
                
                # Pick from candidates with some randomness
                if random.random() < 0.7:
                    chosen = candidates[0]  # Greedy
                else:
                    chosen = random.choice(candidates[:min(3, len(candidates))])
                
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
            
            if not success:
                break
            rows.append(row)
        
        if success and len(rows) == n_rows:
            if validate(rows):
                m = get_max(rows)
                if m < best_max:
                    best_max = m
                    best_rows = [list(r) for r in rows]
    
    # Strategy 3: Backtracking search with time limit
    if time.time() < deadline - 30:
        bt_result = backtrack_search([], set(), deadline - 10)
        if bt_result:
            m = get_max(bt_result)
            if m < best_max:
                best_max = m
                best_rows = [list(r) for r in bt_result]
    
    # Strategy 4: Local improvement on best solution
    if best_rows and time.time() < deadline - 5:
        improved = local_improve(best_rows, deadline - 2)
        if validate(improved):
            m = get_max(improved)
            if m < best_max:
                best_max = m
                best_rows = improved
    
    # Final validation and return
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows
    
    return best_rows
# EVOLVE-BLOCK-END
