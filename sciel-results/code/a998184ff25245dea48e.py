import time
import random
import copy
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    def greedy_construct():
        """Build rows greedily, one at a time, using smallest available numbers."""
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
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def calc_conflicts(rows):
        """Return conflict count and diff_counter for the given rows."""
        diff_counter = Counter()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff_counter[row[j] - row[i]] += 1
        conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
        return conflicts, diff_counter
    
    def row_diffs(row):
        """All positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    # Initial greedy construction
    best_rows = greedy_construct()
    best_scope = get_scope(best_rows)
    
    # Try to improve with a more aggressive greedy: randomize row order
    for trial in range(20):
        rows = []
        used = set()
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
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = copy.deepcopy(rows)
    
    current_rows = copy.deepcopy(best_rows)
    curr_conflicts, diff_counter = calc_conflicts(current_rows)
    curr_scope = get_scope(current_rows)
    
    T = 80.0
    cooling = 0.995
    
    def try_move(r, j, new_val, rows, dcounter):
        """Attempt to move element (r,j) to new_val. Returns (success, new_scope, new_conflicts)."""
        if new_val <= rows[r][j - 1]:
            return False, 0, 0
        if j < k and new_val >= rows[r][j + 1]:
            return False, 0, 0
        
        old_val = rows[r][j]
        if new_val == old_val:
            return False, 0, 0
        
        # Remove old diffs
        old_diffs = []
        for i in range(j):
            d = old_val - rows[r][i]
            old_diffs.append(d)
            dcounter[d] -= 1
            if dcounter[d] == 0:
                del dcounter[d]
        for i in range(j + 1, len(rows[r])):
            d = rows[r][i] - old_val
            old_diffs.append(d)
            dcounter[d] -= 1
            if dcounter[d] == 0:
                del dcounter[d]
        
        # Apply new
        rows[r][j] = new_val
        
        # Add new diffs
        for i in range(j):
            d = new_val - rows[r][i]
            dcounter[d] += 1
        for i in range(j + 1, len(rows[r])):
            d = rows[r][i] - new_val
            dcounter[d] += 1
        
        new_scope = max(max(row) for row in rows)
        new_conflicts = sum(c - 1 for c in dcounter.values() if c > 1)
        
        return True, new_scope, new_conflicts
    
    def revert_move(r, j, old_val, new_val, rows, dcounter):
        """Revert a move from new_val back to old_val."""
        # Remove new diffs
        for i in range(j):
            d = new_val - rows[r][i]
            dcounter[d] -= 1
            if dcounter[d] == 0:
                del dcounter[d]
        for i in range(j + 1, len(rows[r])):
            d = rows[r][i] - new_val
            dcounter[d] -= 1
            if dcounter[d] == 0:
                del dcounter[d]
        
        # Restore old
        rows[r][j] = old_val
        
        # Re-add old diffs
        for i in range(j):
            d = old_val - rows[r][i]
            dcounter[d] += 1
        for i in range(j + 1, len(rows[r])):
            d = rows[r][i] - old_val
            dcounter[d] += 1
    
    # Local search: try to shrink each element to minimum valid value
    def local_search_shrink(rows, dcounter, scope, conflicts):
        """Try to shrink elements to reduce scope."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for r in range(n_rows):
                for j in range(k, 0, -1):  # from right to left
                    old_val = rows[r][j]
                    # Try to shrink to minimum valid
                    min_possible = rows[r][j - 1] + 1
                    if min_possible >= old_val:
                        continue
                    
                    # Binary search for minimum valid value
                    lo = min_possible
                    hi = old_val - 1
                    found = False
                    while lo <= hi and time.time() < deadline:
                        mid = (lo + hi) // 2
                        success, new_scope, new_conflicts = try_move(r, j, mid, rows, dcounter)
                        if success:
                            if new_conflicts == 0:
                                # Accept this move
                                scope = new_scope
                                conflicts = new_conflicts
                                found = True
                                hi = mid - 1
                            else:
                                revert_move(r, j, old_val, mid, rows, dcounter)
                                lo = mid + 1
                        else:
                            lo = mid + 1
                    
                    if found:
                        improved = True
                        if scope < best_scope:
                            best_scope = scope
                            best_rows = copy.deepcopy(rows)
        return scope, conflicts
    
    # Run local search from initial state
    curr_scope, curr_conflicts = local_search_shrink(current_rows, diff_counter, curr_scope, curr_conflicts)
    
    # SA loop
    while time.time() < deadline:
        # Phase 1: Local search shrink
        curr_scope, curr_conflicts = local_search_shrink(current_rows, diff_counter, curr_scope, curr_conflicts)
        
        if time.time() > deadline - 5:
            break
        
        # Phase 2: SA perturbation
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k)
        
        old_val = current_rows[r][j]
        
        min_val = current_rows[r][j - 1] + 1
        max_val = min(old_val + 15, best_scope + 10)
        if j < k:
            max_val = min(max_val, current_rows[r][j + 1] - 1)
        
        if min_val > max_val:
            continue
        
        new_val = random.randint(min_val, max_val)
        if new_val == old_val:
            continue
        
        success, new_scope, new_conflicts = try_move(r, j, new_val, current_rows, diff_counter)
        if not success:
            continue
        
        delta_scope = new_scope - curr_scope
        delta_conflicts = new_conflicts - curr_conflicts
        delta_energy = delta_scope + 15 * delta_conflicts
        
        accept = False
        if delta_energy < 0:
            accept = True
        else:
            if T > 0.01:
                accept = random.random() < math.exp(-delta_energy / T)
        
        if accept:
            curr_scope = new_scope
            curr_conflicts = new_conflicts
            if new_conflicts == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = copy.deepcopy(current_rows)
        else:
            revert_move(r, j, old_val, new_val, current_rows, diff_counter)
        
        T *= cooling
        if T < 0.5:
            T = 0.5
    
    return best_rows
