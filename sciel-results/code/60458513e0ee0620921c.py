# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a hybrid approach: greedy baseline + local search with simulated
    annealing on value assignments to minimize max entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    start_time = time.time()
    time_limit = 145.0  # 150s limit, leave 5s safety margin
    
    # Greedy baseline
    def greedy_construct(n_rows, k):
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    def get_max_entry(rows):
        return max(max(r) for r in rows)
    
    # Local search: start from greedy, try to reduce max entry
    # Strategy: iteratively try to "compress" rows by reducing the largest element
    # and repairing conflicts via local moves
    
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Simulated annealing / local search
    def local_search(rows, max_iter, deadline, rng):
        """
        Local search that tries to reduce the max entry by swapping or adjusting
        values in rows while maintaining validity.
        """
        current_rows = [list(r) for r in rows]
        current_max = get_max_entry(current_rows)
        best_rows_ls = [list(r) for r in current_rows]
        best_max_ls = current_max
        
        # Compute all differences for current solution
        def compute_diffs(rows):
            diffs = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        diffs.add(row[j] - row[i])
            return diffs
        
        # Temperature schedule
        T0 = 10.0
        T1 = 0.1
        alpha = 0.995
        
        T = T0
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Random move: pick a row, pick a position (not 0), try to change value
            row_idx = rng.randint(0, n_rows - 1)
            pos_idx = rng.randint(1, k)  # positions 1..k (0 is fixed)
            
            old_val = current_rows[row_idx][pos_idx]
            # Try to reduce the value (compression) or small perturbation
            if rng.random() < 0.7:
                # Try to reduce
                delta = rng.randint(1, min(5, old_val))
                new_val = old_val - delta
            else:
                # Small random perturbation
                delta = rng.randint(-3, 3)
                new_val = old_val + delta
            
            if new_val <= 0 or new_val >= current_rows[row_idx][pos_idx + 1] if pos_idx < k else new_val >= current_max + 1:
                continue
            
            # Check if new_val maintains increasing order
            if pos_idx > 0 and new_val <= current_rows[row_idx][pos_idx - 1]:
                continue
            if pos_idx < k and new_val >= current_rows[row_idx][pos_idx + 1]:
                continue
            
            # Check if the move creates any duplicate differences
            # Remove old differences involving this position
            row = current_rows[row_idx]
            old_diffs = set()
            for i in range(k+1):
                if i != pos_idx:
                    if i < pos_idx:
                        old_diffs.add(row[pos_idx] - row[i])
                    else:
                        old_diffs.add(row[i] - row[pos_idx])
            
            # Compute new differences with new_val
            new_diffs = set()
            for i in range(k+1):
                if i != pos_idx:
                    if i < pos_idx:
                        new_diffs.add(new_val - row[i])
                    else:
                        new_diffs.add(row[i] - new_val)
            
            # Check for conflicts: new_diffs should not conflict with any other row's diffs
            # or with the same row's other diffs (excluding those involving pos_idx)
            all_other_diffs = set()
            for ri in range(n_rows):
                if ri == row_idx:
                    continue
                r = current_rows[ri]
                for i in range(k+1):
                    for j in range(i+1, k+1):
                        all_other_diffs.add(r[j] - r[i])
            
            # Same row diffs not involving pos_idx
            same_row_other_diffs = set()
            for i in range(k+1):
                for j in range(i+1, k+1):
                    if i != pos_idx and j != pos_idx:
                        same_row_other_diffs.add(row[j] - row[i])
            
            conflict_diffs = (new_diffs & all_other_diffs) | (new_diffs & same_row_other_diffs)
            
            if conflict_diffs:
                # Reject or accept with probability based on temperature
                # Penalty = number of conflicts
                penalty = len(conflict_diffs)
                if rng.random() > (1.0 / (1.0 + penalty * T)):
                    continue
            
            # Apply move
            row[pos_idx] = new_val
            
            # Recompute max
            new_max = get_max_entry(current_rows)
            
            if new_max < best_max_ls:
                best_max_ls = new_max
                best_rows_ls = [list(r) for r in current_rows]
            
            # Accept/reject
            if new_max <= current_max:
                current_max = new_max
            else:
                # Reject
                row[pos_idx] = old_val
            
            T *= alpha
        
        return best_rows_ls, best_max_ls
    
    # Run local search with multiple restarts
    deadline = start_time + time_limit - 2
    
    # Phase 1: Local search from greedy
    if time.time() < deadline - 5:
        rng = random.Random(42)
        ls_rows, ls_max = local_search(greedy_rows, 50000, deadline, rng)
        if ls_max < best_max and validate(ls_rows, n_rows, k):
            best_rows = ls_rows
            best_max = ls_max
    
    # Phase 2: Try to construct with a target max and use local search to repair
    # Start from a slightly larger max and try to compress
    for target_offset in range(0, 20):
        if time.time() > deadline - 5:
            break
        
        target_max = lb + target_offset
        if target_max >= best_max:
            continue
        
        # Try to build a solution with max <= target_max using greedy with bound
        def greedy_with_bound(n_rows, k, max_entry):
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    if x > max_entry:
                        return None
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            return rows
        
        bounded_rows = greedy_with_bound(n_rows, k, target_max)
        if bounded_rows is not None:
            bmax = get_max_entry(bounded_rows)
            if bmax < best_max and validate(bounded_rows, n_rows, k):
                best_rows = bounded_rows
                best_max = bmax
            
            # Try to further improve with local search
            if time.time() < deadline - 3:
                rng = random.Random(target_offset)
                ls_rows, ls_max = local_search(bounded_rows, 20000, deadline, rng)
                if ls_max < best_max and validate(ls_rows, n_rows, k):
                    best_rows = ls_rows
                    best_max = ls_max
    
    # Final validation
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
