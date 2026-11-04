import time
import random
import sys
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def greedy_construct(scope_limit=None):
        """
        Greedy construction: build rows one by one, each row greedily.
        Returns list of rows or None if stuck.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                max_val = scope_limit if scope_limit is not None else 10000
                for x in range(last_val + 1, max_val + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def local_polish(rows, time_limit):
        """
        High-precision polish: try to reduce the last element of each row
        by scanning downward from current value. Also tries intermediate positions.
        """
        rows = [r[:] for r in rows]
        
        # Build used differences set
        used_set = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_set.add(row[j] - row[i])
        
        row_maxs = [max(r) for r in rows]
        
        def compute_scope():
            return max(row_maxs)
        
        current_scope = compute_scope()
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            improved = False
            for row_idx in range(n_rows):
                if time.time() - t_start >= time_limit:
                    break
                
                row = rows[row_idx]
                
                # Try to reduce the last element first (most impactful)
                last_val = row[k]
                
                # Scan downward from last_val - 1
                for new_val in range(last_val - 1, row[k-1], -1):
                    # Check if we can set row[k] to new_val
                    old_diffs = set()
                    for i in range(k):
                        d = last_val - row[i]
                        old_diffs.add(d)
                    
                    # Check new diffs
                    valid = True
                    for i in range(k):
                        d = new_val - row[i]
                        if d in used_set or d in old_diffs:
                            valid = False
                            break
                    
                    if valid:
                        # Apply change
                        for d in old_diffs:
                            used_set.discard(d)
                        for i in range(k):
                            d = new_val - row[i]
                            used_set.add(d)
                        
                        row[k] = new_val
                        row_maxs[row_idx] = max(row)
                        current_scope = compute_scope()
                        improved = True
                        
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        
                        # Continue trying to reduce further
                        last_val = new_val
                    else:
                        break
                
                # Also try to reduce intermediate elements that are the row max
                for pos in range(1, k):
                    if time.time() - t_start >= time_limit:
                        break
                    cur_val = row[pos]
                    if cur_val < row_maxs[row_idx]:
                        continue
                    # Try reducing this position
                    for new_val in range(cur_val - 1, row[pos-1], -1):
                        if new_val >= row[pos+1]:
                            continue
                        old_diffs = set()
                        for i in range(k + 1):
                            if i != pos:
                                if i < pos:
                                    d = cur_val - row[i]
                                else:
                                    d = row[i] - cur_val
                                old_diffs.add(d)
                        
                        valid = True
                        for i in range(k + 1):
                            if i != pos:
                                if i < pos:
                                    d = new_val - row[i]
                                else:
                                    d = row[i] - new_val
                                if d in used_set or d in old_diffs:
                                    valid = False
                                    break
                        
                        if valid:
                            for d in old_diffs:
                                used_set.discard(d)
                            for i in range(k + 1):
                                if i != pos:
                                    if i < pos:
                                        d = new_val - row[i]
                                    else:
                                        d = row[i] - new_val
                                    used_set.add(d)
                            
                            row[pos] = new_val
                            row_maxs[row_idx] = max(row)
                            current_scope = compute_scope()
                            improved = True
                            
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in rows]
                            
                            cur_val = new_val
                        else:
                            break
            
            if not improved:
                break
        
        return best_rows, best_scope

    def two_phase_search(initial_rows, time_limit):
        """
        Two-phase approach:
        Phase 1: Aggressive random restarts with greedy construction
        Phase 2: Local polish on best candidates
        """
        best_rows = [r[:] for r in initial_rows]
        best_scope = max(max(r) for r in best_rows)
        
        t_start = time.time()
        phase1_time = time_limit * 0.5
        phase2_time = time_limit * 0.5
        
        # Phase 1: Random restarts
        while time.time() - t_start < phase1_time:
            # Try greedy with different random seeds/strategies
            rows = greedy_construct()
            if rows is None:
                continue
            
            current_scope = max(max(r) for r in rows)
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
            
            # Quick polish on promising candidates
            if current_scope <= best_scope + 5:
                remaining = phase1_time - (time.time() - t_start)
                if remaining > 2:
                    polished_rows, polished_scope = local_polish(rows, min(remaining, 5))
                    if polished_scope < best_scope:
                        best_scope = polished_scope
                        best_rows = polished_rows
        
        # Phase 2: Thorough polish on best
        remaining = deadline - time.time()
        if remaining > 5:
            final_rows, final_scope = local_polish(best_rows, min(phase2_time, remaining - 3))
            return final_rows
        
        return best_rows

    # Step 1: Initial greedy construction
    rows = greedy_construct()
    if rows is None:
        # Fallback: try with a larger scope
        rows = greedy_construct(scope_limit=5000)
        if rows is None:
            # Last resort: simple backtracking for small cases
            used_diffs = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for y in row:
                        if x - y in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                rows.append(row)
            return rows

    # Step 2: Two-phase search
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        best_rows = two_phase_search(rows, remaining_time)
        return best_rows

    return rows
