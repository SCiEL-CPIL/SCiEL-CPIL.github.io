import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = None
    best_max = float('inf')
    num_elements = k + 1
    
    def count_bits(n):
        return bin(n).count('1')
    
    def try_build_row(used_mask, row, pos, current_max, deadline_check):
        """Build a row starting from position pos, given current row and used_mask."""
        if deadline_check():
            return None
        
        if pos == num_elements:
            return row[:]
        
        start_val = current_max + 1
        
        # Try candidates
        for x in range(start_val, start_val + 300):
            if deadline_check():
                return None
            
            # Check if x can be added
            new_diffs = 0
            valid = True
            for a in row:
                d = x - a
                if d <= 0:
                    valid = False
                    break
                bit = 1 << (d - 1)
                if (used_mask & bit) or (new_diffs & bit):
                    valid = False
                    break
                new_diffs |= bit
            
            if not valid:
                continue
            
            # Pruning: check if we have enough remaining differences
            # We need (num_elements - pos - 1) more elements, each adding (pos+1) differences
            # Total remaining differences needed
            remaining_pos = num_elements - pos - 1
            # Each future element at position j adds (j+1) differences (to elements 0..j)
            # But we don't know the exact positions yet, so this is a loose bound
            
            row.append(x)
            result = try_build_row(used_mask | new_diffs, row, pos + 1, x, deadline_check)
            row.pop()
            
            if result is not None:
                return result
        
        return None
    
    def solve_greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << (d - 1)
                    if (used & bit) or (new & bit):
                        valid = False
                        break
                    new |= bit
                if valid:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # First, get a baseline with greedy
    baseline = solve_greedy()
    best_result = baseline
    best_max = max(max(r) for r in baseline)
    
    def elapsed():
        return time.time() - start_time
    
    def deadline_check():
        return elapsed() > deadline
    
    # Try randomized backtracking
    for attempt in range(500):
        if deadline_check():
            break
        rng = random.Random(attempt * 1000 + 42)
        
        used = 0
        rows = []
        success = True
        
        for row_idx in range(n_rows):
            if deadline_check():
                success = False
                break
            
            row = [0]
            # Use randomized backtracking for this row
            def build_row_random(pos, current_max, used_mask, row_list):
                if deadline_check():
                    return False
                if pos == num_elements:
                    return True
                
                start_val = current_max + 1
                # Generate candidates in random order within a window
                window = 50
                candidates = list(range(start_val, start_val + window))
                rng.shuffle(candidates)
                
                for x in candidates:
                    if deadline_check():
                        return False
                    
                    new_diffs = 0
                    valid = True
                    for a in row_list:
                        d = x - a
                        bit = 1 << (d - 1)
                        if (used_mask & bit) or (new_diffs & bit):
                            valid = False
                            break
                        new_diffs |= bit
                    
                    if not valid:
                        continue
                    
                    row_list.append(x)
                    if build_row_random(pos + 1, x, used_mask | new_diffs, row_list):
                        return True
                    row_list.pop()
                
                return False
            
            if build_row_random(1, 0, used, row):
                used |= 0  # already tracked in recursion
                # Need to recompute used from row
                row_diffs = 0
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        row_diffs |= (1 << (d - 1))
                used |= row_diffs
                rows.append(row)
            else:
                success = False
                break
        
        if success and len(rows) == n_rows:
            max_val = max(max(r) for r in rows)
            if max_val < best_max:
                best_max = max_val
                best_result = [r[:] for r in rows]
    
    return best_result
