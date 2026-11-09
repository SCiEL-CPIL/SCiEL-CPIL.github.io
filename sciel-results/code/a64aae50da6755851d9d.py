"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Greedy baseline
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = greedy()
    best_scope = max(r[-1] for r in best_rows)
    
    # Backtracking with iterative deepening on scope
    def build_all_rows(used_diffs, rows, row_idx, scope):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        row = [0]
        if build_single_row(used_diffs, row, 0, scope):
            rows.append(row)
            if build_all_rows(used_diffs, rows, row_idx + 1, scope):
                return True
            rows.pop()
        
        return False
    
    def build_single_row(used_diffs, row, last_val, scope):
        if time.time() > deadline:
            return False
        
        if len(row) == k + 1:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
            return True
        
        remaining = k + 1 - len(row)
        if last_val + remaining > scope:
            return False
        
        for cand in range(last_val + 1, scope + 1):
            valid = True
            new_diffs = []
            for a in row:
                d = cand - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            row.append(cand)
            for d in new_diffs:
                used_diffs.add(d)
            
            if build_single_row(used_diffs, row, cand, scope):
                return True
            
            for d in new_diffs:
                used_diffs.discard(d)
            row.pop()
        
        return False
    
    # Iterative deepening
    lower_bound = total_diffs
    search_start = max(lower_bound, best_scope - 40)
    
    for scope in range(search_start, best_scope + 1):
        if time.time() > deadline:
            break
        if time.time() - start_time > 130:
            break
        
        rows = []
        used_diffs = set()
        if build_all_rows(used_diffs, rows, 0, scope):
            best_rows = [list(r) for r in rows]
            best_scope = scope
    
    return best_rows

# EVOLVE-BLOCK-END
