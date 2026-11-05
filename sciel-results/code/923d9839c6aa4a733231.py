import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0
    
    # Helper to generate a quick greedy solution for initial bound
    def get_greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last = row[-1]
                x = last + 1
                while True:
                    # Check if x creates a conflict
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                    if valid:
                        break
                    x += 1
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
            rows.append(row)
        return rows

    init_rows = get_greedy_solution()
    best_max = max(row[-1] for row in init_rows)
    best_solution = [r[:] for r in init_rows]
    
    def backtrack(rows, used_mask, current_max, r, c):
        nonlocal best_max, best_solution
        
        if time.time() - start_time > time_limit:
            return False
        
        if r == n_rows:
            if current_max < best_max:
                best_max = current_max
                best_solution = [row[:] for row in rows]
            return True
        
        if c == k + 1:
            return backtrack(rows, used_mask, current_max, r + 1, 0)
        
        if c == 0:
            rows[r][c] = 0
            return backtrack(rows, used_mask, current_max, r, c + 1)
        
        prev_val = rows[r][c - 1]
        
        # We only care about solutions with max < best_max
        # So val must be < best_max
        # Also val > prev_val
        
        for val in range(prev_val + 1, best_max):
            conflict = False
            new_diffs_bits = 0
            for i in range(c):
                diff = val - rows[r][i]
                bit = 1 << diff
                if (used_mask & bit) or (new_diffs_bits & bit):
                    conflict = True
                    break
                new_diffs_bits |= bit
            
            if conflict:
                continue
            
            rows[r][c] = val
            new_max = val if val > current_max else current_max
            
            # Prune if new_max is not better
            if new_max >= best_max:
                continue
            
            if backtrack(rows, used_mask | new_diffs_bits, new_max, r, c + 1):
                return True
        
        return False

    rows = [[0] * (k + 1) for _ in range(n_rows)]
    backtrack(rows, 0, 0, 0, 0)
    
    return best_solution
