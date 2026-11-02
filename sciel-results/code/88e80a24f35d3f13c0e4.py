import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    # If parameters are too large for exhaustive search, use greedy
    # Heuristic: total diffs > 1000 is likely too slow for pure Python backtracking in 110s
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    def greedy_construct():
        used = set()
        res = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            res.append(row)
        return res

    # If problem is small, try to optimize
    if total_diffs_needed <= 500:
        lb = total_diffs_needed
        
        # Get greedy upper bound
        greedy_sol = greedy_construct()
        greedy_max = max(max(r) for r in greedy_sol)
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, curr_row, M_limit):
            if time.time() > deadline:
                return False # Force unwind

            if row_idx == n_rows:
                return True
            
            if len(curr_row) == k + 1:
                # Symmetry breaking: rows sorted by last element
                if row_idx > 0 and curr_row[-1] < rows[-1][-1]:
                    return False
                
                rows.append(curr_row)
                if backtrack(row_idx + 1, [0], M_limit):
                    return True
                rows.pop()
                return False
            
            last_val = curr_row[-1]
            
            # Heuristic: try to keep numbers small?
            # Just iterate from last_val + 1
            
            for x in range(last_val + 1, M_limit + 1):
                valid = True
                new_diffs = []
                for a in curr_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check if we can still fit remaining numbers in the row?
                # We need (k + 1 - len(curr_row) - 1) more numbers.
                # Min next is x + 1.
                # Max allowed is M_limit.
                # If x + (k + 1 - len(curr_row) - 1) > M_limit: break
                
                if x + (k + 1 - len(curr_row) - 1) > M_limit:
                    break

                used_diffs.update(new_diffs)
                curr_row.append(x)
                
                if backtrack(row_idx, curr_row, M_limit):
                    return True
                
                curr_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False

        for M in range(lb, greedy_max + 1):
            if time.time() > deadline:
                break
            
            used_diffs.clear()
            rows.clear()
            
            if backtrack(0, [0], M):
                return rows

    return greedy_construct()
