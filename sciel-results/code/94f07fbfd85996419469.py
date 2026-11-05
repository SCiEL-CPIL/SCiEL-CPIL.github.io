import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Lower bound on M
    # Total differences = n_rows * k * (k+1) / 2
    # Min M such that 1..M contains enough diffs?
    # Actually, the diffs are distinct. So we need at least total_diffs distinct values.
    # The max diff is M. So M >= total_diffs? No.
    # The diffs are in [1, M]. So we need M >= total_diffs.
    # total_diffs = 7 * 15 = 105.
    # So M >= 105.
    
    min_M = n_rows * k * (k + 1) // 2
    max_M_search = min_M + 50 # Search up to 155
    
    best_solution = None
    best_M = float('inf')
    
    # Seed solution for fallback
    # Run the original greedy to have a valid solution
    def greedy_construct():
        used, rows = set(), []
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

    seed_rows = greedy_construct()
    seed_M = max(max(row) for row in seed_rows)
    best_solution = seed_rows
    best_M = seed_M
    
    # Backtracking search
    def search(M):
        used = bytearray(M + 1) # 1 if used
        rows = []
        
        def backtrack(r_idx, m_idx, cur_row):
            nonlocal best_solution, best_M
            if time.time() > deadline:
                return False
            
            if m_idx == k + 1:
                rows.append(cur_row[:])
                if r_idx == n_rows - 1:
                    # Success
                    best_solution = [row[:] for row in rows]
                    best_M = M
                    return True
                else:
                    # Start next row
                    if backtrack(r_idx + 1, 0, [0]):
                        return True
                    rows.pop()
                    return False
            else:
                last = cur_row[-1]
                # Pruning: minimum possible last mark
                # We need (k+1 - m_idx) more marks.
                # Smallest they can be: last+1, last+2, ...
                # So min last mark = last + (k+1 - m_idx)
                if last + (k + 1 - m_idx) > M:
                    return False
                
                # Try candidates
                # Start from last + 1
                # To add randomness, we can shuffle a small window of candidates
                # But for minimization, increasing order is usually better.
                # Let's try increasing order first.
                
                for x in range(last + 1, M + 1):
                    # Check diffs
                    valid = True
                    diffs_to_add = []
                    for a in cur_row:
                        d = x - a
                        if used[d]:
                            valid = False
                            break
                        diffs_to_add.append(d)
                    
                    if not valid:
                        continue
                    
                    # Add diffs
                    for d in diffs_to_add:
                        used[d] = 1
                    cur_row.append(x)
                    
                    if backtrack(r_idx, m_idx + 1, cur_row):
                        return True
                    
                    # Backtrack
                    for d in diffs_to_add:
                        used[d] = 0
                    cur_row.pop()
                
                return False
        
        return backtrack(0, 0, [0])

    # Try M values
    for M in range(min_M, max_M_search + 1):
        if time.time() > deadline:
            break
        
        # Random restarts for each M
        # Maybe 5 restarts?
        found = False
        for _ in range(5):
            if time.time() > deadline:
                break
            if search(M):
                found = True
                break
        
        if found:
            return best_solution

    return best_solution
