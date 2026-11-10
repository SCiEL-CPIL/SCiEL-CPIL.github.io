import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # 1. Get a baseline solution using the greedy method
    # This ensures we always have a valid answer to return
    used_greedy = set()
    rows_greedy = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used_greedy:
                row.append(x)
                used_greedy |= new
        rows_greedy.append(row)
    
    m_baseline = max(max(r) for r in rows_greedy)
    
    # Theoretical lower bound
    num_diffs = n_rows * ((k + 1) * k // 2)
    m_lb = num_diffs
    
    # We search for M in [m_lb, m_baseline - 1]
    # If we find a solution, we return it.
    # If we exhaust the range or time out, we return the greedy solution.
    
    # To be efficient, we can try a few M values.
    # If the gap is large, we might binary search, but linear from LB is safer for "first found".
    # However, if m_baseline is 200 and m_lb is 105, checking 95 values might be slow if each takes 1s.
    # Let's add a time check in the loop.
    
    # Optimization: Bitmask for used differences
    # Max M is likely < 200, so bitmask fits in int.
    
    def solve_for_m(m):
        # used_mask bit i is 1 if difference i is used
        used_mask = 0
        rows = []
        
        def backtrack(row_idx, current_row):
            nonlocal used_mask, rows
            
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                # Start next row
                return backtrack(row_idx + 1, [0])
            
            last_val = current_row[-1]
            # Try next values
            # Heuristic: Try values that create "rare" differences? 
            # For now, just iterate.
            for x in range(last_val + 1, m + 1):
                # Calculate new differences
                new_diffs_mask = 0
                for y in current_row:
                    d = x - y
                    # d is always >= 1
                    new_diffs_mask |= (1 << d)
                
                if (new_diffs_mask & used_mask) == 0:
                    # Safe to place x
                    current_row.append(x)
                    used_mask |= new_diffs_mask
                    
                    if backtrack(row_idx, current_row):
                        return True
                    
                    # Undo
                    current_row.pop()
                    used_mask ^= new_diffs_mask # Or just reset, but XOR works if we are careful. 
                    # Actually, used_mask is nonlocal. 
                    # used_mask = used_mask ^ new_diffs_mask is safe if no other bits changed.
            
            return False
        
        if backtrack(0, [0]):
            return rows
        return None

    # Try to find a better M
    # We should probably start from m_lb and go up.
    # But if m_baseline is close to m_lb, it's fast.
    # If m_baseline is much larger, we might waste time.
    # Let's try a few steps.
    
    # Actually, the greedy solution is often not that far from optimal for small n.
    # Let's iterate M from m_lb to min(m_baseline - 1, m_lb + 50) to be safe on time.
    # Or just go up to m_baseline - 1 with time check.
    
    for m in range(m_lb, m_baseline):
        if time.time() - start_time > timeout:
            break
        
        # Optional: Quick check if m is even possible? 
        # We already know m >= m_lb.
        
        sol = solve_for_m(m)
        if sol is not None:
            return sol
            
    return rows_greedy
