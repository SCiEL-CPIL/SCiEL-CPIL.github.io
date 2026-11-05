import time
import sys
import random

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0
    
    # Lower bound for scope
    # Total diffs = n_rows * k * (k+1) / 2
    total_diffs = n_rows * k * (k + 1) // 2
    min_scope = total_diffs
    
    # We know the best is 112. Let's try from min_scope up to 115.
    # If we find a solution at M, we return it.
    
    def popcount(x):
        return bin(x).count('1')
    
    def solve_scope(M, randomize=False):
        # Precompute diff requirements for pruning
        # diffs_needed[m] = number of diffs in a row of length m
        # Actually, we need remaining diffs.
        
        rows = []
        # used_mask: integer where bit d is set if difference d is used.
        # d can be up to M.
        
        def backtrack(row_idx, current_row, used_mask):
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0], used_mask):
                    return True
                rows.pop()
                return False
            
            # Pruning: Check if enough diffs are left
            m = len(current_row)
            # Diffs needed to complete current row:
            # Next element adds m diffs. Next adds m+1. ... Last adds k.
            # Sum = m + (m+1) + ... + k
            # Count of terms = k - m + 1
            # Sum = (k - m + 1) * (m + k) // 2
            rem_diffs_current = (k - m + 1) * (m + k) // 2 if m <= k else 0
            
            # Diffs needed for full rows remaining
            rem_rows = n_rows - row_idx - 1
            rem_diffs_full = rem_rows * (k * (k + 1) // 2)
            
            total_needed = rem_diffs_current + rem_diffs_full
            
            # Available diffs
            # Total possible diffs in 1..M is M.
            # Used is popcount(used_mask).
            # But we must ensure we don't use diffs > M?
            # The mask only contains bits for diffs we actually used.
            # If we used a diff d, d <= M is guaranteed by construction.
            used_count = popcount(used_mask)
            
            if total_needed > M - used_count:
                return False
            
            last_val = current_row[-1]
            
            # Generate candidates
            # x from last_val + 1 to M
            candidates = []
            for x in range(last_val + 1, M + 1):
                new_diffs_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    # If d > M, it's invalid? No, d <= x <= M.
                    # Check if d is already used
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if valid:
                    candidates.append((x, new_diffs_mask))
            
            if not candidates:
                return False
            
            # Heuristic: Sort by x (ascending) to find small scope solutions first?
            # We are fixing M, so we just need *a* solution.
            # However, finding a solution with smaller intermediate values might help?
            # Actually, for a fixed M, any valid x is fine.
            # To improve speed, we might want to try "tighter" packings?
            # Let's stick to ascending x.
            
            if randomize:
                random.shuffle(candidates)
            else:
                candidates.sort(key=lambda c: c[0])
            
            for x, new_mask in candidates:
                if backtrack(row_idx, current_row + [x], used_mask | new_mask):
                    return True
            
            return False
        
        if backtrack(0, [0], 0):
            return rows
        return None

    # Try scopes
    # Start from min_scope.
    # If it takes too long, we might want to jump.
    # But 105 to 112 is a small range.
    
    current_scope = min_scope
    best_solution = None
    
    # We will try deterministic first.
    while current_scope <= 115:
        if time.time() - start_time > time_limit:
            break
            
        # Try deterministic search
        sol = solve_scope(current_scope, randomize=False)
        if sol is not None:
            return sol
            
        current_scope += 1
        
    # If deterministic failed (likely due to time or complexity), try randomized search for a few scopes
    # Or fall back to a greedy/local search.
    
    # Greedy Fallback
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
        
    return rows
