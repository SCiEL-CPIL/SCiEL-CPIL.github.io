# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a CSP approach with bitset-based difference tracking and heuristic pruning.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
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
    
    start_time = time.time()
    time_limit = 115.0
    
    def solve_csp(n_rows, k, M, timeout, seed):
        deadline = time.time() + timeout
        rng = random.Random(seed)
        used = 0
        rows = []
        
        def backtrack(row, used, row_idx):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(row) == k + 1:
                rows.append(row)
                return backtrack([0], used, row_idx + 1)
            
            last = row[-1]
            rem = k + 1 - len(row)
            
            min_x = last + 1
            max_x = M - (rem - 1)
            
            if min_x > max_x:
                return False
            
            candidates = []
            for x in range(min_x, max_x + 1):
                conflict = False
                new_bits = 0
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Pruning: Check if enough space remains for the rest of the row
                s_new = len(row) + 1
                rem_new = rem - 1
                if rem_new > 0:
                    # Diffs needed: (new elements vs old) + (new elements vs new)
                    needed = rem_new * s_new + rem_new * (rem_new - 1) // 2
                    available = M - (used | new_bits).bit_count()
                    if available < needed:
                        continue
                
                candidates.append((x, new_bits))
            
            if not candidates:
                return False
            
            rng.shuffle(candidates)
            
            for x, new_bits in candidates:
                new_used = used | new_bits
                row.append(x)
                if backtrack(row, new_used, row_idx):
                    return True
                row.pop()
            
            return False
        
        success = backtrack([0], 0, 0)
        if success:
            return rows
        return None

    best_rows = greedy_rows
    best_max = greedy_max
    
    # Search for better M
    # We iterate from lb upwards.
    # To save time, we can limit the number of restarts per M.
    
    for M in range(lb, greedy_max + 1):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2:
            break
        
        # Determine budget for this M
        # Earlier M's are more likely to be the answer, so give them more time?
        # Or give uniform time?
        # If M is close to lb, it's harder to find a solution if it exists?
        # Actually, smaller M is tighter, so harder.
        # Let's give a fixed small budget per M, say 5s?
        # But we have ~110s total.
        # If we try 20 values, 5s each is 100s.
        
        budget = 5.0
        if M == lb:
            budget = 10.0 # Give more time to the theoretical minimum
        
        # Restart strategy
        # Try a few seeds
        found = False
        num_restarts = 3
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            restart_budget = budget / (num_restarts - seed)
            result = solve_csp(n_rows, k, M, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = M
                found = True
                break
        
        if found:
            break
    
    # Validate
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
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
