import time
import random

def construct(n_rows, k):
    """
    Finds a difference triangle set with n_rows rows and k elements per row (excluding 0).
    Returns the list of rows.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # Parameters for this task: n_rows=4, k=7
    # Row length is k + 1 = 8.
    # Elements: 0, x1, ..., x7
    
    # Lower bound for the largest entry S
    # Total distinct differences = n_rows * (k * (k+1) / 2)
    # Differences range from 1 to S.
    # So S >= total_diffs
    lower_bound = n_rows * k * (k + 1) // 2
    
    best_scope = float('inf')
    best_rows = None
    
    # 1. Greedy Construction to get a valid baseline
    # This ensures we return something valid even if search fails.
    def greedy_solve():
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            for i in range(1, k + 1):
                # Try to find the smallest x > row[-1]
                x = row[-1] + 1
                while True:
                    valid = True
                    new_mask = 0
                    for a in row:
                        d = x - a
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        new_mask |= (1 << d)
                    if valid:
                        used_mask |= new_mask
                        row.append(x)
                        break
                    x += 1
                    if x > 500: # Hard cap to prevent infinite loop in bad cases
                        return None
            rows.append(row)
        return rows

    greedy_sol = greedy_solve()
    if greedy_sol:
        best_rows = greedy_sol
        best_scope = max(max(r) for r in best_rows)
    else:
        # Fallback: very simple construction, might be suboptimal but valid
        # Just use large gaps
        rows = []
        base = 1
        for _ in range(n_rows):
            row = [0]
            for i in range(1, k + 1):
                row.append(row[-1] + 100) # Large gap
            rows.append(row)
        best_rows = rows
        best_scope = max(max(r) for r in best_rows)

    # 2. Randomized Backtracking Search
    # Search for S in [lower_bound, best_scope - 1]
    # We iterate downwards from best_scope - 1 to lower_bound.
    
    def attempt_solve(S_max, rng):
        """
        Attempts to find a solution with max value <= S_max using backtracking.
        Returns list of rows or None.
        """
        # rows[r][p] where r in 0..3, p in 0..7
        # rows[r][0] is always 0
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        
        def backtrack(r, p, used_mask):
            if r == n_rows:
                return True
            
            if p == k + 1:
                # Move to next row
                return backtrack(r + 1, 1, used_mask)
            
            prev_val = rows[r][p - 1]
            candidates = []
            
            # Generate valid candidates for this position
            # x must be > prev_val and <= S_max
            for x in range(prev_val + 1, S_max + 1):
                new_mask = 0
                valid = True
                # Check differences with all previous elements in this row
                for i in range(p):
                    d = x - rows[r][i]
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if valid:
                    candidates.append((x, new_mask))
            
            if not candidates:
                return False
            
            # Shuffle to diversify search paths
            rng.shuffle(candidates)
            
            for x, new_mask in candidates:
                rows[r][p] = x
                if backtrack(r, p + 1, used_mask | new_mask):
                    return True
            
            return False
        
        if backtrack(0, 1, 0):
            return [list(row) for row in rows]
        return None

    # Search strategy
    # We want to minimize S.
    # We check S from best_scope - 1 down to lower_bound.
    # If we find a solution at S, we update best_scope = S and continue to S-1.
    
    # Time allocation
    # If best_scope is 150, we have ~40 values to check.
    # 135s / 40 = 3.375s per value.
    
    current_S = best_scope - 1
    # To avoid iterating too many if best_scope is huge, cap the initial range?
    # But best_scope from greedy should be reasonable (~140-160).
    
    while current_S >= lower_bound:
        if time.time() > deadline - 5:
            break
        
        # Allocate time for this S
        # Let's use a fixed time slice, e.g., 2.0 seconds per S value.
        # This allows trying ~60 values in 120s.
        
        slice_start = time.time()
        max_time_per_S = 2.0
        
        found = False
        restart_count = 0
        max_restarts = 50 # Upper bound on restarts if time allows
        
        while time.time() - slice_start < max_time_per_S and restart_count < max_restarts:
            # Create a unique seed for each restart to ensure different paths
            # Use current_S and restart_count for determinism in debugging, 
            # but random.SystemRandom or just random.Random() for variety?
            # Using a fixed seed per S-restart allows reproducibility, 
            # but we want to explore space. 
            # Let's use a global random seed offset.
            seed = (current_S * 1000) + restart_count
            rng = random.Random(seed)
            
            sol = attempt_solve(current_S, rng)
            if sol:
                # Verify solution
                all_diffs = set()
                valid = True
                for row in sol:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                
                if valid:
                    best_scope = current_S
                    best_rows = sol
                    found = True
                    # We found a valid solution for this S.
                    # We break out of the restart loop to try S-1.
                    break
            restart_count += 1
        
        # If we found a solution, we might want to spend a bit more time 
        # to see if we can find it faster or with different structure? 
        # No, the scope is the same. We just need to move to lower S.
        
        current_S -= 1

    return best_rows
