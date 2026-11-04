"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry
    lower_bound = total_diffs
    
    # Helper to compute differences of a row
    def row_diffs(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))]
    
    # Helper to generate a candidate row with max value bounded by M
    # Uses a greedy approach with randomization to find a valid row within scope M
    # Returns None if no valid row found (though greedy usually finds one if M is large enough)
    def generate_candidate_row(used_global, M, rng):
        # We need k+1 numbers: 0 = a0 < a1 < ... < ak <= M
        # We fill them one by one.
        row = [0]
        for _ in range(k):
            # Find a valid next value x
            # Constraints:
            # 1. x > row[-1]
            # 2. x <= M
            # 3. (x - a_i) not in used_global for all i < current_len
            # 4. (x - a_i) not in {a_j - a_i for j < i} (internal distinctness is automatic if x > a_i)
            # Actually, internal distinctness: x - a_i are distinct because a_i are distinct.
            # So we just need to check against used_global.
            
            # To make it faster, we can iterate x from row[-1]+1 to M
            # But M can be large.
            # Optimization: We can pick x randomly in [row[-1]+1, M] and check?
            # No, we need to ensure we can complete the row.
            # Greedy: pick the smallest valid x.
            # Randomized Greedy: pick a random valid x from a subset?
            
            # Let's try a "smart" greedy:
            # Iterate x from row[-1] + 1 to M.
            # Check conflicts.
            # If found, append.
            # If we reach M and no x found, fail.
            
            # To add randomness, we can sometimes skip valid x?
            # Or we can try a few random x first?
            
            # For now, standard greedy is fast and deterministic for a given used_global.
            # To get different candidates, we need to vary the used_global or the row construction.
            # Since used_global is fixed for the candidate generation step (initially empty or partial),
            # we need to randomize the row itself.
            
            # Strategy:
            # 1. Generate a "base" row using pure greedy.
            # 2. Perturb it?
            
            # Alternative:
            # Randomly select x from [row[-1]+1, M] with a bias towards smaller values?
            # If we pick x too large, we might not fit k elements.
            
            # Let's stick to:
            # Iterate x from row[-1]+1 to M.
            # If (x - a_i) not in used_global for all i, accept x.
            # This produces the "tightest" row.
            # To get diversity, we can randomize the order of checking? No.
            
            # Better:
            # We will generate multiple candidates by allowing "gaps".
            # Pick x randomly in [row[-1]+1, M].
            # Check if valid.
            # If valid, accept.
            # If not, try next?
            # This might be slow if M is large.
            
            # Efficient approach:
            # The space of valid x is dense enough for small k.
            # Let's try a hybrid:
            # Try a few random x first (e.g., 10).
            # If none work, fall back to linear scan from row[-1]+1.
            
            found = False
            # Random attempts
            for _ in range(10):
                x = rng.randint(row[-1] + 1, M)
                valid = True
                for a in row:
                    if (x - a) in used_global:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    found = True
                    break
            
            if not found:
                # Linear scan
                for x in range(row[-1] + 1, M + 1):
                    valid = True
                    for a in row:
                        if (x - a) in used_global:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        found = True
                        break
            
            if not found:
                return None
                
        return row

    # Main loop: Decrease M until a valid set of n_rows is found
    # Start M from a safe upper bound
    # Upper bound estimate: 2 * lower_bound + 100
    M_max = 2 * lower_bound + 100
    if M_max < 1000:
        M_max = 1000
        
    # We will search downwards from M_max
    # But we don't know M_max is valid. Let's start with a known valid one?
    # The greedy algorithm with M=inf produces a valid set.
    # Let's find a valid set with a large M first to have a baseline.
    
    # Actually, we can just start M at M_max and decrease.
    # If we fail to find a set for M, we try M-1?
    # This is slow if M_max is huge.
    
    # Better: Binary search?
    # Is "existence of a valid set with max entry <= M" monotonic?
    # Yes, if a set exists for M, it exists for M+1 (just use the same set).
    # So we can binary search for the minimum M.
    
    # But the search space for a single M is hard (NP-hard).
    # We will use a heuristic:
    # 1. Set M = M_max
    # 2. Try to find n_rows disjoint rows with scope <= M.
    # 3. If successful, set M_new = M - step.
    # 4. If failed, we might need to increase M or restart with different random seeds.
    
    # Let's implement a function `try_find_set(M, n_rows, k, rng)`
    # It returns a list of rows if found, else None.
    
    def try_find_set(M, n_rows, k, rng, time_limit=5.0):
        start_t = time.time()
        # We will try to build the set greedily one row at a time.
        # To avoid getting stuck, we can use backtracking?
        # Or we can generate many candidates for each row and pick the best combination?
        
        # Strategy:
        # For row 0: Generate a list of candidate rows (diverse).
        # For row 1: Generate candidates compatible with row 0.
        # ...
        
        # This is a sequential constraint satisfaction.
        # If we get stuck at row i, we can backtrack to row i-1 and try a different candidate.
        
        # To keep it simple and fast, we will use a "Randomized Greedy with Restarts".
        # Each restart:
        # 1. used_global = set()
        # 2. rows = []
        # 3. For i in range(n_rows):
        #    Generate a candidate row with scope <= M, avoiding used_global.
        #    If fail, restart.
        #    If success, add to rows, update used_global.
        # 4. If all n_rows generated, return rows.
        
        # To improve success rate, we can generate multiple candidates for each step and pick the one that leaves the most "room" for future rows?
        # "Room" is hard to define.
        
        # Let's stick to simple randomized greedy with many restarts.
        
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_t > time_limit:
                return None
                
            used_global = set()
            rows = []
            success = True
            
            for i in range(n_rows):
                # Generate a candidate
                # We want to be "tight" to save space, but "random" to avoid local traps.
                # Let's bias the random x towards the lower end.
                
                row = [0]
                for _ in range(k):
                    found = False
                    # Try random values
                    # Bias towards smaller values: x = row[-1] + 1 + rng.randint(0, (M - row[-1]) // 2)
                    # This keeps the row compact.
                    
                    # Try 20 random attempts
                    for _ in range(20):
                        # Uniform in [row[-1]+1, M]
                        # But we want compactness.
                        # Let's try uniform first.
                        x = rng.randint(row[-1] + 1, M)
                        valid = True
                        for a in row:
                            if (x - a) in used_global:
                                valid = False
                                break
                        if valid:
                            row.append(x)
                            found = True
                            break
                    
                    if not found:
                        # Fall back to linear scan from row[-1]+1
                        # This ensures we find the smallest valid x if it exists
                        for x in range(row[-1] + 1, M + 1):
                            valid = True
                            for a in row:
                                if (x - a) in used_global:
                                    valid = False
                                    break
                            if valid:
                                row.append(x)
                                found = True
                                break
                    
                    if not found:
                        success = False
                        break
                
            if success:
                return rows
                
        return None

    # Search for the minimal M
    # Start with a high M, verify it works, then binary search down?
    # Or just step down?
    
    # Let's find a valid M first.
    # M = lower_bound is the theoretical min.
    # M = 2 * lower_bound is likely valid.
    
    # Let's try to find a valid set for M = lower_bound + offset
    # We will iterate M from lower_bound upwards?
    # No, we want to minimize M.
    # Let's start from a known good M (e.g., from a simple greedy) and decrease.
    
    # Simple greedy to get an initial M
    used = set()
    init_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        init_rows.append(row)
    
    current_best_M = max(max(r) for r in init_rows)
    current_best_rows = [r[:] for r in init_rows]
    
    # Now try to improve by decreasing M
    # We will try M values: current_best_M - 1, current_best_M - 2, ...
    # But we can skip some if we fail.
    
    # To be efficient, we will use a "step" that decreases.
    # If we find a solution for M, we set current_best_M = M.
    # If we fail for M, we try M-1?
    # This might be slow.
    
    # Let's use a time-boxed search for each M.
    # We will try M from current_best_M - 1 down to lower_bound.
    # We will spend a fixed amount of time on each M (e.g., 5s).
    
    # Actually, binary search is better if the feasibility is monotonic.
    # Feasibility is monotonic.
    # So we can binary search for the smallest M that is feasible.
    
    # But our heuristic `try_find_set` is not perfect. It might fail for a feasible M.
    # So we need to be careful.
    
    # Strategy:
    # 1. Set low = lower_bound, high = current_best_M
    # 2. While low < high and time < deadline:
    #    mid = (low + high) // 2
    #    Try to find a set for mid with a time limit.
    #    If found:
    #       high = mid
    #       Update best
    #    Else:
    #       low = mid + 1
    #       # If we fail, it might be infeasible, or our heuristic failed.
    #       # We should probably increase the time limit for mid if we are close to the boundary?
    
    # Let's implement this binary search.
    
    low = lower_bound
    high = current_best_M
    rng = random.Random(42)
    
    # We will allocate time budget.
    # Total time ~ 115s.
    # We can do ~ 20 iterations of binary search (log2(1000) ~ 10).
    # So we have ~ 10s per iteration.
    
    while low < high and time.time() < deadline - 5.0:
        mid = (low + high) // 2
        
        # Time limit for this M
        # As we get closer to the end, we should be more conservative?
        # Or just use a fixed time per iteration.
        time_limit = 5.0
        
        rows = try_find_set(mid, n_rows, k, rng, time_limit)
        
        if rows is not None:
            # Found a valid set with max entry <= mid
            # Check actual max
            actual_max = max(max(r) for r in rows)
            if actual_max < current_best_M:
                current_best_M = actual_max
                current_best_rows = [r[:] for r in rows]
            
            high = mid
        else:
            low = mid + 1
            
    return current_best_rows
# EVOLVE-BLOCK-END
