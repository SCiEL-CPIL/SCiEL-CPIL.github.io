# EVOLVE-BLOCK-START
import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a randomized CSP approach with bitset-based difference constraint propagation.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: We need 'total_diffs_needed' distinct differences.
    # The differences are integers from 1 to M.
    # So M must be at least total_diffs_needed.
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

    def solve_csp(n_rows, k, max_entry, timeout, seed):
        """
        CSP solver with randomized backtracking and bitset difference tracking.
        """
        deadline = time.time() + timeout
        used_bits = 0
        rows = []
        rng = random.Random(seed)
        
        # Precompute powers of 2 for bit operations? Python ints are arbitrary precision, so 1 << d is fine.
        
        def get_valid_candidates(current_row, used_bits, max_entry):
            """
            Returns a list of valid next values for the current row.
            """
            if len(current_row) == k + 1:
                return []
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # We need to pick x such that:
            # 1. x > last
            # 2. x <= max_entry
            # 3. There is enough room for 'remaining - 1' more elements after x.
            #    Min gap is 1, so x <= max_entry - (remaining - 1)
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return []
            
            candidates = []
            # Optimization: Iterate through potential x values.
            # For each x, check if differences x - a_i are not in used_bits.
            for x in range(min_x, max_x + 1):
                conflict = False
                # Check against current row elements
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                
                if not conflict:
                    # Additional heuristic: Check if this choice leaves enough "difference space"
                    # This is hard to check exactly without deeper search, so we rely on backtracking.
                    candidates.append(x)
            
            return candidates

        def backtrack(current_row, used_bits):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            candidates = get_valid_candidates(current_row, used_bits, max_entry)
            if not candidates:
                return False
            
            # Heuristic: Shuffle to randomize order
            rng.shuffle(candidates)
            
            # Optional: Prioritize candidates that use "rare" differences?
            # Or just random.
            
            for x in candidates:
                if time.time() > deadline:
                    return False
                
                # Compute new used bits
                new_diff_bits = 0
                for a in current_row:
                    d = x - a
                    new_diff_bits |= (1 << d)
                
                new_used = used_bits | new_diff_bits
                
                current_row.append(x)
                if backtrack(current_row, new_used):
                    return True
                current_row.pop()
            
            return False

        # Build rows one by one
        for _ in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if backtrack(current_row, used_bits):
                rows.append(current_row)
                # Update used_bits with the differences of the completed row
                # Note: backtrack already updated used_bits locally, but we need to persist it for the next row.
                # In the current structure, used_bits is a local variable in solve_csp.
                # We need to accumulate it.
                # Let's restructure: backtrack returns the new used_bits if successful?
                # Or just re-calculate from rows? Re-calculating is O(N^2) which is small.
                # But passing it is cleaner.
                # Let's modify backtrack to return (success, new_used_bits)
                # Actually, the current implementation of backtrack modifies `used_bits` passed by reference?
                # No, integers are immutable. `new_used` is a new int.
                # So `backtrack` needs to return the final `used_bits` if successful.
                
                # Let's rewrite backtrack to return the final used_bits.
                
                # Wait, the current code above doesn't do that. I will rewrite the whole function in the edit.
                pass
        
        return rows

    # We need a better solver. Let's write one that returns the used_bits.
    
    def solve_csp_v2(n_rows, k, max_entry, timeout, seed):
        deadline = time.time() + timeout
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        
        def backtrack(current_row, used_bits):
            if time.time() > deadline:
                return False, used_bits
            
            if len(current_row) == k + 1:
                return True, used_bits
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False, used_bits
            
            candidates = []
            for x in range(min_x, max_x + 1):
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
            
            if not candidates:
                return False, used_bits
            
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    return False, used_bits
                
                new_diff_bits = 0
                for a in current_row:
                    d = x - a
                    new_diff_bits |= (1 << d)
                
                new_used = used_bits | new_diff_bits
                
                current_row.append(x)
                success, final_used = backtrack(current_row, new_used)
                if success:
                    return True, final_used
                current_row.pop()
            
            return False, used_bits

        for _ in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            success, used_bits = backtrack(current_row, used_bits)
            if not success:
                return None
            rows.append(current_row)
        
        return rows

    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy:
    # 1. Try to find a solution with max_entry < greedy_max.
    # 2. We iterate M from a heuristic lower bound up to greedy_max - 1.
    # 3. If we find one, we update best and continue to try for even smaller M?
    #    Or just take the first one found?
    #    Since we want to minimize, we should iterate M from low to high.
    #    But low is 112, high is greedy_max (likely ~150-200).
    #    Searching 112-133 might take forever if infeasible.
    #    Better: Start from greedy_max - 1 and go down?
    #    If we find a solution at M, is it optimal? No, maybe M-1 is possible.
    #    But checking M-1 is also expensive.
    #    Given time limit, we should focus on the range likely to contain the optimum.
    #    Known best is 134. Greedy is likely > 134.
    #    Let's try M in [lb, greedy_max].
    
    # To save time, we can sample M values or use a binary search-like approach?
    # No, CSP is not monotonic in a simple way for search time.
    
    # Let's allocate time to different M values.
    # We'll try a few M values close to the expected optimum.
    
    # Heuristic: The optimum is usually close to the counting bound but higher.
    # Let's try M from lb to greedy_max.
    
    # To avoid spending all time on infeasible small M, we limit the time per M.
    
    # We will try M values in increasing order.
    # If we find a solution for M, we record it.
    # We can then try M-1?
    # Or we can just stop? No, we want the minimum.
    # But if we find a solution for M=134, and we started from 112, we must have tried 112-133.
    # If 112-133 are infeasible, the solver will fail quickly (hopefully).
    
    # Let's optimize the failure detection.
    # If we fail to find a solution for M, we move to M+1.
    
    # We will use random restarts for each M.
    
    search_start = lb
    search_end = greedy_max
    
    # If search space is large, we might want to skip some M?
    # No, we must check all to be sure.
    # But we can limit time per M.
    
    total_time_budget = time_limit - 5.0
    num_m_values = search_end - search_start + 1
    time_per_m = total_time_budget / num_m_values if num_m_values > 0 else total_time_budget
    
    # If time_per_m is too small, we won't make progress.
    # We can prioritize M values closer to the known best (134) or greedy_max.
    
    # Let's create a list of M values to try, prioritizing those closer to the "sweet spot".
    # Sweet spot is often around 1.2 * lb.
    # lb = 112. 1.2 * 112 = 134.
    
    # We'll try M in increasing order, but give more time to M values in the middle of the range.
    
    m_values = list(range(search_start, search_end + 1))
    
    # Shuffle? No, we want to find the smallest M.
    # But if small M are infeasible, we waste time.
    # Let's try a mix:
    # 1. Try M near 134 (known best) first to secure a good score.
    # 2. Then try smaller M to see if we can beat it.
    
    # Actually, the score is seed/value. Lower value is better.
    # So we MUST find the smallest possible M.
    # We cannot just find *a* solution.
    
    # So we must iterate M from lb upwards.
    # To speed up infeasible cases, we need fast failure.
    
    for M in range(search_start, search_end + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2.0
        
        if remaining_time < 0.5:
            break
        
        # Allocate time for this M
        # If we are far from lb, we have less hope, but we must check.
        # Let's give a fixed small budget to each M, and if we find a solution, we stop.
        # Wait, if we find a solution for M, is it the global minimum?
        # Yes, because we are iterating from lb upwards.
        # So if we find a solution for M, we can return it immediately?
        # No, we should validate it.
        # And we should check if we can find a *smaller* M?
        # No, we started from lb. If we found one at M, all M' < M failed.
        # So M is the minimum.
        # So we can break!
        
        # So the algorithm is:
        # for M in range(lb, greedy_max + 1):
        #    if solve(M): return M
        
        # The risk is that solve(M) takes too long for infeasible M.
        
        # Let's give each M a budget of 1-2 seconds.
        budget_for_m = min(2.0, remaining_time)
        
        # Try a few random seeds for this M
        found = False
        num_restarts = max(1, int(budget_for_m / 0.5))
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            restart_budget = budget_for_m / (num_restarts - seed)
            
            result = solve_csp_v2(n_rows, k, M, restart_budget, seed)
            if result is not None:
                if validate(result, n_rows, k):
                    best_rows = result
                    best_max = M
                    found = True
                    break
        
        if found:
            break
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
