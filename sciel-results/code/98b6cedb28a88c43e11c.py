import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Evaluator limit is 120s (or 150s per prompt, but 120s is safer to assume for strictness).
    # Use a safety margin.
    deadline = start_time + 115.0
    
    # Total number of differences per row is k*(k+1)/2
    # Total differences required: n_rows * k*(k+1)/2
    # These must be distinct positive integers.
    # Min possible max difference (scope) is at least the count of diffs if they are 1..M.
    # Actually, the differences are values d = a_j - a_i.
    # If the largest entry is M, the max difference is M.
    # So we need M >= total_diffs.
    # Also, the elements are increasing, so the row itself consumes space.
    
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on scope M:
    # We need to fit 'total_diffs' distinct differences into [1, M].
    # So M >= total_diffs.
    # However, the structure of the row (increasing integers starting at 0) imposes further constraints.
    # For a single row of length k+1, the max diff is the last element.
    # The set of diffs for one row has size k(k+1)/2.
    # The max element of that row must be at least k(k+1)/2? No, not necessarily.
    # Example: k=2, row [0, 2, 3]. Diffs: 2, 3, 1. Max element 3. Diffs 1,2,3.
    # Example: k=3, row [0, 1, 4, 6]. Diffs: 1,4,6,3,5,2. Max element 6. Diffs 1..6.
    # It turns out that for a single row, the minimal max element is related to the Sidon set / Golomb ruler properties.
    # A Golomb ruler of order k+1 has distinct pairwise differences.
    # The minimal length of a Golomb ruler of order n is G(n).
    # Here we have multiple rows, and differences must be distinct ACROSS rows.
    # This is equivalent to finding n_rows disjoint Golomb rulers (in terms of their difference sets) 
    # packed into a common range [1, M].
    
    # Strategy:
    # 1. Determine a target scope M. Start with a lower bound and increase if search fails.
    # 2. Use randomized backtracking to assign elements to rows.
    
    # Lower bound for M:
    # Each row i has differences D_i. Union of all D_i is a subset of {1, ..., M}.
    # So M >= total_diffs.
    # Also, for a single row of length k+1, the max element m_i must satisfy m_i >= minimal Golomb ruler length for k+1 marks.
    # Let G(n) be minimal Golomb ruler length for n marks.
    # We need M >= max_i m_i.
    # So M >= max(total_diffs, G(k+1)).
    
    # Approximation for G(n): n^2/4 + n/2 + 1? 
    # For small n, we can hardcode or estimate.
    # G(1)=0, G(2)=1, G(3)=3, G(4)=6, G(5)=11, G(6)=17, G(7)=25, G(8)=34, G(9)=44, G(10)=55
    # Formula approx: n^2/4.
    
    # Let's start with a reasonable M.
    # total_diffs = n_rows * k*(k+1)/2
    # If n_rows is large, total_diffs dominates.
    # If k is large, G(k+1) ~ k^2/4.
    
    # Let's try to solve for a specific M.
    # We will increment M until we find a solution.
    
    # To make backtracking efficient, we represent the state as:
    # rows: list of lists, partially filled.
    # used_diffs: set of all differences used so far.
    
    # Variable ordering: 
    # Fill rows one by one? Or element by element across rows?
    # Filling a row requires ensuring its internal differences are unique and don't clash with used_diffs.
    # Within a row, we must form a Golomb ruler subset of the remaining available differences?
    # No, the differences are derived from the positions.
    
    # Backtracking on a single row:
    # Given a set of available differences (or just a set of used diffs to avoid),
    # can we construct a row of length k+1 starting at 0 with max element <= M?
    
    # Let's define a helper: can_construct_row(k, M, used_diffs)
    # Returns a valid row or None.
    # This is a CSP itself.
    # Variables: x_1, ..., x_k (x_0 = 0).
    # Constraints:
    # 1. 0 < x_1 < ... < x_k <= M
    # 2. For all i < j, x_j - x_i not in used_diffs.
    # 3. For all i < j < l, (x_j - x_i) != (x_l - x_j) etc. (Internal uniqueness).
    #    Actually, condition 2 with a local set handles internal uniqueness if we add diffs to local set.
    
    # Algorithm for can_construct_row:
    # Backtrack on x_1, ..., x_k.
    # Forward checking: when choosing x_i, check if it creates a diff already in used_diffs OR in local_diffs.
    # Also, we must ensure that it is *possible* to complete the row.
    # A simple heuristic: pick the next x_i from the smallest available values that don't conflict.
    
    # Global Search:
    # We have n_rows rows.
    # We can solve them sequentially.
    # Row 1: Find a row using diffs from [1, M]. Update used_diffs.
    # Row 2: Find a row using diffs from [1, M] \ used_diffs.
    # ...
    # This is prone to getting stuck if Row 1 uses "bad" diffs.
    # Randomization: Randomly permute the order of rows? Or randomize the choices within the row construction.
    
    # To improve success rate, we can use "Look-Ahead" or restarts.
    # Given the time limit, we will do multiple restarts with increasing M if necessary, 
    # but likely we can find a solution with M close to the bound.
    
    # Let's refine the row construction to be fast.
    # k is likely small (e.g., 4, 5, 8). The prompt says "4 rows of 8 increasing integers" -> k=7? 
    # "8 increasing integers" means length 8, so k+1=8 => k=7.
    # n_rows = 4.
    # Total diffs = 4 * (7*8/2) = 4 * 28 = 112.
    # So M must be at least 112.
    # The prompt says "simple counting bound is 112" and "best known set has largest entry 134".
    # So we are looking for M between 112 and 134.
    
    # If the task parameters are fixed (n_rows=4, k=7), we can hardcode a known good solution or search efficiently.
    # But the function signature implies generic n_rows, k.
    # We must handle generic cases, but optimize for the likely case.
    
    # Let's implement the backtracking for a single row.
    
    def construct_row(k, M, forbidden_diffs):
        """
        Constructs a row of length k+1 starting at 0, max element <= M.
        All pairwise differences must be distinct and not in forbidden_diffs.
        Returns the row as a list, or None if not possible.
        """
        row = [0]
        local_diffs = set()
        
        def backtrack(idx):
            if idx == k:
                return True
            
            # Heuristic: Try smallest available x first? Or random?
            # To find *a* solution, smallest is good for minimizing scope, 
            # but we are fixing M. So we just need *any* valid x.
            # Trying smallest first might lead to dead ends sooner? 
            # Actually, for Golomb rulers, spreading out is often better.
            # Let's try a range of values.
            
            # Lower bound for next element:
            # Must be > row[-1].
            # Also, x - row[i] must not be in forbidden_diffs or local_diffs.
            
            # Upper bound for next element:
            # x <= M.
            # Also, we need to be able to fit (k - idx) more elements after x.
            # Minimal spacing is 1. So x <= M - (k - idx).
            
            start_val = row[-1] + 1
            end_val = M - (k - idx)
            
            if start_val > end_val:
                return False
            
            # Randomize the order of trying values to avoid local traps?
            # For a single row, deterministic greedy (smallest) is usually fine if M is large enough.
            # But if M is tight, we might need to search.
            # Let's iterate from start_val to end_val.
            
            # Optimization: Pre-filter candidates?
            # A candidate x is valid if:
            # 1. x - row[i] not in forbidden_diffs for all i < idx
            # 2. x - row[i] not in local_diffs for all i < idx
            # 3. x - row[i] != x - row[j] (impossible since row[i] != row[j])
            
            # Let's just loop.
            for x in range(start_val, end_val + 1):
                # Check conflicts
                valid = True
                new_diffs = []
                for i in range(idx):
                    d = x - row[i]
                    if d in forbidden_diffs or d in local_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # If valid, place x
                row.append(x)
                local_diffs.update(new_diffs)
                
                if backtrack(idx + 1):
                    return True
                
                # Backtrack
                row.pop()
                local_diffs.difference_update(new_diffs)
            
            return False
        
        return backtrack(1) if backtrack(1) else None

    # Main Loop
    best_rows = None
    current_M = total_diffs
    
    # We might need to go higher than total_diffs.
    # Max M to try? 
    # If k=7, n_rows=4, total_diffs=112. Best known 134.
    # Let's try up to total_diffs + 50 or so.
    
    # To speed up, we can randomize the order in which we pick rows? 
    # No, rows are symmetric. 
    # But the *specific* set of diffs used by row 1 affects row 2.
    # Randomizing the choice of valid x in construct_row can help.
    
    # Let's modify construct_row to accept a random seed or use global random.
    # We will add a parameter to randomize the search order in construct_row.
    
    def construct_row_random(k, M, forbidden_diffs, rng):
        row = [0]
        local_diffs = set()
        
        def backtrack(idx):
            if idx == k:
                return True
            
            start_val = row[-1] + 1
            end_val = M - (k - idx)
            
            if start_val > end_val:
                return False
            
            # Create list of valid candidates
            candidates = []
            for x in range(start_val, end_val + 1):
                valid = True
                new_diffs = []
                for i in range(idx):
                    d = x - row[i]
                    if d in forbidden_diffs or d in local_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
            
            # Shuffle candidates to explore different branches
            rng.shuffle(candidates)
            
            for x, new_diffs in candidates:
                row.append(x)
                local_diffs.update(new_diffs)
                
                if backtrack(idx + 1):
                    return True
                
                row.pop()
                local_diffs.difference_update(new_diffs)
            
            return False
        
        if backtrack(1):
            return row
        return None

    # Try increasing M
    max_M_to_try = total_diffs + 100 # Generous upper bound for small cases
    if n_rows > 4 or k > 7:
        max_M_to_try = total_diffs + 200
    
    # If we know the specific case (4 rows, 8 elements), we can be more precise.
    # But let's keep it generic.
    
    # We will perform multiple restarts for each M to escape local minima in the row construction.
    
    found = False
    for M in range(current_M, max_M_to_try + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Try a few restarts for this M
        for attempt in range(20):
            if time.time() > deadline - 2.0:
                break
                
            rng = random.Random()
            used_diffs = set()
            rows = []
            success = True
            
            for i in range(n_rows):
                # Randomly choose which row to build next? 
                # No, we build 0 to n_rows-1.
                # But we can randomize the internal choices.
                
                # Optimization: If we get stuck, we might want to restart the whole thing.
                # For now, sequential construction with randomized row generator.
                
                row = construct_row_random(k, M, used_diffs, rng)
                if row is None:
                    success = False
                    break
                
                rows.append(row)
                # Add all diffs from this row to used_diffs
                for j in range(k + 1):
                    for l in range(j + 1, k + 1):
                        used_diffs.add(row[l] - row[j])
            
            if success:
                # We found a valid set with scope M
                best_rows = rows
                found = True
                break # Break out of attempt loop
                
        if found:
            break
            
    if not found:
        # Fallback: Return a greedy solution if we couldn't find a better one in time?
        # The greedy solution from the seed program is a safe fallback.
        # Let's implement a fast greedy fallback to ensure we return *something* valid.
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        best_rows = rows

    return best_rows
