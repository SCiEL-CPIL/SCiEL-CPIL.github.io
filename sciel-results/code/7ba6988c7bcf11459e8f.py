import time
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry using a backtracking search over candidate rows.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer

    def get_valid_rows(max_val):
        """
        Generates all valid rows (starting at 0, strictly increasing, distinct internal diffs)
        with max element <= max_val.
        Returns a list of tuples (row_tuple, diff_set).
        """
        rows = []
        
        def gen(prefix, current_max, used_diffs):
            if len(prefix) == k + 1:
                # Calculate full diff set for storage
                full_diffs = set()
                for i in range(len(prefix)):
                    for j in range(i+1, len(prefix)):
                        full_diffs.add(prefix[j] - prefix[i])
                rows.append((tuple(prefix), full_diffs))
                return
            
            last = prefix[-1]
            # Next element x must be > last and <= max_val
            # Optimization: If we need (k+1 - len(prefix)) more elements,
            # and the max gap allowed is limited by max_val, we can bound x.
            # But simple range is fine for small k.
            for x in range(last + 1, max_val + 1):
                # Check new diffs
                new_diffs = set()
                valid = True
                for p in prefix:
                    d = x - p
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    gen(prefix + [x], x, used_diffs | new_diffs)
        
        gen([0], 0, set())
        return rows

    def find_solution(candidates, n_rows):
        """
        Finds a subset of n_rows candidates with disjoint diff sets,
        minimizing the max element of the selected rows.
        candidates: list of (row_tuple, diff_set)
        """
        if not candidates:
            return None
        
        # Sort by max element (last element of row)
        candidates.sort(key=lambda x: x[0][-1])
        
        best = None
        best_max = float('inf')
        
        # Pre-compute lengths
        n_cand = len(candidates)
        
        def backtrack(idx, selected, used_diffs, current_max):
            nonlocal best, best_max
            
            if len(selected) == n_rows:
                if current_max < best_max:
                    best_max = current_max
                    best = selected[:]
                return
            
            # Pruning 1: If we can't fit enough rows, return
            if idx + (n_rows - len(selected)) > n_cand:
                return
            
            # Pruning 2: Minimum possible max value
            # The smallest max value we can achieve with the remaining picks is the max value of the
            # (n_rows - len(selected))-th smallest candidate from idx onwards.
            # Since sorted, this is candidates[idx + (n_rows - len(selected) - 1)][0][-1]
            min_possible_max = candidates[idx + (n_rows - len(selected) - 1)][0][-1]
            if min_possible_max >= best_max:
                return

            # Try each candidate from idx
            for i in range(idx, n_cand):
                row, row_diffs = candidates[i]
                
                # Check conflict
                if used_diffs & row_diffs:
                    continue
                
                new_max = max(current_max, row[-1])
                
                # Pruning 3: If this branch already exceeds best, skip
                if new_max >= best_max:
                    continue

                new_used = used_diffs | row_diffs
                backtrack(i + 1, selected + [row], new_used, new_max)
                
                # If we found a solution, we could potentially break early if we are
                # looking for ANY solution, but we want the MIN MAX.
                # However, if best_max is updated, the pruning in subsequent loops will handle it.
        
        backtrack(0, [], set(), 0)
        
        if best:
            return best
        return None

    def greedy_fallback():
        """
        Simple randomized greedy fallback if exact search fails or times out.
        """
        import random
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        return rows

    # Main Logic
    # We want to find the minimum max_val.
    # Iterate max_val from a lower bound upwards.
    # Lower bound: simple counting bound or just start at 100.
    # For k=5, n=7, best is 112.
    # Let's start at 105 and go up to 130.
    
    # To be safe and general, we can estimate a start.
    # But for the specific task (7,5), 105-120 is the range.
    # If n_rows, k are different, we need a dynamic range.
    # Let's use a heuristic start.
    
    # Estimate: (n_rows * k * (k+1) / 2) ** 0.5 * 2 ?
    # Let's just start at 50 and go up by 5s? Or 1s?
    # 1s is safer for optimality but slower.
    # Let's try 1s increments.
    
    # However, generating rows for each M is redundant.
    # We can generate rows up to a high limit (e.g., 150) once,
    # then filter them for each M?
    # Or just generate up to M each time.
    # Generating up to 120 is fast.
    
    best_result = None
    
    # Dynamic range selection
    # If k is small, start low.
    # Let's start at 10 * k * n_rows? No.
    # Let's start at 100 for now, as it's a specific task hint.
    # To be robust:
    start_m = 50
    end_m = 200
    
    # Check time
    if time.time() - start_time > time_limit - 10:
        return greedy_fallback()
    
    for m in range(start_m, end_m + 1):
        if time.time() - start_time > time_limit - 5:
            break
            
        candidates = get_valid_rows(m)
        
        if not candidates:
            continue
            
        # Optimization: If we have fewer candidates than n_rows, skip
        if len(candidates) < n_rows:
            continue
            
        result = find_solution(candidates, n_rows)
        
        if result:
            # We found a solution with max <= m.
            # Since we are iterating m upwards, this is the optimal max value
            # (assuming find_solution finds the min max within the pool).
            # Note: find_solution returns the min max *within the pool*.
            # So if result max is 110, and we are at m=112, the answer is 110.
            # We can return immediately.
            return [list(r) for r in result]
            
    # If no solution found in the range, use fallback
    return greedy_fallback()
