import time
import random
import sys
from itertools import combinations

def _compute_diffs(row):
    return frozenset(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))

def _is_valid(rows):
    all_diffs = set()
    for row in rows:
        diffs = _compute_diffs(row)
        if diffs & all_diffs:
            return False
        all_diffs |= diffs
    return True

def _get_scope(rows):
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, deadline):
    """
    Generate an initial set of rows using a greedy approach.
    """
    rows = []
    used_diffs = set()
    
    for _ in range(n_rows):
        if time.time() > deadline:
            break
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, 300):
                new_diffs = frozenset(x - a for a in row)
                if new_diffs & used_diffs:
                    continue
                row.append(x)
                used_diffs |= new_diffs
                current_max = x
                found = True
                break
            if not found:
                return None
        rows.append(row)
    return rows

def _backtrack_search(n_rows, k, M, deadline):
    """
    Attempt to find a solution with scope M using backtracking.
    Returns a list of rows if found, otherwise None.
    """
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, current_row):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        # We need to construct row_idx-th row
        # It must be strictly increasing, start at 0, end <= M
        # And all diffs must be distinct and not in used_diffs
        
        # Optimization: If we are at the last row, we can just check if we can fill it
        # But let's do it generally.
        
        # Heuristic: Try to fill the row greedily first?
        # Or backtrack on the elements.
        
        # To reduce symmetry, we can enforce that the maximum element of row i is <= maximum element of row i+1?
        # Not necessarily required, but helps pruning if we sort rows by max.
        # However, the problem doesn't require rows to be sorted.
        # Let's just try to find ANY valid row.
        
        # We need to pick k elements from 1..M such that:
        # 1. They are increasing.
        # 2. Diffs within the row are distinct.
        # 3. Diffs within the row are not in used_diffs.
        
        # This is equivalent to finding a Golomb ruler of length M with k+1 marks (including 0)
        # that avoids used_diffs.
        
        # Let's use a recursive helper to build the row.
        def build_row(pos, current_elements, local_used_diffs):
            if time.time() > deadline:
                return None
            
            if pos == k + 1:
                return current_elements[:]
            
            # Next element must be > last element
            # And <= M
            start_val = current_elements[-1] + 1
            for val in range(start_val, M + 1):
                new_diffs = frozenset(val - a for a in current_elements)
                if new_diffs & local_used_diffs:
                    continue
                if new_diffs & used_diffs:
                    continue
                
                current_elements.append(val)
                local_used_diffs |= new_diffs
                
                result = build_row(pos + 1, current_elements, local_used_diffs)
                if result is not None:
                    return result
                
                current_elements.pop()
                local_used_diffs -= new_diffs
            
            return None
        
        new_row = build_row(0, [0], set())
        if new_row is None:
            return False
        
        rows.append(new_row)
        used_diffs |= _compute_diffs(new_row)
        
        if backtrack(row_idx + 1, []):
            return True
        
        # Backtrack
        rows.pop()
        used_diffs -= _compute_diffs(new_row)
        return False
    
    if backtrack(0, []):
        return rows
    return None

def _sa_optimize(rows, k, deadline):
    """
    Simulated Annealing optimization to reduce the scope.
    This is a refined version that handles invalid states by penalizing collisions.
    """
    current_rows = [row[:] for row in rows]
    current_scope = _get_scope(current_rows)
    best_rows = [row[:] for row in current_rows]
    best_scope = current_scope
    
    def get_cost(rows):
        # Cost is primarily the number of collisions, secondarily the scope
        all_diffs = []
        max_val = 0
        for row in rows:
            max_val = max(max_val, max(row))
            diffs = _compute_diffs(row)
            # We need to count duplicates across all rows
            # This is expensive to compute fully every time.
            # Let's just return the scope for now and rely on validity checks?
            # No, SA needs a continuous cost.
            # Let's compute collisions.
            for d in diffs:
                all_diffs.append(d)
        
        unique_diffs = set(all_diffs)
        collisions = len(all_diffs) - len(unique_diffs)
        
        # Heavy penalty for collisions
        cost = collisions * 10000 + max_val
        return cost, max_val, collisions

    current_cost, _, _ = get_cost(current_rows)
    
    t = 50.0
    cooling = 0.999
    
    while time.time() < deadline and t > 0.1:
        # Pick a random row and a random element to modify
        idx = random.randint(0, len(current_rows) - 1)
        elem_idx = random.randint(1, k) # Indices 1..k (0 is fixed)
        
        old_val = current_rows[idx][elem_idx]
        
        # Determine range for new value
        # Must be > previous element and < next element (if exists)
        low = current_rows[idx][elem_idx - 1] + 1
        high = current_rows[idx][elem_idx + 1] - 1 if elem_idx < k else current_scope + 10
        
        if low > high:
            t *= cooling
            continue
            
        new_val = random.randint(low, high)
        if new_val == old_val:
            t *= cooling
            continue
            
        current_rows[idx][elem_idx] = new_val
        
        new_cost, new_scope, new_collisions = get_cost(current_rows)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / t):
            current_cost = new_cost
            current_scope = new_scope
            
            # If valid, update best
            if new_collisions == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = [row[:] for row in current_rows]
        else:
            current_rows[idx][elem_idx] = old_val
            
        t *= cooling
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Get initial solution via greedy
    rows = _greedy_init(n_rows, k, deadline)
    if rows is None:
        # Fallback: just return a trivially valid (but large) set
        # Greedy should always find one unless k is huge
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                found = False
                for x in range(current_max + 1, 1000):
                    new_diffs = frozenset(x - a for a in row)
                    if new_diffs & used_diffs:
                        continue
                    row.append(x)
                    used_diffs |= new_diffs
                    current_max = x
                    found = True
                    break
                if not found:
                    break
            rows.append(row)
        if len(rows) < n_rows:
            # Pad with large numbers to ensure validity (though they might conflict)
            # Better to just return what we have if we can't fill?
            # But the contract expects n_rows.
            # Let's assume greedy works for reasonable inputs.
            pass

    current_best = _get_scope(rows)
    best_rows = rows[:]

    # 2. Try to improve using Backtracking with decreasing M
    # Start from current best - 1 and go down
    # But backtracking is expensive. Let's try a few steps.
    if time.time() < deadline - 60: # Leave 60s for SA
        for M in range(current_best - 1, max(current_best - 20, 100), -1):
            if time.time() > deadline - 60:
                break
            
            # Try to find a solution with scope M
            # We can use a time-limited search
            search_deadline = time.time() + 5.0 # 5 seconds per M
            solution = _backtrack_search(n_rows, k, M, search_deadline)
            if solution is not None:
                current_best = M
                best_rows = solution
                # If we found a solution, we can try smaller M
                # But if we fail, we might have spent time.
                # If we succeed, continue to M-1
                continue
            else:
                # If we failed, we can break or try a random restart?
                # For now, break to save time for SA
                break

    # 3. Simulated Annealing to polish
    if time.time() < deadline - 5:
        # Use the best rows found so far
        polished_rows = _sa_optimize(best_rows, k, deadline - 2)
        polished_scope = _get_scope(polished_rows)
        
        # Verify validity of polished rows
        if _is_valid(polished_rows) and polished_scope < current_best:
            best_rows = polished_rows
            current_best = polished_scope
    
    return best_rows
