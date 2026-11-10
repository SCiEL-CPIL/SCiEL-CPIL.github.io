import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: 150s limit, keep 5s buffer
    deadline = start_time + 145
    
    # Total number of distinct differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry M: M(M+1)/2 >= total_diffs
    # M ~ sqrt(2 * total_diffs)
    lower_bound = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))
    
    # Heuristic upper bound: usually slightly above lower bound
    # For (4,7), total_diffs = 4*28=112. LB = ceil((sqrt(1+896)-1)/2) = ceil((30.1-1)/2) = 15.
    # Wait, the task says simple counting bound is 112? No, "simple counting bound is 112" might refer to something else or I am misinterpreting.
    # Let's re-read: "simple counting bound is 112". 
    # If M=112, diffs available = 112*113/2 = 6328. Total needed = 112. That's huge margin.
    # Ah, the task text says: "The best known set has largest entry 134... simple counting bound is 112."
    # This implies the bound on the *largest entry* is 112?
    # Let's check: If M=112, max diff is 112. Number of positive integers <= 112 is 112.
    # We need 112 distinct differences. So we must use ALL integers from 1 to 112.
    # This is a "perfect difference set" scenario.
    # If we can pack them perfectly, M=112. If not, M > 112.
    
    # The current program found a score of 1.5299, which corresponds to a value.
    # Score = seed / value. Seed is likely 1.0? No, "1.0 equals the seed program".
    # If score is 1.5299, and seed is 1, value = 1/1.5299 = 0.65? No.
    # "Score: the scope... minimized... search score is seed/value".
    # Usually seed is the value of the seed program.
    # If seed program value is V_seed, score = V_seed / V_current.
    # If score > 1, V_current < V_seed.
    # So 1.5299 = V_seed / V_current.
    # We don't know V_seed exactly, but we want to minimize V_current.
    
    # Strategy:
    # 1. Try to construct a solution with max entry M starting from lower_bound.
    # 2. Use randomized backtracking with MRV (Minimum Remaining Values) heuristic.
    # 3. If we find a solution, try to improve it with local search.
    
    best_solution = None
    best_scope = float('inf')
    
    def validate(rows):
        if not rows:
            return False
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
        
    def get_scope(rows):
        return max(max(row) for row in rows)

    # CSP Solver:
    # Variables: (row_idx, col_idx) for row_idx in [0, n_rows), col_idx in [1, k]
    # Domain: integers > previous element
    # Constraint: all differences distinct.
    
    # To make backtracking efficient, we maintain a global set of used differences.
    # We build rows one by one. Within a row, we build left to right.
    # Heuristic: When choosing the next element for a row, pick the smallest valid one?
    # Or pick randomly from valid ones to explore different structures?
    
    def solve_with_backtracking(max_scope, seed):
        rng = random.Random(seed)
        used_diffs = set()
        rows = []
        
        # We will build rows sequentially.
        # To add diversity, we can randomize the order of rows or the choices within a row.
        
        def backtrack(row_idx, col_idx, current_row):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                # All rows built
                nonlocal best_solution, best_scope
                # This is a valid solution with max entry <= max_scope
                # We want to minimize the actual max entry, which is max(rows[-1][-1])
                current_scope = max(max(r) for r in rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_solution = [list(r) for r in rows]
                return True
            
            if col_idx == k + 1:
                # Current row complete
                rows.append(list(current_row))
                return backtrack(row_idx + 1, 1, [0])
            
            last_val = current_row[col_idx - 1] if col_idx > 0 else -1
            
            # Determine possible values for current_row[col_idx]
            # Must be > last_val
            # Must be <= max_scope
            # All differences with previous elements in current_row must be unused
            
            # To avoid iterating through too many values, we can bound the search.
            # If we are at col_idx, we have k - col_idx + 1 elements left to place.
            # The maximum possible value is constrained by max_scope.
            
            # Optimization: Instead of trying all x from last_val+1 to max_scope,
            # we can generate candidates.
            
            # For simplicity and robustness, we iterate, but with a limit on backtracks.
            # To make it more "search-like" and less "greedy-like", we shuffle candidates.
            
            # However, for small k and n_rows, the domain is small.
            # Let's try values in a randomized order, but biased towards smaller values
            # to keep the scope tight, but occasionally try larger ones to escape local minima.
            
            # Actually, a better heuristic for difference sets is to try to pack them tightly.
            # So we prefer small values.
            
            # Let's collect valid candidates up to a certain limit or all valid ones?
            # If max_scope is close to lower_bound, the number of candidates is small.
            
            candidates = []
            x = last_val + 1
            while x <= max_scope:
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & used_diffs):
                    candidates.append(x)
                x += 1
                if len(candidates) > 50: # Limit candidate list size for performance
                    break
            
            if not candidates:
                return False
            
            # Shuffle to introduce randomness, but keep small values more likely?
            # Weighted shuffle: smaller values have higher weight.
            # Or just random sample.
            rng.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                # Tentatively add
                used_diffs.update(new_diffs)
                current_row.append(x)
                
                if backtrack(row_idx, col_idx + 1, current_row):
                    return True
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        # Start with first row
        # To add diversity between restarts, we can randomize the initial choice of the first element?
        # No, first element is 0.
        # We can randomize the order in which we build rows?
        # But rows are indistinguishable in terms of constraints, except for the "scope" which is global max.
        # So order doesn't matter for validity, but might matter for search efficiency.
        
        return backtrack(0, 1, [0])

    # Local Search Refinement
    def local_search(rows, iterations=1000):
        """
        Try to reduce the scope by moving elements.
        """
        if not rows:
            return rows
        
        rows = [list(r) for r in rows]
        current_scope = get_scope(rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        
        # Precompute diff counts for fast update
        # diff_count[d] = number of times difference d appears
        diff_count = {}
        
        def update_diffs(row_idx, pos, old_val, new_val, sign):
            """
            sign = +1 to add diffs, -1 to remove
            """
            row = rows[row_idx]
            for i in range(len(row)):
                if i == pos:
                    continue
                d = abs(row[i] - (old_val if sign == -1 else new_val))
                # Actually, we need to remove old and add new carefully.
                # Let's do it explicitly outside.
                pass
        
        # Simpler: Recalculate diff counts? O(N * K^2).
        # For K=7, N=4, N*K^2 = 4*49 = 196. Very fast.
        # We can afford to recalculate validity on each move.
        
        rng = random.Random(12345)
        
        for it in range(iterations):
            if time.time() > deadline:
                break
                
            # Pick random row and position
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k) # Don't move 0
            
            old_val = rows[r_idx][pos]
            
            # Determine valid range for new_val
            lower = rows[r_idx][pos - 1] + 1
            upper = rows[r_idx][pos + 1] - 1 if pos < k else current_scope + 10 # Allow expansion slightly
            
            # Try to find a new_val that is valid and potentially reduces scope
            # We prefer new_val < old_val if old_val is contributing to scope.
            
            # Generate candidates
            candidates = set()
            # Try nearby values
            for delta in range(-5, 6):
                nv = old_val + delta
                if lower <= nv <= upper:
                    candidates.add(nv)
            
            # Try random values in range
            if upper > lower:
                for _ in range(5):
                    candidates.add(rng.randint(lower, upper))
            
            # Try to minimize scope: if old_val is the max, try smaller values
            if old_val == max(rows[r_idx]):
                candidates.update(range(lower, old_val))
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                
                # Check validity
                # 1. Check local constraints (strictly increasing)
                if new_val <= rows[r_idx][pos - 1]:
                    continue
                if pos < k and new_val >= rows[r_idx][pos + 1]:
                    continue
                
                # 2. Check global difference constraints
                # Remove old diffs involving pos
                # Add new diffs involving pos
                # Check for conflicts
                
                # To be efficient, we can just validate the whole thing.
                # With N=4, K=7, validation is fast.
                
                # Temporarily change
                rows[r_idx][pos] = new_val
                
                if validate(rows):
                    new_scope = get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    # Accept move (hill climbing)
                    current_scope = new_scope
                    break # Move to next iteration
                else:
                    # Revert
                    rows[r_idx][pos] = old_val
        
        return best_rows

    # Main Loop
    
    # 1. Find an initial solution using backtracking
    # Start with a generous max_scope and tighten?
    # Or start with lower_bound and increase if no solution found.
    
    current_max_scope = lower_bound
    
    while time.time() < deadline - 10:
        # Try to solve with current_max_scope
        # Use multiple random seeds for backtracking
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            # Reset best_solution for this scope attempt?
            # No, we want the global best.
            # But backtrack returns True if it finds ANY solution within max_scope.
            # It updates best_solution if it finds a better one.
            
            # We need to ensure backtrack doesn't stop at the first solution if we want to find the min scope.
            # But if we set max_scope = lower_bound, any solution found is optimal (or close).
            # If we set max_scope higher, we might find a suboptimal one first.
            
            # Strategy:
            # If current_max_scope == lower_bound, we are looking for the optimal solution.
            # If we find one, we can stop searching for lower scopes (impossible).
            # We should then try to improve it via local search? No, it's already minimal scope.
            
            # If current_max_scope > lower_bound, we are just trying to find *a* solution.
            
            solve_with_backtracking(current_max_scope, seed)
            
            if best_solution is not None:
                found = True
                # If we found a solution, try to improve it with local search
                # But only if it's not already at lower_bound
                if best_scope > lower_bound:
                    # Run local search
                    refined = local_search(best_solution, iterations=500)
                    if refined:
                        refined_scope = get_scope(refined)
                        if refined_scope < best_scope:
                            best_scope = refined_scope
                            best_solution = refined
                            if best_scope == lower_bound:
                                return best_solution
                else:
                    return best_solution
                break
        
        if found:
            # We have a solution. Try to tighten the scope by reducing max_scope?
            # But the solution found has scope best_scope.
            # We can try to find a solution with max_scope = best_scope - 1.
            if best_scope > lower_bound:
                current_max_scope = best_scope - 1
            else:
                break
        else:
            # No solution found with current_max_scope
            # Increase max_scope
            current_max_scope += 1
            # If we go too high, we might be inefficient.
            # Cap it?
            if current_max_scope > lower_bound + 50:
                break

    # Fallback: if no solution found via CSP, use greedy
    if best_solution is None:
        def greedy_deterministic():
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                rows.append(row)
            return rows
        
        det_rows = greedy_deterministic()
        if det_rows and validate(det_rows):
            best_solution = det_rows
            best_scope = get_scope(det_rows)
            
            # Try to improve with local search
            if time.time() < deadline - 5:
                refined = local_search(best_solution, iterations=1000)
                if refined:
                    refined_scope = get_scope(refined)
                    if refined_scope < best_scope:
                        best_solution = refined
                        best_scope = refined_scope

    return best_solution

