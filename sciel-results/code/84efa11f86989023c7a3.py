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
    
    # Bitset-based solver using Python integers as bitsets
    # Bit i is set if difference i is used
    
    def solve_csp_random(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and randomized backtracking.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        rng = random.Random(seed)
        
        # Precompute masks for quick checking? 
        # We need to check if any difference in {x-a for a in row} is in used_bits.
        
        def count_valid_next(row, last, remaining, used_bits, max_entry):
            """
            Heuristic: Count how many valid candidates for the next element exist.
            This helps in choosing the 'best' current element by looking ahead one step.
            """
            if remaining == 1:
                # We just need to check if we can place the last element.
                # The last element y must be > last and <= max_entry.
                # And y - a must not be in used_bits for all a in row.
                # We don't need to count, just return 1 if possible, 0 otherwise.
                # But for heuristic, we want to know if it's "easy" or "hard".
                # If remaining == 1, the choice of 'last' doesn't matter for this count,
                # but the value of 'last' determines the range of y.
                # Actually, this function is called to evaluate a candidate 'x' for the CURRENT position.
                # So 'row' includes 'x'. 'last' is 'x'. 'remaining' is k+1 - len(row).
                # We want to see if we can easily fill the rest.
                
                # For now, a simple count of valid next values is sufficient.
                count = 0
                min_y = last + 1
                max_y = max_entry - (remaining - 1)
                
                if min_y > max_y:
                    return 0
                
                # To speed up, we can sample or just count all if range is small.
                # Range is at most max_entry.
                for y in range(min_y, max_y + 1):
                    conflict = False
                    for a in row:
                        d = y - a
                        if (used_bits >> d) & 1:
                            conflict = True
                            break
                    if not conflict:
                        count += 1
                        # Early exit if count is high? 
                        # If count > some threshold, it's "easy".
                        if count > 50: 
                            return count
                return count

        def build_row(current_row, used_bits, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            candidates = []
            for x in range(min_x, max_x + 1):
                # Compute new differences as bitset
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                candidates.append((x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Heuristic: Sort candidates by the number of valid options for the NEXT step.
            # We want to pick x that leaves the most flexibility for the rest of the row.
            # This is a "Least Constraining Value" (LCV) heuristic.
            
            # To save time, we only do this look-ahead if the number of candidates is manageable
            # or if we are deep in the search.
            
            scored_candidates = []
            for x, new_diff_bits in candidates:
                # Simulate adding x
                test_row = current_row + [x]
                test_used = used_bits | new_diff_bits
                
                # Look ahead: how many valid y's are there for the next position?
                # If remaining - 1 == 0, we are done, so score is high.
                if remaining == 1:
                    score = 1000
                else:
                    score = count_valid_next(test_row, x, remaining - 1, test_used, max_entry)
                
                scored_candidates.append((score, x, new_diff_bits))
            
            # Sort by score descending (most flexible first)
            scored_candidates.sort(key=lambda t: t[0], reverse=True)
            
            # To add randomness, we can pick from the top few or shuffle the top k
            # But for CSP, LCV is strong. Let's stick to it but maybe randomize ties.
            
            # If there are many candidates, we might want to limit the look-ahead cost.
            # For now, we assume k is small enough (k=7 => 8 elements, 7 diffs).
            # The loop in count_valid_next is O(k * range).
            
            for score, x, new_diff_bits in scored_candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        # Try to build all rows
        # We can try to randomize the order of rows if we had different structures, 
        # but here they are identical. 
        # However, the state of 'used_bits' changes.
        # Building rows in a fixed order is standard.
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline):
                rows.append(current_row)
                # Update used_bits
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows

    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try max_entry values from lb upward
    search_upper_bound = greedy_max
    
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2  # leave 2 sec safety
        
        if remaining_time < 0.5:
            break
        
        # Allocate time
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 10:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
            
        # Try a few random seeds for each max_entry
        found = False
        num_restarts = max(1, int(budget / 1.0))
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            
            # Give each restart a slice of the budget
            restart_budget = budget / (num_restarts - seed) if seed < num_restarts - 1 else budget - seed * (budget/num_restarts)
            restart_budget = min(1.5, restart_budget) # Cap per restart to allow more restarts
            
            result = solve_csp_random(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
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
