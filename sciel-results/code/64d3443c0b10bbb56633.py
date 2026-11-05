"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    Uses a CSP approach with bitset-based difference constraint propagation
    and dynamic variable ordering to find tight solutions, plus a simulated
    annealing polish pass.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: m*(m+1)/2 >= total_diffs_needed
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
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
    time_limit = 140.0  # safety margin before 150s kill
    
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
    
    def get_max_entry(rows):
        return max(max(r) for r in rows)
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    # Improved CSP solver with forward checking
    def solve_csp(n_rows, k, max_entry, timeout, seed=None):
        """
        CSP solver using bitset difference tracking with forward checking.
        Returns list of rows or None.
        """
        if seed is not None:
            random.seed(seed)
        
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        
        def build_row(current_row, used_bits, deadline, max_entry):
            """
            Recursively build a row using backtracking with bitset checks.
            Uses MRV-like heuristic: try values that leave most flexibility.
            """
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Forward checking: ensure we can still place remaining elements
            # Need at least 'remaining' more distinct values > last
            if max_entry - last < remaining:
                return False
            
            # Generate candidate values with better ordering
            candidates = []
            for x in range(last + 1, max_entry + 1):
                # Prune: need room for remaining-1 more elements after x
                if max_entry - x < remaining - 1:
                    break
                
                # Compute new differences as bitset
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if d <= 0:
                        continue
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Check forward propagation: after placing x, can we still
                # place remaining-1 more elements?
                # Simple check: there must be at least remaining-1 values > x
                # that don't create conflicts (approximate)
                
                # Score: prefer values that use fewer small differences
                # and leave more room
                small_diff_count = 0
                for i in range(1, min(15, max_entry) + 1):
                    if (new_diff_bits >> i) & 1:
                        small_diff_count += 1
                
                # Also prefer values that are "spread out"
                spread_penalty = 0
                if len(current_row) > 1:
                    prev_diff = x - current_row[-2]
                    spread_penalty = max(0, 10 - prev_diff)
                
                score = small_diff_count * 2 + spread_penalty
                candidates.append((score, x, new_diff_bits))
            
            # Sort by score
            candidates.sort()
            
            # Randomize slightly for diversity
            if len(candidates) > 3 and random.random() < 0.3:
                idx = random.randint(0, min(2, len(candidates)-1))
                candidates.insert(0, candidates.pop(idx))
            
            for score, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline, max_entry):
                    return True
                
                current_row.pop()
            
            return False
        
        # Build rows one by one
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline, max_entry):
                rows.append(current_row)
                # Update used_bits with all diffs from this row
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows
    
    # Simulated annealing polish
    def polish_rows(rows, time_budget):
        """
        Try to reduce the maximum entry using local search / simulated annealing.
        """
        if time_budget < 1.0:
            return rows
        
        deadline = time.time() + time_budget
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        
        n = len(best_rows)
        kk = k
        
        def compute_max(rows):
            return max(max(r) for r in rows)
        
        def is_valid(rows):
            used = set()
            for row in rows:
                if len(row) != kk + 1:
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
        
        current_rows = [list(r) for r in best_rows]
        current_max = compute_max(current_rows)
        
        temperature = 50.0
        cooling = 0.995
        
        while time.time() < deadline and temperature > 0.5:
            # Pick a random row and position to modify
            row_idx = random.randint(0, n - 1)
            pos = random.randint(1, kk)  # position 1..kk (not 0)
            
            old_val = current_rows[row_idx][pos]
            
            # Try a new value
            # Bias towards reducing the max
            if old_val == current_max:
                # Try to reduce this value
                delta = random.randint(1, max(1, old_val // 4))
                new_val = old_val - delta
            else:
                # Try small perturbations
                delta = random.randint(-max(1, old_val // 5), max(1, old_val // 5))
                new_val = old_val + delta
            
            if new_val <= 0:
                new_val = 1
            if new_val >= current_max + 10:
                new_val = current_max - random.randint(1, 5)
            
            # Check if new value maintains strictly increasing order
            if pos > 0 and new_val <= current_rows[row_idx][pos - 1]:
                new_val = current_rows[row_idx][pos - 1] + 1
            if pos < kk and new_val >= current_rows[row_idx][pos + 1]:
                new_val = current_rows[row_idx][pos + 1] - 1
            
            if new_val <= current_rows[row_idx][pos - 1] or (pos < kk and new_val >= current_rows[row_idx][pos + 1]):
                temperature *= cooling
                continue
            
            # Temporarily set new value
            current_rows[row_idx][pos] = new_val
            
            if is_valid(current_rows):
                new_max = compute_max(current_rows)
                if new_max < current_max:
                    current_max = new_max
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in current_rows]
                # Accept improvement always
            else:
                # Invalid: revert
                current_rows[row_idx][pos] = old_val
                temperature *= cooling
                continue
            
            # Accept/reject based on temperature
            if current_max > best_max:
                # This is worse than best, but might be accepted temporarily
                delta_obj = current_max - best_max
                if random.random() < (1.0 / (1.0 + delta_obj / max(temperature, 0.1))):
                    pass  # accept
                else:
                    current_rows[row_idx][pos] = old_val
                    current_max = compute_max(current_rows)
            
            temperature *= cooling
        
        return best_rows
    
    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try max_entry values from lb upward with time budgets
    max_search_range = min(greedy_max, lb + 40)
    
    for max_entry in range(lb, max_search_range + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 10  # leave 10 sec for polish
        
        if remaining_time < 1.0:
            break
        
        # Allocate time based on distance from lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(20.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 15:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.5:
            break
        
        # Try multiple seeds for diversity
        found = False
        for seed in [None, 42, 123, 456]:
            if time.time() > start_time + time_limit - 10:
                break
            result = solve_csp(n_rows, k, max_entry, budget, seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            break  # Found best possible for this max_entry
    
    # Polish with simulated annealing
    elapsed = time.time() - start_time
    remaining_for_polish = time_limit - elapsed - 5
    if remaining_for_polish > 2.0 and best_max > lb:
        polished = polish_rows(best_rows, remaining_for_polish)
        if polished is not None and is_valid(polished, n_rows, k):
            polished_max = get_max_entry(polished)
            if polished_max < best_max:
                best_rows = polished
                best_max = polished_max
    
    # Final validation
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
