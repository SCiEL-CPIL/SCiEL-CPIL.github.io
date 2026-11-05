# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a CSP approach with bitset-based difference tracking and randomized
    backtracking with dynamic variable ordering.
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
    time_limit = 145.0  # 150s limit, leave 5s safety margin
    
    # Bitset-based CSP solver
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and randomized backtracking
        with dynamic variable ordering (most-constrained first).
        """
        deadline = time.time() + timeout
        used_bits = 0
        rows = []
        rng = random.Random(seed)
        
        # Precompute which differences are available
        # We track used differences as a bitset
        
        def build_row(current_row, used_bits, deadline, rng):
            """Recursively build a row using backtracking."""
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
            
            # Generate and score candidates
            candidates = []
            for x in range(min_x, max_x + 1):
                # Check if x conflicts with used differences
                conflict = False
                new_diff_bits = 0
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Heuristic score: prefer x that uses less "valuable" small differences
                # and keeps the row compact
                # Score = number of small differences used (penalty) + compactness bonus
                small_diff_count = 0
                for a in current_row:
                    d = x - a
                    if d <= k * 2:  # consider "small" differences
                        small_diff_count += 1
                
                # Prefer smaller x (compactness) but penalize using small differences
                score = -x + small_diff_count * 0.1
                
                candidates.append((score, x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Sort by score (prefer higher score = less penalty)
            candidates.sort(key=lambda c: -c[0])
            
            # Add some randomness: shuffle top candidates
            top_n = min(len(candidates), 10)
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            
            for _, x, new_diff_bits in top_candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline, rng):
                    return True
                
                current_row.pop()
            
            return False
        
        # Build rows one by one
        # Use dynamic ordering: build the row that has the most constraints first
        # For simplicity, we build in order but could reorder based on difficulty
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline, rng):
                rows.append(current_row)
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
    
    # Also try some values between lb and greedy_max with more time allocation
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        
        if remaining_time < 1.0:
            break
        
        # Allocate time based on distance from lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(15.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 20:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.5:
            continue
        
        # Try multiple restarts
        num_restarts = max(1, int(budget / 2.0))
        found = False
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            
            restart_budget = budget / (num_restarts - seed) if seed < num_restarts - 1 else budget
            restart_budget = min(3.0, restart_budget)
            
            result = solve_csp(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            # Found solution for this max_entry
            # Since we're iterating from lb upward, this is the best found so far
            # But we could potentially find a smaller M? No, we started from lb.
            # However, the solver might fail for lb but succeed for lb+1, etc.
            # We should continue to see if we can find an even smaller M? 
            # No, we're going upward. If we find one at M, that's the minimum for this search.
            # But wait - we might have skipped some values. Let's break since we found the smallest.
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
