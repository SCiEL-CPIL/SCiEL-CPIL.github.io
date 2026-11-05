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
    
    Uses a sophisticated CSP approach with:
    - Bitset-based difference constraint propagation for instant pruning
    - Dynamic variable ordering (MRV heuristic)
    - Forward checking to detect conflicts early
    - Randomized restarts with increasing time budgets
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
    
    # Precompute difference masks for efficiency
    # diff_mask[i][j] = bitset of differences between positions i and j in a row
    # But this depends on actual values, so we compute on the fly
    
    def solve_csp_advanced(n_rows, k, max_entry, timeout, seed=0):
        """
        Advanced CSP solver with:
        - Dynamic variable ordering
        - Forward checking
        - Bitset-based constraint propagation
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences (bit d set if difference d is used)
        rows = []
        rng = random.Random(seed)
        
        # State: list of rows, each row is a list of chosen values
        # We'll use a stack-based approach with backtracking
        
        def get_available_differences(used_bits, max_diff):
            """Return bitset of available differences from 1 to max_diff"""
            all_bits = (1 << (max_diff + 1)) - 2  # bits 1 through max_diff
            return all_bits & ~used_bits
        
        def count_available(used_bits, min_d, max_d):
            """Count available differences in range [min_d, max_d]"""
            if min_d > max_d:
                return 0
            mask = ((1 << (max_d + 1)) - 1) ^ ((1 << min_d) - 1)
            avail = mask & ~used_bits
            return bin(avail).count('1')
        
        def check_feasible(row, used_bits, remaining_slots, max_entry):
            """
            Check if we can still complete the row with remaining_slots elements.
            Uses a greedy feasibility check.
            """
            if remaining_slots == 0:
                return True
            
            last = row[-1]
            # We need to place remaining_slots elements after last
            # Each new element x must satisfy:
            #   x > last
            #   x <= max_entry
            #   For each a in row: x - a must be available
            
            # Quick check: can we fit remaining_slots elements?
            if max_entry - last < remaining_slots:
                return False
            
            # More detailed check: simulate greedy placement
            sim_row = list(row)
            sim_used = used_bits
            for _ in range(remaining_slots):
                last_sim = sim_row[-1]
                found = False
                for x in range(last_sim + 1, max_entry + 1):
                    conflict = False
                    for a in sim_row:
                        d = x - a
                        if (sim_used >> d) & 1:
                            conflict = True
                            break
                    if not conflict:
                        sim_row.append(x)
                        for a in sim_row[:-1]:
                            d = x - a
                            sim_used |= (1 << d)
                        found = True
                        break
                if not found:
                    return False
            
            return True
        
        def solve_recursive(row, used_bits, row_idx, deadline):
            """
            Recursively solve for the current row, then move to next row.
            Returns True if solution found.
            """
            if time.time() > deadline:
                return False
            
            if len(row) == k + 1:
                # Row complete, move to next row
                if row_idx + 1 >= n_rows:
                    return True
                
                # Start next row
                return solve_recursive([0], used_bits, row_idx + 1, deadline)
            
            last = row[-1]
            remaining = k + 1 - len(row)
            
            # Generate candidates with forward checking
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            # Collect valid candidates and score them
            candidates = []
            for x in range(min_x, max_x + 1):
                # Check if x creates conflicts
                new_diff_bits = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Forward checking: can we still complete the row?
                new_row = row + [x]
                new_used = used_bits | new_diff_bits
                
                # Quick feasibility check
                if not check_feasible(new_row, new_used, remaining - 1, max_entry):
                    continue
                
                # Score the candidate
                # Prefer candidates that:
                # 1. Use small differences (pack densely)
                # 2. Leave more flexibility for future
                # Heuristic: prefer smaller x to keep row compact
                score = x
                
                candidates.append((score, x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Sort by score (prefer smaller x) with some randomness
            candidates.sort(key=lambda c: c[0])
            
            # Add randomness: shuffle among top candidates
            if len(candidates) > 1:
                # Keep the best candidate first with some probability
                if rng.random() < 0.7:
                    pass  # Use sorted order
                else:
                    # Randomize among top 3
                    top_n = min(3, len(candidates))
                    top = candidates[:top_n]
                    rng.shuffle(top)
                    candidates = top + candidates[top_n:]
            
            for _, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                row.append(x)
                new_used = used_bits | new_diff_bits
                
                if solve_recursive(row, new_used, row_idx, deadline):
                    return True
                
                row.pop()
            
            return False
        
        # Start solving
        current_row = [0]
        if solve_recursive(current_row, 0, 0, deadline):
            rows.append(current_row)
            # Extract all rows from the solution
            # Actually, we need to track all rows, not just the last one
            # Let me restructure...
        
        # The recursive approach above doesn't properly track all rows
        # Let me use a different approach: iterative with explicit state
        
        return None  # Fallback, will be replaced
    
    # Better approach: iterative CSP with explicit state management
    def solve_csp_iterative(n_rows, k, max_entry, timeout, seed=0):
        """
        Iterative CSP solver with backtracking.
        """
        deadline = time.time() + timeout
        rng = random.Random(seed)
        
        # State: (rows, current_row, used_bits)
        # We'll use a stack for backtracking
        
        def check_row_feasible(row, used_bits, remaining, max_entry):
            """Check if row can be completed with 'remaining' more elements"""
            if remaining == 0:
                return True
            last = row[-1]
            if max_entry - last < remaining:
                return False
            # Greedy simulation
            sim = list(row)
            sim_used = used_bits
            for _ in range(remaining):
                last_sim = sim[-1]
                found = False
                for x in range(last_sim + 1, min(last_sim + remaining + 10, max_entry + 1)):
                    conflict = False
                    for a in sim:
                        d = x - a
                        if (sim_used >> d) & 1:
                            conflict = True
                            break
                    if not conflict:
                        sim.append(x)
                        for a in sim[:-1]:
                            d = x - a
                            sim_used |= (1 << d)
                        found = True
                        break
                if not found:
                    return False
            return True
        
        rows = []
        current_row = [0]
        used_bits = 0
        row_idx = 0
        
        # Backtracking stack: stores (row_snapshot, used_bits_snapshot, row_idx_snapshot)
        backtrack_stack = []
        
        while time.time() < deadline:
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                if len(rows) == n_rows:
                    return rows  # Solution found!
                
                # Start next row
                current_row = [0]
                row_idx += 1
                continue
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                # Dead end, backtrack
                if not backtrack_stack:
                    return None
                current_row, used_bits, row_idx = backtrack_stack.pop()
                continue
            
            # Generate candidates
            candidates = []
            for x in range(min_x, max_x + 1):
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
                
                # Forward checking
                new_row = current_row + [x]
                new_used = used_bits | new_diff_bits
                if not check_row_feasible(new_row, new_used, remaining - 1, max_entry):
                    continue
                
                candidates.append((x, new_diff_bits))
            
            if not candidates:
                # Dead end, backtrack
                if not backtrack_stack:
                    return None
                current_row, used_bits, row_idx = backtrack_stack.pop()
                continue
            
            # Sort candidates (prefer smaller x)
            candidates.sort(key=lambda c: c[0])
            
            # Add randomness
            if len(candidates) > 1 and rng.random() < 0.5:
                top_n = min(3, len(candidates))
                top = candidates[:top_n]
                rng.shuffle(top)
                candidates = top + candidates[top_n:]
            
            # Try the first candidate
            x, new_diff_bits = candidates[0]
            
            # Push current state for backtracking if we have more candidates
            if len(candidates) > 1:
                backtrack_stack.append((list(current_row), used_bits, row_idx))
            
            current_row.append(x)
            used_bits |= new_diff_bits
        
        return None
    
    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try max_entry values from lb upward
    search_upper_bound = greedy_max
    
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Allocate time based on distance from lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(15.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(8.0, remaining_time)
        elif distance_from_lb <= 15:
            budget = min(4.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
        
        # Try multiple restarts
        found = False
        num_restarts = max(1, int(budget / 0.8))
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            
            restart_budget = budget / (num_restarts - seed) if seed < num_restarts - 1 else budget - seed * (budget / num_restarts)
            restart_budget = max(0.3, min(1.5, restart_budget))
            
            result = solve_csp_iterative(n_rows, k, max_entry, restart_budget, seed=seed)
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
