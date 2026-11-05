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
    
    Uses a randomized CSP approach with bitset-based difference constraint propagation
    and heuristic candidate ordering (prefer large differences to preserve small ones).
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: We need 'total_diffs_needed' distinct differences.
    # The differences are integers from 1 to M.
    # So M must be at least total_diffs_needed.
    lb = total_diffs_needed
    
    # Greedy baseline with two strategies
    def greedy_construct(n_rows, k, strategy='small_first'):
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
    time_limit = 118.0  # 150s wall clock, leave 32s safety margin
    
    # Bitset-based solver using Python integers as bitsets
    # Bit i is set if difference i is used
    
    def solve_csp_random(n_rows, k, max_entry, timeout, seed=0, prefer_large=True):
        """
        CSP solver using bitset difference tracking and randomized backtracking.
        Returns list of rows or None.
        prefer_large: if True, prefer candidates that use larger differences
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        rng = random.Random(seed)
        
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
                
                # Score candidate: prefer those that use larger differences
                # This preserves small differences for future rows
                # The "cost" of a candidate is the sum of its differences
                # Higher cost = uses larger diffs = better when prefer_large
                cost = 0
                for a in current_row:
                    cost += (x - a)
                
                candidates.append((cost, x, new_diff_bits))
            
            if not candidates:
                return False
            
            if prefer_large:
                # Sort by cost descending, then add randomness
                candidates.sort(key=lambda c: c[0], reverse=True)
                # Take top candidates and shuffle within groups
                # Use a window of top candidates to add diversity
                top_n = min(len(candidates), max(5, len(candidates) // 3))
                top_candidates = candidates[:top_n]
                rng.shuffle(top_candidates)
                # Add some random candidates from the rest
                rest = candidates[top_n:]
                if rest:
                    extra_n = min(3, len(rest))
                    top_candidates.extend(rng.sample(rest, extra_n))
                candidates = top_candidates
            else:
                # Sort by cost ascending
                candidates.sort(key=lambda c: c[0])
                top_n = min(len(candidates), max(5, len(candidates) // 3))
                top_candidates = candidates[:top_n]
                rng.shuffle(top_candidates)
                rest = candidates[top_n:]
                if rest:
                    extra_n = min(3, len(rest))
                    top_candidates.extend(rng.sample(rest, extra_n))
                candidates = top_candidates
            
            for cost, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline):
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
    
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
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
        
        found = False
        num_restarts = max(1, int(budget / 0.8))
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            
            restart_budget = min(0.8, budget / max(1, num_restarts - seed))
            
            # Alternate between prefer_large and prefer_small strategies
            prefer_large = (seed % 2 == 0)
            
            result = solve_csp_random(n_rows, k, max_entry, restart_budget, seed=seed, prefer_large=prefer_large)
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
