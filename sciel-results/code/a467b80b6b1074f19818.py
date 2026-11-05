"""Difference triangle set (4,7) - spectral packing approach."""
import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a randomized greedy spectral descent approach:
    - Each row is a binary sequence of length M+1 with k+1 ones
    - Global constraint: union of all pairwise autocorrelation peaks (lag > 0) has no value > 1
    - Iteratively shift row supports to minimize conflicts
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
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
    
    # Spectral packing approach
    def spectral_packing(n_rows, k, max_entry, deadline):
        """
        Build rows using spectral packing:
        - Track used differences as a set
        - For each row, use randomized greedy with backtracking
        - Use difference pressure heuristic: prefer candidates that minimize max new diff
        """
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            row = [0]
            current_max = 0
            
            # Try multiple random orderings of candidate values
            best_row = None
            best_max_diff = float('inf')
            
            # Use a priority-based approach: try to keep differences small
            # but spread them out to avoid conflicts
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                row = [0]
                success = True
                
                # Randomize the order slightly for diversity
                for pos in range(1, k + 1):
                    # Generate candidates: values that are increasing and leave room
                    min_val = row[-1] + 1
                    max_val = max_entry - (k + 1 - pos - 1)  # leave room for remaining elements
                    
                    if min_val > max_val:
                        success = False
                        break
                    
                    # Try candidates in random order, but bias toward smaller values
                    candidates = list(range(min_val, max_val + 1))
                    random.shuffle(candidates)
                    
                    # Sort by a heuristic: prefer values that create fewer conflicts
                    # and have smaller maximum new difference
                    def score_candidate(x):
                        new_diffs = [x - a for a in row]
                        # Count conflicts with used
                        conflicts = sum(1 for d in new_diffs if d in used)
                        if conflicts > 0:
                            return (1, conflicts)  # Invalid
                        # Score: prefer smaller max diff, then smaller sum
                        max_d = max(new_diffs)
                        sum_d = sum(new_diffs)
                        return (0, max_d, sum_d)
                    
                    valid_candidates = []
                    for x in candidates:
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        valid_candidates.append(x)
                    
                    if not valid_candidates:
                        success = False
                        break
                    
                    # Pick the best candidate by score
                    valid_candidates.sort(key=score_candidate)
                    # With some probability, pick a random valid candidate for diversity
                    if attempts > 10 and random.random() < 0.3:
                        chosen = random.choice(valid_candidates[:max(1, len(valid_candidates)//2)])
                    else:
                        chosen = valid_candidates[0]
                    
                    row.append(chosen)
                    for a in row[:-1]:
                        used.add(chosen - a)
                
                if success and len(row) == k + 1:
                    if max(row) < best_max_diff:
                        best_max_diff = max(row)
                        best_row = row[:]
                        # If we found a good solution, we can stop early
                        if best_max_diff <= max_entry:
                            break
            
            if best_row is None:
                # Roll back and try a different approach
                # Remove the last few rows and retry
                if rows:
                    # This is a failure, return None
                    return None
                else:
                    return None
            
            # Remove the diffs we added during exploration (keep only best_row's diffs)
            # Actually, we need to be careful: we added diffs to 'used' during exploration
            # Let's rebuild used from scratch for this row
            # But that's expensive. Instead, let's just trust the best_row
            rows.append(best_row)
        
        return rows
    
    # Try spectral packing with different max_entry values
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try from lb upward
    for max_entry in range(lb, min(greedy_max + 1, lb + 30)):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5
        
        if remaining < 1.0:
            break
        
        # Budget based on distance from lb
        dist = max_entry - lb
        if dist == 0:
            budget = min(25.0, remaining)
        elif dist <= 3:
            budget = min(15.0, remaining)
        elif dist <= 8:
            budget = min(8.0, remaining)
        elif dist <= 15:
            budget = min(4.0, remaining)
        else:
            budget = min(2.0, remaining)
        
        if budget < 0.5:
            break
        
        deadline = time.time() + budget
        result = spectral_packing(n_rows, k, max_entry, deadline)
        
        if result is not None:
            if validate(result, n_rows, k):
                best_rows = result
                best_max = max_entry
                break
    
    # If spectral packing didn't beat greedy, try a local search improvement
    if best_max == greedy_max and greedy_max > lb:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5
        if remaining > 10.0:
            # Try to improve the greedy solution via local search
            improved = local_search_improve(greedy_rows, n_rows, k, time.time() + remaining)
            if improved is not None:
                best_rows = improved
                best_max = max(max(r) for r in improved)
    
    # Final validation
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows


def local_search_improve(rows, n_rows, k, deadline):
    """
    Local search: try to reduce the maximum entry by shifting elements.
    """
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    current_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    while time.time() < deadline:
        improved = False
        # Try to reduce the max element in each row
        for row_idx in range(n_rows):
            row = best_rows[row_idx]
            if len(row) < 2:
                continue
            max_idx = len(row) - 1
            if row[max_idx] <= 1:
                continue
            
            # Try to reduce row[max_idx]
            for new_val in range(row[max_idx] - 1, row[max_idx] - 10, -1):
                if new_val <= row[max_idx - 1]:
                    break
                
                # Check if new_val creates conflicts
                old_val = row[max_idx]
                # Remove old diffs
                old_diffs = {old_val - a for a in row[:-1]}
                for d in old_diffs:
                    if d in used:
                        used.discard(d)
                
                # Check new diffs
                new_diffs = {new_val - a for a in row[:-1]}
                if any(d in used for d in new_diffs):
                    # Restore old diffs
                    for d in old_diffs:
                        used.add(d)
                    continue
                
                # Accept
                row[max_idx] = new_val
                for d in new_diffs:
                    used.add(d)
                improved = True
                break
            
            if improved:
                new_max = max(max(r) for r in best_rows)
                if new_max < best_max:
                    best_max = new_max
    
    if best_max < max(max(r) for r in rows):
        return best_rows
    return None
