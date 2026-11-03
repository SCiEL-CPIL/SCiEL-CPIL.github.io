import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Counting bound: largest entry M must satisfy M >= total_diffs
    # Actually, differences are from 1 to M, so we need M >= total_diffs
    counting_bound = total_diffs
    
    # Get a good initial solution using greedy
    best_rows = greedy_construct(n_rows, k)
    if best_rows is None:
        # Fallback: try harder
        best_rows = greedy_construct(n_rows, k, start_from=0)
    
    best_max_val = max(max(row) for row in best_rows)
    
    # Try to improve using local search / compaction
    # We'll try to reduce the largest entry by trying to pack things tighter
    
    # Strategy: Try decreasing target values and see if we can construct a valid set
    # Start from best_max_val and go down
    for target in range(best_max_val - 1, max(counting_bound - 1, 0), -1):
        if time.time() > deadline:
            break
        
        # Try to construct a solution with max entry <= target
        # Use backtracking with pruning
        result = try_construct_with_target(n_rows, k, target)
        if result is not None:
            best_rows = result
            best_max_val = target
            # Continue trying to go lower
            continue
    
    # Try some randomized approaches to potentially find better solutions
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try randomized local search / hill climbing
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            
            # Create a perturbed version of best_rows
            perturbed = perturb_rows(best_rows, n_rows, k)
            if perturbed is None:
                continue
            
            # Try to compact/improve
            improved = try_compact(perturbed, n_rows, k)
            if improved is not None:
                max_val = max(max(row) for row in improved)
                if max_val < best_max_val:
                    best_max_val = max_val
                    best_rows = improved
    
    return best_rows

def greedy_construct(n_rows, k, start_from=0):
    """Greedy construction: for each row, pick the smallest valid next element."""
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        current = 0
        while len(row) < k + 1:
            current += 1
            # Check if current - a is valid for all a in row
            valid = True
            for a in row:
                diff = current - a
                if diff in used_diffs:
                    valid = False
                    break
            if valid:
                # Add all new differences
                for a in row:
                    used_diffs.add(current - a)
                row.append(current)
        
        rows.append(row)
    
    return rows

def try_construct_with_target(n_rows, k, target):
    """Try to construct a valid set where all entries are <= target.
    Uses backtracking with pruning."""
    
    # We need to place n_rows rows, each with k+1 entries starting at 0
    # All entries must be in [0, target]
    # All pairwise differences within each row must be distinct across all rows
    
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, pos_in_row, current_row):
        """Recursively build the solution."""
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if pos_in_row == k + 1:
            # Row is complete
            rows.append(current_row[:])
            return backtrack(row_idx + 1, 0, [0])
        
        if pos_in_row == 0:
            # Start of a new row
            current_row = [0]
            return backtrack(row_idx, 1, current_row)
        
        # Try to find the next element
        # It must be > current_row[-1] and <= target
        start_val = current_row[-1] + 1
        for val in range(start_val, target + 1):
            # Check if val - a is valid for all a in current_row
            valid = True
            new_diffs = []
            for a in current_row:
                diff = val - a
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            # Try this value
            for d in new_diffs:
                used_diffs.add(d)
            current_row.append(val)
            
            if backtrack(row_idx, pos_in_row + 1, current_row):
                return True
            
            # Backtrack
            for d in new_diffs:
                used_diffs.discard(d)
            current_row.pop()
        
        return False
    
    # Start backtracking
    if backtrack(0, 0, []):
        return rows
    
    return None

def perturb_rows(rows, n_rows, k):
    """Create a slightly perturbed version of the rows for local search."""
    # This is a simple perturbation: randomly swap some elements or shift them
    # For now, just return a copy (this is a placeholder for more sophisticated perturbation)
    new_rows = []
    for row in rows:
        new_row = row[:]
        new_rows.append(new_row)
    return new_rows

def try_compact(rows, n_rows, k):
    """Try to compact the rows to reduce the maximum entry."""
    # This is a placeholder for a compaction algorithm
    # A simple approach: try to shift elements down if possible
    used_diffs = set()
    
    # First, collect all differences
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    # Try to reduce each row by shifting elements
    improved_rows = []
    for row in rows:
        new_row = [0]
        for val in row[1:]:
            # Try to find a smaller value that works
            found = False
            for candidate in range(new_row[-1] + 1, val):
                valid = True
                for a in new_row:
                    diff = candidate - a
                    if diff in used_diffs:
                        valid = False
                        break
                if valid:
                    for a in new_row:
                        used_diffs.discard(candidate - a)
                    new_row.append(candidate)
                    found = True
                    break
            if not found:
                new_row.append(val)
        improved_rows.append(new_row)
    
    # Verify the result
    if is_valid(improved_rows, n_rows, k):
        return improved_rows
    
    return None

def is_valid(rows, n_rows, k):
    """Check if the rows form a valid difference triangle set."""
    if len(rows) != n_rows:
        return False
    
    used_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        # Check differences
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff = row[j] - row[i]
                if diff in used_diffs:
                    return False
                used_diffs.add(diff)
    
    return True
