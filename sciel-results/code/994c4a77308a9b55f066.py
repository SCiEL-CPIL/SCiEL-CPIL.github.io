"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import sys
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    
    Uses a constraint satisfaction approach with backtracking and
    dynamic variable ordering based on difference conflicts.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Lower bound: need at least total_diffs distinct differences in [1, m]
    # So m >= total_diffs
    # But we also need each row to fit in [0, m]
    
    # Better lower bound: for each row, we need k*(k+1)/2 distinct differences
    # The minimum span for a Golomb ruler of order k+1 is at least k(k+1)/2
    # but actually the known minimum is larger.
    # For n_rows rows, the total difference count is n_rows * k*(k+1)/2
    
    # Start from a reasonable estimate
    # The counting bound is m >= total_diffs
    # But we can often do better
    
    # Use known results: for (4,7), best known is 134
    # Simple bound is 112
    
    start_m = total_diffs
    
    # Try increasing m
    max_search_time = 110  # Leave some time for overhead
    start_time = time.time()
    
    for offset in range(200):
        current_m = start_m + offset
        
        # Check if we're running out of time
        if time.time() - start_time > max_search_time:
            break
        
        # Try to construct with this m
        result = _csp_construct(n_rows, k, current_m, time.time() - start_time)
        if result is not None:
            return result
    
    # Fallback: greedy construction
    return _greedy_construct(n_rows, k)


def _csp_construct(n_rows, k, max_val, elapsed_time):
    """
    Constraint satisfaction construction using backtracking with
    dynamic variable ordering.
    
    We model this as: we have n_rows rows, each with k+1 positions.
    Position 0 is always 0. We need to assign values to positions 1..k
    for each row such that:
    1. Values are strictly increasing within each row
    2. All differences a[j] - a[i] for i < j are distinct across all rows
    3. All values <= max_val
    """
    used_diffs = set()
    rows = [[] for _ in range(n_rows)]
    
    # Initialize: all rows start with [0]
    for i in range(n_rows):
        rows[i].append(0)
    
    # Track which position in each row we're filling next
    # We'll use a priority queue to select the most constrained variable
    
    def get_constraints_for_placement(row_idx, pos, candidate):
        """
        Check if placing 'candidate' at position 'pos' in row 'row_idx'
        creates any conflicts with existing differences.
        Returns True if valid, False otherwise.
        """
        row = rows[row_idx]
        # Check differences with all previously placed elements in this row
        for prev_pos in range(pos):
            diff = candidate - row[prev_pos]
            if diff <= 0:
                return False  # Not increasing
            if diff in used_diffs:
                return False  # Conflict with existing difference
        return True
    
    def count_available_candidates(row_idx, pos):
        """
        Count how many valid candidates exist for placing at position 'pos'
        in row 'row_idx'.
        """
        row = rows[row_idx]
        if pos == 0:
            return 1  # Always 0
        
        last_val = row[-1]
        count = 0
        for candidate in range(last_val + 1, max_val + 1):
            if get_constraints_for_placement(row_idx, pos, candidate):
                count += 1
        return count
    
    def find_most_constrained_variable():
        """
        Find the (row_idx, pos) with the fewest available candidates.
        This is the MRV (Minimum Remaining Values) heuristic.
        """
        best_var = None
        best_count = float('inf')
        
        for row_idx in range(n_rows):
            pos = len(rows[row_idx])
            if pos > k:
                continue  # Row is complete
            
            count = count_available_candidates(row_idx, pos)
            if count < best_count:
                best_count = count
                best_var = (row_idx, pos)
        
        return best_var, best_count
    
    def backtrack(depth, time_limit):
        """
        Backtracking search with time limit.
        """
        if time.time() - start_time > time_limit:
            return False
        
        # Check if all rows are complete
        all_complete = all(len(rows[i]) == k + 1 for i in range(n_rows))
        if all_complete:
            return True
        
        # Find the most constrained variable
        var, count = find_most_constrained_variable()
        if var is None:
            return False
        
        row_idx, pos = var
        
        # Try all valid candidates for this variable
        row = rows[row_idx]
        last_val = row[-1] if row else -1
        
        for candidate in range(last_val + 1, max_val + 1):
            # Check if this placement is valid
            valid = True
            new_diffs = []
            for prev_pos in range(pos):
                diff = candidate - row[prev_pos]
                if diff <= 0 or diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            # Check if we can still complete the row
            # We need (k - pos) more values after this one
            remaining_positions = k - pos
            if max_val - candidate < remaining_positions:
                break  # Not enough room
            
            # Place the candidate
            row.append(candidate)
            used_diffs.update(new_diffs)
            
            if backtrack(depth + 1, time_limit):
                return True
            
            # Backtrack
            row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    # Start the backtracking search
    time_limit = 115  # Leave 5 seconds for overhead
    if backtrack(0, time_limit):
        return rows
    
    return None


def _greedy_construct(n_rows, k):
    """Greedy fallback: same as the original seed program."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END

