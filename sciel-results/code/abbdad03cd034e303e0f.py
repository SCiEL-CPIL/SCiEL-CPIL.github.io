"""Rewritten program for the difference triangle set task."""
import random
import sys
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    
    Uses a backtracking search with branch and bound to find the minimal
    maximum entry.
    """
    # Calculate the counting bound as a starting lower bound
    lower_bound = _counting_bound(n_rows, k)
    
    # We will search for a solution with max entry M.
    # We start from the lower bound and increase M if no solution is found.
    # However, since we want the *smallest* M, we can use a branch-and-bound
    # approach where we try to find a solution and keep the best found.
    
    best_rows = None
    best_max = float('inf')
    
    # Set a time limit for the search to ensure we return within 120s
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # We can iterate M from lower_bound upwards, but branch-and-bound is more efficient
    # if we let the solver prune branches that exceed the current best.
    # Let's initialize best_max with a greedy solution to have a tight upper bound.
    greedy_rows = _greedy_construct(n_rows, k)
    if greedy_rows:
        best_max = max(max(r) for r in greedy_rows)
        best_rows = [list(r) for r in greedy_rows]
    
    # Backtracking state
    used_diffs = set()
    current_rows = []
    
    def backtrack(row_idx, current_row, current_max):
        nonlocal best_rows, best_max, used_diffs, current_rows
        nonlocal start_time, time_limit
        
        # Time check
        if time.time() - start_time > time_limit:
            return False
        
        # Base case: all rows constructed
        if row_idx == n_rows:
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in current_rows]
            return True
        
        # Pruning: if current max already exceeds best, no point continuing
        if current_max >= best_max:
            return False
        
        # Determine the next element to place
        # The row must start with 0
        if len(current_row) == 0:
            # First element is always 0
            new_row = [0]
            # No differences yet for the first element
            if backtrack(row_idx, new_row, current_max):
                return True
            return False
        
        # If we are filling the current row, we need to pick the next value
        # The row has len(current_row) elements so far. We need k+1 elements total.
        if len(current_row) == k + 1:
            # Row is complete, move to next row
            current_rows.append(current_row)
            if backtrack(row_idx + 1, [], current_max):
                # Undo the append if we want to keep state clean, 
                # but since we copy to best_rows, we can just let it be.
                # Actually, we should pop to maintain consistent state for backtracking
                current_rows.pop()
                return True
            current_rows.pop()
            return False
        
        # Pick the next value for the current row
        last_val = current_row[-1]
        
        # We try values from last_val + 1 up to best_max - 1 (since we prune >= best_max)
        # To find the smallest max, we should try smaller values first.
        # However, to diversify, we could randomize the order slightly?
        # For now, deterministic increasing order is best for minimization.
        
        for next_val in range(last_val + 1, best_max):
            # Check if this value is valid
            # Calculate new differences
            new_diffs = {next_val - a for a in current_row}
            
            # Check against global used_diffs and local diffs (already in current_row)
            # Wait, current_row contains previous elements. The diffs between next_val and previous elements
            # are new_diffs. We must ensure these are not in used_diffs.
            # Also, we must ensure that these new_diffs are distinct from each other?
            # Yes, they are differences with a fixed next_val, so they are distinct 
            # because the previous elements are distinct.
            
            if new_diffs & used_diffs:
                continue
            
            # Valid move
            # Update state
            used_diffs.update(new_diffs)
            new_current_row = current_row + [next_val]
            new_current_max = max(current_max, next_val)
            
            if backtrack(row_idx, new_current_row, new_current_max):
                # If a solution was found, we can return True immediately 
                # if we just want *a* solution, but we want the *best*.
                # The nonlocal best_max is updated inside.
                # We should continue searching to see if we can find a better one?
                # Branch and bound: if we find a solution, best_max is updated.
                # Subsequent branches will be pruned if they exceed this new best_max.
                # So we should NOT return True immediately unless we want to stop early.
                # But returning True propagates up, stopping the search for this branch.
                # This is correct: we found a valid completion. We don't need to explore 
                # other completions for this specific prefix if we are only interested 
                # in the global best, because the global best is tracked in nonlocal variables.
                # However, there might be a *better* solution in a *different* branch 
                # of the *same* prefix? No, the prefix is fixed. 
                # There might be a better solution with a *different* prefix.
                # So we return True to unwind the stack, but the outer loops 
                # will continue to try other values for the previous variables.
                return True
            
            # Undo state
            used_diffs.difference_update(new_diffs)
        
        return False
    
    # Start the search
    # We can add a small randomization to the initial greedy bound or restarts if needed.
    # For now, run the main backtracker.
    backtrack(0, [], 0)
    
    if best_rows is None:
        # Fallback to greedy if backtracking failed or timed out without finding anything
        return greedy_rows
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Original greedy construction."""
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


def _counting_bound(n_rows, k):
    """
    Simple counting bound: total distinct differences needed is n_rows * k*(k+1)/2.
    The number of possible differences up to m is m(m+1)/2.
    So we need m(m+1)/2 >= n_rows * k*(k+1)/2
    => m(m+1) >= n_rows * k*(k+1)
    """
    target = n_rows * k * (k + 1)
    m = 1
    while m * (m + 1) < target:
        m += 1
    return m
# EVOLVE-BLOCK-END
