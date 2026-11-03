import sys
import random

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * n_rows] if k == 0 else []
    
    # Total differences required
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Counting bound
    M_min = total_diffs
    
    # We will try M from M_min upwards
    # To handle potential long runs, we limit the number of nodes visited per M
    # and restart with different random seeds if it fails.
    
    for M in range(M_min, M_min + 50): # Search a bit above counting bound
        if _solve(M, n_rows, k):
            return _best_rows
    
    # Fallback: Return a valid solution if found, or best effort
    # If no solution found in the loop, we should probably return the last valid one or raise an error?
    # The task says "Score: ... minimized".
    # If we don't find one, we should return a valid one with a larger M.
    # Let's extend the range or use a greedy fallback.
    
    # Greedy fallback to ensure we return *something* valid
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# Global variables for the solver
_best_rows = []
_node_limit = 100000
_nodes_visited = 0

def _solve(M, n_rows, k):
    global _best_rows, _nodes_visited
    
    # We need to find rows.
    # We use backtracking.
    # To avoid global state issues in recursion, we pass state.
    
    used_diffs = set()
    rows = []
    
    # Heuristic: We can try to fill rows in a way that balances the differences?
    # For now, simple backtracking with pruning.
    
    def backtrack(row_idx, current_row):
        global _nodes_visited
        if _nodes_visited > _node_limit:
            return False
            
        _nodes_visited += 1
        
        if row_idx == n_rows:
            _best_rows = [r[:] for r in rows]
            return True
            
        if len(current_row) == k + 1:
            rows.append(current_row[:])
            if backtrack(row_idx + 1, []):
                return True
            rows.pop()
            return False
            
        # If current_row is empty, we start with 0
        if not current_row:
            current_row = [0]
            
        last_val = current_row[-1]
        
        # Candidates for next value
        # We try values from last_val + 1 to M
        # To improve performance, we can shuffle candidates if we want randomization,
        # but for minimizing M, we usually want to try smaller values first to keep the row small.
        # However, sometimes a slightly larger value allows a smaller max later.
        # Let's try increasing order first.
        
        for x in range(last_val + 1, M + 1):
            # Check differences
            new_diffs = set()
            conflict = False
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
                
            # Forward Checking: 
            # We have filled len(current_row) elements.
            # We need k+1 - len(current_row) more.
            # The remaining elements must be > x.
            # The max value is M.
            # This doesn't prune much yet.
            
            # Update state
            used_diffs.update(new_diffs)
            current_row.append(x)
            
            if backtrack(row_idx, current_row):
                return True
                
            # Backtrack
            current_row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
                
        return False

    # Run with a few restarts/shuffles if needed?
    # For now, deterministic increasing order.
    # If it's too slow, we might need to randomize the candidate order.
    
    # To make it faster, we can use a heuristic:
    # Prioritize x that creates differences that are "less common" or something?
    # No, let's stick to simple.
    
    # Reset node limit
    _nodes_visited = 0
    
    if backtrack(0, []):
        return True
        
    return False
