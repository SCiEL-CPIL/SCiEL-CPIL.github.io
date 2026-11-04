import sys

# Global variables for the CSP solver
_node_count = 0
_MAX_NODES = 1000000  # Limit nodes per M to prevent timeout

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within each row are distinct across all rows.
    
    Uses a Constraint Satisfaction Problem (CSP) approach with backtracking
    and constraint propagation.
    """
    if n_rows <= 0 or k <= 0:
        return []
    
    # Total number of differences needed: n_rows * (k+1)*k/2
    num_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound on largest entry M:
    # The differences are distinct integers in [1, M].
    # So we need M >= num_diffs.
    low = num_diffs
    # Also M must be at least k (since we need k+1 distinct numbers 0..M)
    low = max(low, k)
    
    best_rows = None
    best_scope = float('inf')
    
    # Search for the smallest M
    for M in range(low, low + 100):
        global _node_count
        _node_count = 0
        rows = _solve_csp(n_rows, k, M)
        if rows is not None:
            best_rows = rows
            best_scope = M
            break
            
    if best_rows is None:
        # Fallback to a simple greedy if CSP fails or times out
        best_rows = _construct_simple_greedy(n_rows, k)
        
    return best_rows

def _solve_csp(n_rows, k, M):
    """
    Solve the DTS problem for fixed M using backtracking.
    Returns a list of rows or None.
    """
    used_diffs = set()
    rows = []
    
    def backtrack_row(row_idx):
        if row_idx == n_rows:
            return True
        
        current_row = [0]
        current_diffs = set()
        
        if backtrack_elem(row_idx, 1, current_row, current_diffs):
            return True
        
        return False

    def backtrack_elem(row_idx, pos, current_row, current_diffs):
        nonlocal _node_count
        _node_count += 1
        if _node_count > _MAX_NODES:
            return False
        
        if pos == k + 1:
            # Check lexicographical order to break symmetry
            if row_idx > 0:
                prev_row = rows[-1]
                if current_row < prev_row:
                    return False
            
            # Add to global state
            used_diffs.update(current_diffs)
            rows.append(list(current_row))
            
            if backtrack_row(row_idx + 1):
                return True
            
            # Backtrack
            rows.pop()
            for d in current_diffs:
                used_diffs.discard(d)
            return False
        
        prev_val = current_row[-1]
        for val in range(prev_val + 1, M + 1):
            new_diffs = set()
            valid = True
            for a in current_row:
                d = val - a
                if d in used_diffs or d in current_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            current_row.append(val)
            current_diffs.update(new_diffs)
            
            if backtrack_elem(row_idx, pos + 1, current_row, current_diffs):
                return True
            
            current_row.pop()
            for d in new_diffs:
                current_diffs.discard(d)
        
        return False

    if backtrack_row(0):
        return rows
    return None

def _construct_simple_greedy(n_rows, k):
    """Original greedy approach as fallback."""
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
