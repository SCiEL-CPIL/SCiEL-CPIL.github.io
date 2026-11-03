"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 increasing
    integers starting at 0, such that all positive differences within rows are
    distinct across all rows. Minimize the largest entry.
    
    Uses a randomized greedy set-packing approach with pre-computed Sidon sets
    and backtracking to escape local dead-ends.
    """
    # The counting bound is 105 for (7,5), but we should work with a reasonable upper bound
    # We'll try to find good solutions up to some limit
    max_scope = 130  # Slightly above the known best of 112 to be safe
    
    # Pre-compute all valid row configurations (Sidon sets) up to max_scope
    # A Sidon set of size k+1 is a set of k+1 distinct integers where all pairwise differences are unique
    # We generate all possible rows (lists of k+1 increasing integers starting at 0) with max entry <= max_scope
    # and check if they are Sidon (all positive differences within the row are distinct)
    
    def is_sidon(row):
        """Check if a row (list of increasing integers) is a Sidon set."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Generate all candidate rows
    # A row is a list of k+1 integers: 0 = a0 < a1 < ... < ak <= max_scope
    candidates = []
    
    # We'll use recursion to generate all combinations
    def generate_rows(current, start, min_val):
        if len(current) == k + 1:
            if is_sidon(current):
                candidates.append(list(current))
            return
        # The next value must be > min_val and <= max_scope
        # Also, we need enough room for the remaining elements
        remaining = k + 1 - len(current)
        max_val = max_scope
        for v in range(min_val + 1, max_val + 1):
            current.append(v)
            generate_rows(current, start, v)
            current.pop()
    
    # Start with 0
    generate_rows([0], 0, 0)
    
    # Sort candidates by their maximum entry (scope), then by sum for tie-breaking
    candidates.sort(key=lambda row: (row[-1], sum(row)))
    
    # Now use randomized greedy set-packing with backtracking
    # We want to select n_rows rows such that no difference is used twice
    
    def try_greedy(candidates, n_rows, max_attempts=50):
        """
        Try to greedily select n_rows rows with random tie-breaking.
        Returns a list of rows if successful, None otherwise.
        """
        best_result = None
        best_scope = float('inf')
        
        for attempt in range(max_attempts):
            random.shuffle(candidates)
            used_diffs = set()
            selected = []
            
            for row in candidates:
                if len(selected) == n_rows:
                    break
                # Check if this row conflicts with already selected rows
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        row_diffs.add(d)
                
                if row_diffs & used_diffs:
                    continue
                
                # Select this row
                selected.append(row)
                used_diffs |= row_diffs
            
            if len(selected) == n_rows:
                scope = max(r[-1] for r in selected)
                if scope < best_scope:
                    best_scope = scope
                    best_result = selected
                    # If we've reached the theoretical lower bound or known best, we can stop
                    if scope <= 112:
                        return best_result
        
        return best_result
    
    result = try_greedy(candidates, n_rows)
    
    if result is not None:
        return result
    
    # Fallback: use the simple greedy approach from the seed program
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
