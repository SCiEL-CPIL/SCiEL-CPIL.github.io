"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    
    Uses a backtracking approach with constraint propagation:
    - Total differences needed: n_rows * k*(k+1)//2
    - We assign differences to rows, ensuring each row's differences form a valid ruler
    """
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    
    # The simple counting bound: smallest m such that m*(m+1)//2 >= total_diffs
    # Actually, we need to find the smallest M such that we can partition {1..M} into
    # n_rows sets, each forming a valid ruler of length k+1
    # Lower bound on M: need at least total_diffs distinct differences, so M >= total_diffs
    # But also each row with k+1 marks has max difference at most M, and the differences
    # must be a subset of {1..M}
    
    # Simple lower bound: M such that M*(M+1)//2 >= total_diffs is not quite right
    # because differences don't need to be all of 1..M, just distinct
    # The absolute minimum M is at least total_diffs (since we need total_diffs distinct positive integers)
    # But also, for a single row with k+1 marks 0 < a1 < ... < ak = M_row, 
    # the differences include M_row itself, so M >= max over rows of the last element
    
    # Let's use a smarter lower bound. For k=7, each row needs 28 differences.
    # The greedy approach gives a certain scope. We'll try to improve.
    
    # First, let's compute the greedy solution to get an upper bound
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_scope = max(max(row) for row in greedy_rows) if greedy_rows else 0
    
    # Now try to find a better solution using backtracking
    # We'll search for solutions with scope < greedy_scope
    
    # For the specific case n_rows=4, k=7, we know the answer is 134
    # Let's try to verify if we can do better
    
    # Strategy: Use backtracking to assign differences to rows
    # Each row is determined by its set of k+1 marks (including 0)
    # The differences of a row are all pairwise differences
    
    # We'll build rows one at a time, but use constraint checking
    # to ensure feasibility
    
    # For efficiency, let's first try the greedy, then try local search
    # to see if we can reduce the scope
    
    # Actually, let's implement a proper backtracking solver
    # that tries to construct rows with a bounded scope
    
    # Start with the lower bound and increase
    # Lower bound: we need total_diffs distinct differences, each at most M
    # So M must be at least total_diffs
    # But also, the largest difference in each row is the last element of that row
    # So if all rows have last element <= M, all differences are <= M
    
    # Let's try scopes from lower_bound to greedy_scope - 1
    
    # Compute a better lower bound
    # For k marks, the minimum possible maximum difference for a row with k+1 marks
    # that has k*(k+1)//2 distinct differences is at least k*(k+1)//2 (trivial)
    # But more tightly, a ruler of length k+1 with minimal span...
    
    # Let's just try from a reasonable lower bound
    lower_bound = total_diffs  # very loose
    # Better: for n_rows rows each needing total_diffs_per_row differences,
    # and all differences must be distinct and <= M, we need M >= total_diffs
    # But we can do better. The sum of all differences across all rows must be at least
    # the sum of the smallest total_diffs positive integers = total_diffs*(total_diffs+1)//2
    # On the other hand, for a row with marks 0 = a_0 < a_1 < ... < a_k,
    # the sum of all differences is sum_{i<j} (a_j - a_i)
    # = sum_i a_i * (2*i - k) ... let me compute this properly
    # sum_{i<j} (a_j - a_i) = sum_j a_j * j - sum_i a_i * (k - i)
    # = sum_j a_j * j - sum_i a_i * (k - i)
    # = sum_{i=0}^{k} a_i * (i - (k - i))
    # = sum_{i=0}^{k} a_i * (2*i - k)
    # = sum_{i=0}^{k} a_i * (2*i - k)
    
    # For the row to have small span (a_k <= M), we want to minimize the sum of differences
    # given that a_0 = 0 and a_k <= M
    # The minimum sum of differences for a row with a_k = M is achieved when
    # the marks are as evenly spaced as possible
    
    # This is getting complex. Let's just use a practical approach:
    # Try to find a solution with scope < greedy_scope using backtracking
    
    # For the known case (4,7), greedy might give something around 140-150
    # and the optimal is 134
    
    # Let's implement backtracking with pruning
    
    # Key insight: we can represent each row by its difference set
    # Two rows are compatible if their difference sets are disjoint
    
    # We'll precompute all valid rulers of length k+1 with span <= M
    # for a given M, then find a packing of n_rows disjoint rulers
    
    # For M up to ~134 and k=7, the number of rulers might be large
    # but let's try
    
    # Actually, for k=7, a ruler has 8 marks and 28 differences
    # The number of rulers with span <= M is bounded
    # Let's generate them on the fly
    
    # Better approach: backtracking on rows
    # For each row, try all possible rulers (sets of k+1 marks starting at 0, ending at <= M)
    # that have disjoint differences from already-placed rows
    
    # To make this efficient, we need to generate candidate rulers quickly
    
    # Let's implement this
    
    # First, get the greedy solution
    if greedy_scope <= lower_bound:
        return greedy_rows
    
    # Try to improve
    for target_scope in range(lower_bound, greedy_scope):
        result = _backtracking_construct(n_rows, k, target_scope)
        if result is not None:
            return result
    
    return greedy_rows


def _greedy_construct(n_rows, k):
    """Greedy construction as in the original."""
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


def _backtracking_construct(n_rows, k, max_scope):
    """
    Try to construct n_rows rows with max entry <= max_scope.
    Returns None if not possible.
    """
    used = set()
    rows = []
    
    def backtrack(row_idx):
        if row_idx == n_rows:
            return True
        
        # Generate candidate rows: all sets of k+1 marks starting at 0, ending at <= max_scope
        # with all differences unused
        # This is expensive, so we use a heuristic: try to build the row greedily
        # but with the constraint that the last element <= max_scope
        
        # We'll use a DFS to build one row
        return _build_row(0, [0], used, max_scope, k, rows, row_idx)
    
    def _build_row(pos, current_row, used, max_scope, k, rows, row_idx):
        """Recursively build a row."""
        if len(current_row) == k + 1:
            # Check if the last element is <= max_scope
            if current_row[-1] <= max_scope:
                # Check all differences are unused
                diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        diffs.add(current_row[j] - current_row[i])
                if not diffs & used:
                    # Place this row
                    rows.append(current_row[:])
                    new_used = used | diffs
                    if backtrack(row_idx + 1):
                        return True
                    rows.pop()
            return False
        
        # Try next values for the next mark
        start_val = current_row[-1] + 1
        # Upper bound: we need to be able to fit (k+1 - len(current_row)) more marks
        # with differences all <= max_scope
        # The remaining marks must be <= max_scope
        # And the differences from current_row[-1] to new marks must be unused
        
        # Heuristic: try values from start_val up to max_scope
        # But we need to ensure we can complete the row
        # The remaining number of marks is rem = k + 1 - len(current_row)
        # We need rem marks, each > current_row[-1] and <= max_scope
        # Also, all differences must be unused
        
        # For efficiency, limit the search
        end_val = min(max_scope, start_val + rem * 20)  # heuristic limit
        
        for val in range(start_val, end_val + 1):
            # Check if adding val creates any used difference
            new_diffs = {val - a for a in current_row}
            if new_diffs & used:
                continue
            # Also check if we can still complete the row
            # (optimistic check: are there enough unused differences available?)
            if _build_row(pos + 1, current_row + [val], used, max_scope, k, rows, row_idx):
                return True
        
        return False
    
    if backtrack(0):
        return rows
    return None


# EVOLVE-BLOCK-END
