import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def _can_match(m, L, n=12):
    # Build the bipartite graph
    # Left nodes: indices 1..n (mapped to 0..n-1)
    # Right nodes: positions 1..L (mapped to 0..L-1)
    
    row_ind = []
    col_ind = []
    
    for i in range(1, n + 1):
        # We need j in [1, L] such that (m + j) % i == 0
        # j % i == (-m) % i
        target = (-m) % i
        # j is 1-indexed in the problem description (b_i = m + j), 
        # but our right nodes are 0-indexed positions j-1?
        # Let's stick to j being the offset 1..L.
        # Node on right side corresponds to offset j.
        # Let's map offset j to node index j-1.
        
        # Find all j in [1, L] such that j % i == target
        # Note: if target is 0, j must be multiple of i.
        
        # Efficient way:
        # First j >= 1 with j % i == target
        # If target == 0, first j is i (if i<=L) or 0? No j>=1.
        # j = k*i + target.
        # We need 1 <= k*i + target <= L.
        
        if target == 0:
            # j must be i, 2i, ...
            # smallest j >= 1 is i
            start_j = i
        else:
            # j = target, target+i, ...
            # if target < 1? target is 1..i-1. So target >= 1.
            start_j = target
            
        # Iterate
        j = start_j
        while j <= L:
            row_ind.append(i - 1) # Left node index
            col_ind.append(j - 1) # Right node index (offset j-1)
            j += i
            
    if not row_ind:
        # No edges at all? Then no matching possible if n>0
        return False
        
    # Create sparse matrix
    # shape (n, L)
    data = np.ones(len(row_ind), dtype=np.int8)
    graph = csr_matrix((data, (row_ind, col_ind)), shape=(n, L))
    
    # Find maximum bipartite matching
    # Returns (rows, cols) where rows[i] is the matched column for row i, or -1 if unmatched
    # Note: scipy returns arrays of length max(n, L)? 
    # Actually, documentation says: 
    # "rows : 1-D array of int, shape (max(n, m),) ... rows[i] is the column matched to row i"
    # Wait, the return is two arrays: rows and cols.
    # rows[i] is the column index matched to row i, or -1 if no match.
    # cols[j] is the row index matched to column j, or -1 if no match.
    
    # We only care if all rows (indices 1..n) are matched.
    match_rows, _ = maximum_bipartite_matching(graph)
    
    # Check if all n rows are matched
    # match_rows has length n? 
    # "The output arrays are of length max(n, m)"? 
    # Let's check scipy docs carefully.
    # "rows : 1-D array of int, shape (max(n, m),) ... rows[i] is the column matched to row i, or -1 if no match."
    # If n < L, rows will have length L? No, usually it's length of the smaller set or something?
    # Actually, `maximum_bipartite_matching` returns `rows` and `cols`.
    # `rows` has length `n` if we consider left side size n? 
    # Let's just check the first n entries of `match_rows`?
    # Or better, count how many are not -1 in the first n positions?
    # The array length is `max(n, L)`.
    # If n <= L, `match_rows` has length L? 
    # Let's assume `match_rows` corresponds to the left nodes (0..n-1).
    # If the array is longer, we just check the first n elements.
    
    # Actually, from scipy docs:
    # "rows : 1-D array of int, shape (max(n, m),) ... rows[i] is the column matched to row i"
    # So if n=12 and L=144, rows has length 144.
    # Rows 0..11 correspond to our indices 1..12.
    
    if len(match_rows) < n:
        return False
        
    # Check if all left nodes 0..n-1 are matched
    for i in range(n):
        if match_rows[i] == -1:
            return False
            
    return True

def _f_val(m, n=12):
    # Binary search for min L
    lo = n
    hi = 144
    
    # Quick check for upper bound
    if not _can_match(m, hi, n):
        return hi
        
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
            
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85 # Safety margin
    
    best_m = 0
    best_L = 0
    
    period = 27720 # LCM(1..12)
    
    # We scan m in [0, period).
    # Since f(12, m) is periodic with this period, this covers all unique cases.
    # With SciPy, this should be fast.
    
    # Optimization: We can skip some m if we are confident, but let's try full scan first.
    # If time is tight, we can break.
    
    for m in range(period):
        if time.time() > deadline:
            break
            
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
            
    return int(best_m)
