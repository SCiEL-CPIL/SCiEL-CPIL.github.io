import time
import numpy as np

# Try to import SciPy for faster matching
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

P = {'n': 12, 'maximum_start': 1000000}
L_CM = 27720  # LCM(1..12)

def _has_matching_scipy(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i
    using SciPy's maximum_bipartite_matching.
    """
    if L < n:
        return False
    
    row = []
    col = []
    
    for i in range(n):
        div = i + 1
        # We need (m + 1 + j) % div == 0 for j in [0, L-1]
        # j % div == (-m - 1) % div
        target = (-m - 1) % div
        if target < L:
            for j in range(target, L, div):
                row.append(i)
                col.append(j)
                
    if not row:
        return False
        
    # Build sparse matrix
    # Shape (n, L)
    data = np.ones(len(row), dtype=np.int8)
    mat = csr_matrix((data, (row, col)), shape=(n, L))
    
    try:
        left, right = maximum_bipartite_matching(mat)
        # Check if all left nodes are matched
        # left array length should be n
        if len(left) == n:
            for val in left:
                if val == -1:
                    return False
            return True
        return False
    except Exception:
        return False

def _has_matching_python(m, L, n=12):
    """
    Fallback pure Python matching check using DFS with bitmasks.
    """
    if L < n:
        return False
    
    candidates = []
    for i in range(n):
        div = i + 1
        target = (-m - 1) % div
        cands = []
        if target < L:
            for j in range(target, L, div):
                cands.append(j)
        if not cands:
            return False
        candidates.append(cands)
        
    # Order indices by scarcity
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    
    # DFS with bitmask
    # used_mask bits correspond to right nodes (offsets 0..L-1)
    def dfs(idx, used_mask):
        if idx == n:
            return True
        i = order[idx]
        for j in candidates[i]:
            bit = 1 << j
            if not (used_mask & bit):
                if dfs(idx + 1, used_mask | bit):
                    return True
        return False
        
    return dfs(0, 0)

def _compute_f(m, n=12, max_L=144):
    """
    Compute f(n, m): least positive L such that (m, m+L] contains distinct b_i with i|b_i.
    """
    # Calculate lower bound for L
    min_L = 0
    for i in range(1, n + 1):
        next_mult = ((m // i) + 1) * i
        dist = next_mult - m
        if dist > min_L:
            min_L = dist
            
    for L in range(max(min_L, 1), max_L + 1):
        if HAS_SCIPY:
            if _has_matching_scipy(m, L, n):
                return L
        else:
            if _has_matching_python(m, L, n):
                return L
    return max_L

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m).
    """
    deadline = time.time() + 85  # 85s deadline with 5s safety margin
    best_m = 0
    best_f = 0
    
    # Search all residues mod 27720
    for m in range(L_CM):
        if time.time() > deadline:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f >= 144:
                break
                
    return int(best_m)
