"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True


def _round_to_half(arr):
    """Round each entry to the nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _column_triples(U_col, V_col, W_col, n, m, p):
    """Given column vectors, return the set of (i,j,k) triples covered."""
    triples = set()
    for i in range(n):
        for j in range(m):
            if U_col[i * m + j] != 0:
                for k in range(p):
                    if V_col[j * p + k] != 0 and W_col[k * n + i] != 0:
                        triples.add((i, j, k))
    return triples


def _try_binary_search(n, m, p, r, max_attempts=200, deadline=None):
    """
    Try to find a binary (0/1) rank-r decomposition.
    Uses a greedy + local search approach.
    """
    start = time.time()
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    total_triples = len(all_triples)
    
    for attempt in range(max_attempts):
        if deadline and time.time() > deadline - 3:
            break
        
        np.random.seed(42 + attempt * 7919)
        
        # Initialize with random binary columns
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Density heuristic: each column should cover ~total_triples/r triples
        target_per_col = total_triples / r
        # U density: if U[i,j,c]=1 for ~du rows, V for ~dv, W for ~dw
        # expected coverage ~ du*dv*dw / (n*m) * ... complex
        # Just use moderate density
        du = 0.3 + 0.2 * np.random.rand()
        dv = 0.15 + 0.15 * np.random.rand()
        dw = 0.15 + 0.15 * np.random.rand()
        
        U = (np.random.rand(n * m, r) < du).astype(float)
        V = (np.random.rand(m * p, r) < dv).astype(float)
        W = (np.random.rand(p * n, r) < dw).astype(float)
        
        # Compute coverage
        coverage = {t: 0 for t in all_triples}
        for c in range(r):
            for t in _column_triples(U[:, c], V[:, c], W[:, c], n, m, p):
                coverage[t] += 1
        
        # Local search: try to fix violations
        improved = True
        max_iters = 500
        iters = 0
        while improved and iters < max_iters:
            if deadline and time.time() > deadline - 3:
                break
            improved = False
            iters += 1
            
            # Find uncovered or over-covered triples
            errors = []
            for t in all_triples:
                if coverage[t] != 1:
                    errors.append(t)
            
            if not errors:
                return U.copy(), V.copy(), W.copy()
            
            # Pick a random error triple
            t = errors[np.random.randint(len(errors))]
            i, j, k = t
            
            if coverage[t] == 0:
                # Need to add coverage: find a column where we can set all three to 1
                # and it doesn't create new errors
                candidates = []
                for c in range(r):
                    # Check if setting U[i*m+j,c], V[j*p+k,c], W[k*n+i,c] to 1
                    # would help
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    
                    # Count how many of the three need to be set
                    need_u = 1 - u_val
                    need_v = 1 - v_val
                    need_w = 1 - w_val
                    
                    if need_u + need_v + need_w == 0:
                        continue  # Already all 1, but not covered? Shouldn't happen
                    
                    # Simulate: set the needed ones to 1
                    new_u = u_val or 1
                    new_v = v_val or 1
                    new_w = w_val or 1
                    
                    # Check impact on other triples
                    would_help = 0
                    would_hurt = 0
                    
                    # Triples affected by U[i*m+j,c] change
                    if need_u:
                        for k2 in range(p):
                            if V[j * p + k2, c] and W[k2 * n + i, c]:
                                t2 = (i, j, k2)
                                if coverage[t2] == 1:
                                    would_hurt += 1
                                elif coverage[t2] == 0:
                                    would_help += 1
                    
                    # Triples affected by V[j*p+k,c] change
                    if need_v:
                        for i2 in range(n):
                            if U[i2 * m + j, c] and W[k * n + i2, c]:
                                t2 = (i2, j, k)
                                if coverage[t2] == 1:
                                    would_hurt += 1
                                elif coverage[t2] == 0:
                                    would_help += 1
                    
                    # Triples affected by W[k*n+i,c] change
                    if need_w:
                        for j2 in range(m):
                            if U[i * m + j2, c] and V[j2 * p + k, c]:
                                t2 = (i, j2, k)
                                if coverage[t2] == 1:
                                    would_hurt += 1
                                elif coverage[t2] == 0:
                                    would_help += 1
                    
                    if would_help > would_hurt:
                        candidates.append((would_help - would_hurt, c))
                
                if candidates:
                    candidates.sort(reverse=True)
                    _, c = candidates[0]
                    # Apply the change
                    if U[i * m + j, c] == 0:
                        U[i * m + j, c] = 1
                        for k2 in range(p):
                            if V[j * p + k2, c] and W[k2 * n + i, c]:
                                coverage[(i, j, k2)] += 1
                    if V[j * p + k, c] == 0:
                        V[j * p + k, c] = 1
                        for i2 in range(n):
                            if U[i2 * m + j, c] and W[k * n + i2, c]:
                                coverage[(i2, j, k)] += 1
                    if W[k * n + i, c] == 0:
                        W[k * n + i, c] = 1
                        for j2 in range(m):
                            if U[i * m + j2, c] and V[j2 * p + k, c]:
                                coverage[(i, j2, k)] += 1
                    improved = True
            
            else:  # coverage[t] > 1
                # Need to reduce coverage: find a column contributing to t and set one factor to 0
                contributing = [c for c in range(r) 
                              if U[i * m + j, c] and V[j * p + k, c] and W[k * n + i, c]]
                
                if contributing:
                    c = contributing[np.random.randint(len(contributing))]
                    
                    # Try zeroing one of the three
                    best_option = None
                    best_score = -1
                    
                    for factor in ['u', 'v', 'w']:
                        # Count how many other triples would be affected
                        would_fix = 0
                        would_break = 0
                        
                        if factor == 'u':
                            for k2 in range(p):
                                if V[j * p + k2, c] and W[k2 * n + i, c]:
                                    t2 = (i, j, k2)
                                    if coverage[t2] > 1:
                                        would_fix += 1
                                    elif coverage[t2] == 1:
                                        would_break += 1
                        elif factor == 'v':
                            for i2 in range(n):
                                if U[i2 * m + j, c] and W[k * n + i2, c]:
                                    t2 = (i2, j, k)
                                    if coverage[t2] > 1:
                                        would_fix += 1
                                    elif coverage[t2] == 1:
                                        would_break += 1
                        else:  # w
                            for j2 in range(m):
                                if U[i * m + j2, c] and V[j2 * p + k, c]:
                                    t2 = (i, j2, k)
                                    if coverage[t2] > 1:
                                        would_fix += 1
                                    elif coverage[t2] == 1:
                                        would_break += 1
                        
                        score = would_fix - would_break
                        if score > best_score:
                            best_score = score
                            best_option = factor
                    
                    if best_option and best_score > 0:
                        if best_option == 'u':
                            U[i * m + j, c] = 0
                            for k2 in range(p):
                                if V[j * p + k2, c] and W[k2 * n + i, c]:
                                    coverage[(i, j, k2)] -= 1
                        elif best_option == 'v':
                            V[j * p + k, c] = 0
                            for i2 in range(n):
                                if U[i2 * m + j, c] and W[k * n + i2, c]:
                                    coverage[(i2, j, k)] -= 1
                        else:
                            W[k * n + i, c] = 0
                            for j2 in range(m):
                                if U[i * m + j2, c] and V[j2 * p + k, c]:
                                    coverage[(i, j2, k)] -= 1
                        improved = True
    
    return None


def _try_structured_construction(n, m, p, deadline):
    """
    Try a structured construction exploiting the geometry.
    For 3x3 x 3x8, try to find a rank-r binary decomposition
    using a systematic approach.
    """
    start = time.time()
    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from lower bound to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 24
    # But practical lower bound is higher due to structural constraints
    # Try a range of ranks
    for r in range(24, min(72, 65)):
        if time.time() > deadline - 5:
            break
        
        result = _try_binary_search(n, m, p, r, max_attempts=50, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p):
                best_rank = r
                best_solution = (U.copy(), V.copy(), W.copy())
                # Try to go lower
                continue
        # If we found a solution, try a few more ranks below
        if best_rank < n * m * p:
            for r2 in range(24, best_rank):
                if time.time() > deadline - 5:
                    break
                result = _try_binary_search(n, m, p, r2, max_attempts=30, deadline=deadline)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, n, m, p):
                        best_rank = r2
                        best_solution = (U.copy(), V.copy(), W.copy())
            break
    
    if best_solution:
        return best_solution
    return None


def _try_half_integer_search(n, m, p, r, deadline):
    """
    Try to find a decomposition with half-integer entries.
    This allows more flexibility than pure binary.
    """
    start = time.time()
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for attempt in range(50):
        if deadline and time.time() > deadline - 3:
            break
        
        np.random.seed(1000 + attempt * 31)
        
        # Initialize with small random values
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Start from a perturbed schoolbook-like structure
        # Assign each triple to a random column
        col_assignment = {}
        for idx, t in enumerate(all_triples):
            col_assignment[t] = idx % r
        
        # Build initial solution
        for t, c in col_assignment.items():
            i, j, k = t
            U[i * m + j, c] += 1.0
            V[j * p + k, c] += 1.0
            W[k * n + i, c] += 1.0
        
        # Now try to reduce by finding columns that can be merged
        # This is complex, so just verify if it works
        if _verify(U, V, W, n, m, p):
            # Round to half-integers
            U_rounded = _round_to_half(U)
            V_rounded = _round_to_half(V)
            W_rounded = _round_to_half(W)
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard: must finish within 120s, use 110s as safety
    deadline = time.time() + 110
    
    # First, get the schoolbook solution as fallback
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try structured binary search
    result = _try_structured_construction(n, m, p, deadline)
    if result is not None:
        U2, V2, W2 = result
        if _verify(U2, V2, W2, n, m, p):
            best_rank = U2.shape[1]
            best_U, best_V, best_W = U2, V2, W2
    
    # Convert to lists and round to half-integers
    U_list = _round_to_half(best_U).tolist()
    V_list = _round_to_half(best_V).tolist()
    W_list = _round_to_half(best_W).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
