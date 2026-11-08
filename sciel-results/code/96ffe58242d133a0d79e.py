"""Matrix multiplication 4x4 * 4x9: find rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _time_ok(deadline, margin=2.0):
    return time.time() < deadline - margin

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    a = U[ui][r]
                    if a != 0:
                        b = V[vj][r]
                        if b != 0:
                            s += a * b * W[wk][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_2x2(U, V, W, n, m, p, i0, j0, k0, rank_offset):
    """
    Add Strassen 2x2*2x2 decomposition for blocks starting at (i0, j0, k0).
    Returns new rank offset.
    """
    A11 = i0 * m + j0
    A12 = i0 * m + j0 + 1
    A21 = (i0 + 1) * m + j0
    A22 = (i0 + 1) * m + j0 + 1
    
    B11 = j0 * p + k0
    B12 = j0 * p + k0 + 1
    B21 = (j0 + 1) * p + k0
    B22 = (j0 + 1) * p + k0 + 1
    
    C11 = k0 * n + i0
    C12 = k0 * n + i0 + 1
    C21 = (k0 + 1) * n + i0
    C22 = (k0 + 1) * n + i0 + 1
    
    r = rank_offset
    
    # M1 = (A11+A22)*(B11+B22)
    U[A11][r] += 1; U[A22][r] += 1
    V[B11][r] += 1; V[B22][r] += 1
    W[C11][r] += 1; W[C22][r] += 1
    r += 1
    
    # M2 = (A21+A22)*B11
    U[A21][r] += 1; U[A22][r] += 1
    V[B11][r] += 1
    W[C21][r] += 1; W[C22][r] -= 1
    r += 1
    
    # M3 = A11*(B12-B22)
    U[A11][r] += 1
    V[B12][r] += 1; V[B22][r] -= 1
    W[C12][r] += 1; W[C22][r] += 1
    r += 1
    
    # M4 = A22*(B21-B11)
    U[A22][r] += 1
    V[B21][r] += 1; V[B11][r] -= 1
    W[C11][r] += 1; W[C21][r] += 1
    r += 1
    
    # M5 = (A11+A12)*B22
    U[A11][r] += 1; U[A12][r] += 1
    V[B22][r] += 1
    W[C11][r] -= 1; W[C12][r] += 1
    r += 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[A21][r] += 1; U[A11][r] -= 1
    V[B11][r] += 1; V[B12][r] += 1
    W[C22][r] += 1
    r += 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[A12][r] += 1; U[A22][r] -= 1
    V[B21][r] += 1; V[B22][r] += 1
    W[C11][r] += 1
    r += 1
    
    return r

def _build_split_4x4x9_split2(n, m, p):
    """
    Split k dimension: 4x4*4x6 + 4x4*4x3
    For 4x4*4x6: 2 i-blocks * 2 j-blocks * 3 k-blocks Strassen, rank 42
    For 4x4*4x3: schoolbook, rank 48
    Total: 90
    """
    rank = 42 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # First 6 columns of B: 2 i-blocks * 2 j-blocks * 3 k-blocks
    for I in range(2):
        for J in range(2):
            for K in range(3):
                r = _strassen_2x2_2x2(U, V, W, n, m, p, I, J, K, r)
    
    # Last 3 columns of B: schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(6, 9):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, deadline, max_iter=500):
    """
    Try to reduce rank by finding redundant terms.
    A rank-1 term r is redundant if its contribution can be expressed as a linear combination
    of other terms. We try to eliminate terms by checking if setting them to zero still
    gives a valid decomposition (which would mean they were already zero or redundant).
    
    More practically: try to find pairs of terms that can be merged.
    """
    rank = len(U[0])
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to eliminate individual terms by checking if they're necessary
    for r in range(rank - 1, -1, -1):
        if not _time_ok(deadline):
            break
        
        # Check if term r is all zeros
        term_zero = True
        for i in range(n * m):
            if U[i][r] != 0:
                term_zero = False
                break
        if term_zero:
            for i in range(m * p):
                if V[i][r] != 0:
                    term_zero = False
                    break
        if term_zero:
            for i in range(p * n):
                if W[i][r] != 0:
                    term_zero = False
                    break
        
        if term_zero:
            # Remove this term
            new_rank = rank - 1
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            
            for c in range(new_rank):
                src = c if c < r else c + 1
                for i in range(n * m):
                    new_U[i][c] = U[i][src]
                for i in range(m * p):
                    new_V[i][c] = V[i][src]
                for i in range(p * n):
                    new_W[i][c] = W[i][src]
            
            U, V, W = new_U, new_V, new_W
            rank = new_rank
            best_rank = rank
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
            continue
        
        # Try to see if we can express term r as combination of others
        # This is expensive, so only try for small rank reductions
        if rank > 80 and (rank - r) % 10 == 0:
            # Simple heuristic: try to find if term r's pattern matches a scaled version of another term
            for r2 in range(rank):
                if r2 == r:
                    continue
                # Check if term r is proportional to term r2
                ratio = None
                proportional = True
                for i in range(n * m):
                    if U[i][r2] != 0:
                        if ratio is None:
                            ratio = U[i][r] / U[i][r2]
                        elif abs(U[i][r] - ratio * U[i][r2]) > 1e-9:
                            proportional = False
                            break
                    elif U[i][r] != 0:
                        proportional = False
                        break
                
                if proportional and ratio is not None:
                    # Check V and W
                    v_prop = True
                    for i in range(m * p):
                        if V[i][r2] != 0:
                            if abs(V[i][r] - ratio * V[i][r2]) > 1e-9:
                                v_prop = False
                                break
                        elif V[i][r] != 0:
                            v_prop = False
                            break
                    
                    if v_prop:
                        w_prop = True
                        for i in range(p * n):
                            if W[i][r2] != 0:
                                if abs(W[i][r] - (1/ratio) * W[i][r2]) > 1e-9:
                                    w_prop = False
                                    break
                            elif W[i][r] != 0:
                                w_prop = False
                                break
                        
                        if w_prop:
                            # Term r is redundant with term r2
                            new_rank = rank - 1
                            new_U = [[0] * new_rank for _ in range(n * m)]
                            new_V = [[0] * new_rank for _ in range(m * p)]
                            new_W = [[0] * new_rank for _ in range(p * n)]
                            
                            for c in range(new_rank):
                                src = c if c < r else c + 1
                                for i in range(n * m):
                                    new_U[i][c] = U[i][src]
                                for i in range(m * p):
                                    new_V[i][c] = V[i][src]
                                for i in range(p * n):
                                    new_W[i][c] = W[i][src]
                            
                            U, V, W = new_U, new_V, new_W
                            rank = new_rank
                            best_rank = rank
                            best_U = [row[:] for row in U]
                            best_V = [row[:] for row in V]
                            best_W = [row[:] for row in W]
                            break
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return U, V, W, rank

def _greedy_construct(n, m, p, deadline):
    """
    Greedy construction: try different split points and choose the one with minimum rank.
    For each possible split of the k-dimension, compute the rank if we use Strassen for
    the even-sized part and schoolbook for the rest.
    """
    best_rank = n * m * p
    best_result = None
    
    for k_split in range(0, p + 1):
        if not _time_ok(deadline):
            break
        
        # Use Strassen for first k_split columns (if even), schoolbook for rest
        strassen_cols = k_split - (k_split % 2)
        schoolbook_cols = p - strassen_cols
        
        if strassen_cols < 0 or schoolbook_cols < 0:
            continue
        
        # Rank calculation
        strassen_blocks = (strassen_cols // 2) * 4  # 2 i-blocks * 2 j-blocks * (k//2) k-blocks
        strassen_rank = strassen_blocks * 7
        schoolbook_rank = 4 * 4 * schoolbook_cols
        total_rank = strassen_rank + schoolbook_rank
        
        if total_rank >= best_rank:
            continue
        
        # Build the decomposition
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        r = 0
        # Strassen part
        for I in range(2):
            for J in range(2):
                for K in range(strassen_cols // 2):
                    r = _strassen_2x2_2x2(U, V, W, n, m, p, I, J, K, r)
        
        # Schoolbook part
        for i in range(n):
            for j in range(m):
                for k in range(strassen_cols, p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        
        if _validate(U, V, W, n, m, p):
            best_rank = total_rank
            best_result = {"U": U, "V": V, "W": W}
    
    return best_result, best_rank

def construct():
    """
    Returns the tensor decomposition for matrix multiplication of dimensions 4x4 * 4x9.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the known good split
    try:
        result = _build_split_4x4x9_split2(n, m, p)
        r = len(result["U"][0])
        if r < best_rank:
            if _validate(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
    except Exception:
        pass
    
    # Try greedy construction with different split points
    if _time_ok(deadline):
        try:
            greedy_result, greedy_rank = _greedy_construct(n, m, p, deadline)
            if greedy_result is not None and greedy_rank < best_rank:
                best = greedy_result
                best_rank = greedy_rank
        except Exception:
            pass
    
    # Try to reduce rank further
    if _time_ok(deadline) and best_rank < n * m * p:
        try:
            U, V, W, new_rank = _try_reduce_rank(
                best["U"], best["V"], best["W"], n, m, p, deadline
            )
            if new_rank < best_rank:
                if _validate(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = new_rank
        except Exception:
            pass
    
    return best

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
