"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

def _build_rank36_decomposition():
    """
    Build explicit rank-36 decomposition for 2x3 * 3x8.
    Splits 3x8 into four 3x2 blocks.
    Each 3x2 block is decomposed as (2x1 * 1x2) + (2x2 * 2x2).
    Rank(2x1x2) = 2.
    Rank(2x2x2) = 7 (Strassen).
    Total rank per block = 9.
    Total rank = 36.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 36))
    V = np.zeros((m*p, 36))
    W = np.zeros((p*n, 36))
    
    col_offset = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Part 1: A_{2x1} * B_{1x2} (Rank 2)
        # A_1 is A[:, 0] (col 0 of A). Indices 0, 3.
        # B_1 is B[0, k0:k1+1] (row 0 of B). Indices 8+k0, 8+k1.
        
        # Term 1: A_1[0] * B_1 -> C[0, :]
        t = col_offset
        U[0, t] = 1
        V[8+k0, t] = 1
        V[8+k1, t] = 1
        W[k0*2+0, t] = 1
        W[k1*2+0, t] = 1
        col_offset += 1
        
        # Term 2: A_1[1] * B_1 -> C[1, :]
        t = col_offset
        U[3, t] = 1
        V[8+k0, t] = 1
        V[8+k1, t] = 1
        W[k0*2+1, t] = 1
        W[k1*2+1, t] = 1
        col_offset += 1
        
        # Part 2: A_{2x2} * B_{2x2} (Rank 7, Strassen)
        # A_2 is A[:, 1:3]. Indices 1, 2, 4, 5.
        # B_2 is B[1:3, k0:k1+1]. Indices 8+k0, 8+k1, 16+k0, 16+k1.
        
        # Strassen terms for 2x2 block
        # M1 = (A01+A12)(B10+B20)
        t = col_offset
        U[1, t] += 1; U[5, t] += 1
        V[8+k0, t] += 1; V[16+k0, t] += 1
        W[k0*2+0, t] += 1; W[k0*2+1, t] += 1; W[k1*2+0, t] += 1; W[k1*2+1, t] += 1
        col_offset += 1
        
        # M2 = (A11+A12)B10
        t = col_offset
        U[4, t] += 1; U[5, t] += 1
        V[8+k0, t] += 1
        W[k0*2+1, t] += 1; W[k1*2+1, t] += -1
        col_offset += 1
        
        # M3 = A01(B20-B21)
        t = col_offset
        U[1, t] += 1
        V[16+k0, t] += 1; V[16+k1, t] += -1
        W[k1*2+0, t] += 1; W[k1*2+1, t] += 1
        col_offset += 1
        
        # M4 = A12(B10-B11)
        t = col_offset
        U[5, t] += 1
        V[8+k0, t] += 1; V[8+k1, t] += -1
        W[k0*2+0, t] += 1; W[k0*2+1, t] += 1
        col_offset += 1
        
        # M5 = (A01+A02)B21
        t = col_offset
        U[1, t] += 1; U[2, t] += 1
        V[16+k1, t] += 1
        W[k0*2+0, t] += -1; W[k1*2+0, t] += 1
        col_offset += 1
        
        # M6 = (A11-A01)(B10+B20)
        t = col_offset
        U[4, t] += 1; U[1, t] += -1
        V[8+k0, t] += 1; V[16+k0, t] += 1
        W[k1*2+1, t] += 1
        col_offset += 1
        
        # M7 = (A02-A12)(B11+B21)
        t = col_offset
        U[2, t] += 1; U[5, t] += -1
        V[8+k1, t] += 1; V[16+k1, t] += 1
        W[k0*2+0, t] += 1
        col_offset += 1
        
    assert col_offset == 36
    return U.tolist(), V.tolist(), W.tolist()

def _als_decompose(rank, max_iters=200, deadline=None, init=None):
    """
    Alternating Least Squares for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    
    if init is not None:
        U = np.array(init[0], dtype=float)
        V = np.array(init[1], dtype=float)
        W = np.array(init[2], dtype=float)
        # Ensure shapes match
        if U.shape != (n*m, rank):
            U = U[:, :rank]
        if V.shape != (m*p, rank):
            V = V[:, :rank]
        if W.shape != (p*n, rank):
            W = W[:, :rank]
    else:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V[b, :] = x
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W[c, :] = x
                except:
                    pass
                    
        # Check convergence
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    err = abs(T[a,b,c] - 1.0)
                    if err > max_err:
                        max_err = err
        target = np.zeros((n*m, m*p, p*n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = True
        non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 140
    
    # First, try the explicit rank-36 decomposition
    try:
        U36, V36, W36 = _build_rank36_decomposition()
        if _verify(U36, V36, W36, tol=1e-6):
            rounded = _round_and_verify(U36, V36, W36)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 36
            else:
                best_U, best_V, best_W = U36, V36, W36
                best_rank = 36
    except Exception as e:
        pass
    
    # Try to find better with ALS
    # Prioritize ranks 35, 34, 33
    target_ranks = [35, 34, 33]
    
    if best_rank == 36:
        for rank in target_ranks:
            if time.time() > deadline:
                break
            
            # Try a few times
            for trial in range(5):
                if time.time() > deadline:
                    break
                
                # Initialize with a subset of the 44 solution if possible
                if rank < 44:
                    init = (best_U[:rank], best_V[:rank], best_W[:rank])
                else:
                    init = None
                
                result = _als_decompose(rank, max_iters=50, deadline=deadline, init=init)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                            break
                        else:
                            best_U, best_V, best_W = U, V, W
                            best_rank = rank
                            break
            
            if best_rank < 44:
                break
                
    return {"U": best_U, "V": best_V, "W": best_W}
