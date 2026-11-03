import time
import numpy as np
from scipy.optimize import least_squares, minimize

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _try_rank_reduction_optimization(U, V, W, n, m, p, target_rank, time_limit=10.0):
    """
    Attempt to reduce the rank of the given decomposition to target_rank
    using least squares optimization.
    """
    start_time = time.time()
    n_terms = n * m * p
    
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return U, V, W, current_rank
    
    # Initialize with the first target_rank columns
    U_init = np.array(U[:, :target_rank])
    V_init = np.array(V[:, :target_rank])
    W_init = np.array(W[:, :target_rank])
    
    def residuals(params):
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank : ]
        
        U_opt = U_flat.reshape(n*m, target_rank)
        V_opt = V_flat.reshape(m*p, target_rank)
        W_opt = W_flat.reshape(p*n, target_rank)
        
        # Vectorized computation of T_ijk = sum_r U_ir V_jr W_kr
        # We need to match indices carefully.
        # U is (n*m, R), V is (m*p, R), W is (p*n, R)
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_t = U_opt.reshape(n, m, target_rank)
        V_t = V_opt.reshape(m, p, target_rank)
        W_t = W_opt.reshape(p, n, target_rank)
        
        # Compute T
        # T[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
        # This is a bit tricky with einsum due to index mixing.
        # Let's use a loop over k for speed if p is small, or einsum.
        # einsum('ijr,jkr,kr->ijk', U_t, V_t, W_t) -- wait, W_t is (p, n, R) -> indices k, i, r.
        # So 'ijr,jkr,kir->ijk'
        
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        res = T - 1.0
        return res.flatten()
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        # Use least squares
        result = least_squares(residuals, x0, method='lm', max_nfev=2000, xtol=1e-10, ftol=1e-10)
        
        if result.success and result.cost < 1e-8:
            U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = result.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
            
            U_new = U_opt.tolist()
            V_new = V_opt.tolist()
            W_new = W_opt.tolist()
            
            if _verify(U_new, V_new, W_new, n, m, p, tol=1e-5):
                return U_new, V_new, W_new, target_rank
    except Exception:
        pass
        
    return U, V, W, current_rank

def _find_3x3x3_decomp(target_rank, time_limit=5.0):
    """
    Find a decomposition for 3x3x3 matrix multiplication with the given rank.
    Returns (U, V, W, rank) or (None, None, None, None) if not found.
    """
    n, m, p = 3, 3, 3
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try to reduce to target_rank
    if target_rank < n*m*p:
        U, V, W, rank = _try_rank_reduction_optimization(U, V, W, n, m, p, target_rank, time_limit)
        if rank == target_rank:
            return U, V, W, rank
            
    return None, None, None, None

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Strategy 1: Try to find a 3x3x3 decomposition of low rank and combine with 3x3x2.
    # Split k into [0,3], [3,6], [6,8]
    
    best_rank = n * m * p # 72
    best_result = {"U": _schoolbook(n, m, p)[0], "V": _schoolbook(n, m, p)[1], "W": _schoolbook(n, m, p)[2]}
    
    # First, establish a good baseline: 68 rank using 4 Strassen blocks
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    U_base, V_base, W_base = _build_strassen_embedding(n, m, p, blocks)
    if _verify(U_base, V_base, W_base, n, m, p):
        best_rank = len(U_base[0])
        best_result = {"U": U_base, "V": V_base, "W": W_base}
        
    # Try to improve 3x3x3 block rank
    # We want to find U3, V3, W3 for 3x3x3 with rank R3
    # Then combine two such blocks and one 3x3x2 block (rank 17)
    
    # Try ranks 26, 25, 24, 23, 22, 21, 20, 19
    # 26 is guaranteed (embedding)
    
    R3_found = None
    U3, V3, W3 = None, None, None
    
    # We have limited time. Let's try a few.
    # If we find R3=24, total rank = 24+24+17 = 65.
    # If we find R3=19, total rank = 19+19+17 = 55.
    
    # Let's try 24 first, as it's more likely to be found quickly than 19.
    # Actually, let's try 26 (known) to get a baseline for the combination, then try lower.
    
    # Get 3x3x2 decomposition (Rank 17)
    U2, V2, W2 = _build_strassen_embedding(3, 3, 2, [(0,0,0)])
    R2 = 17
    
    # Try to find 3x3x3 decompositions
    for R3 in [24, 23, 22, 21, 20, 19]:
        if _time_left() < 10:
            break
        U3, V3, W3, R3_actual = _find_3x3x3_decomp(R3, time_limit=_time_left() - 5)
        if R3_actual is not None and R3_actual == R3:
            R3_found = R3
            break
            
    if R3_found is not None:
        # Combine 3x3x3 + 3x3x3 + 3x3x2
        # Total Rank = R3 + R3 + R2
        total_rank = R3_found + R3_found + R2
        if total_rank < best_rank:
            # Construct the combined matrices
            # U_total: (3*3, total_rank)
            # V_total: (3*8, total_rank)
            # W_total: (8*3, total_rank)
            
            U_total = [[0]*total_rank for _ in range(9)]
            V_total = [[0]*total_rank for _ in range(24)]
            W_total = [[0]*total_rank for _ in range(24)]
            
            # Block 1: k in [0, 3]
            col_offset = 0
            for r in range(R3_found):
                for i in range(3):
                    for j in range(3):
                        U_total[i*3+j][col_offset+r] = U3[i*3+j][r]
                for j in range(3):
                    for k in range(3):
                        V_total[j*8+k][col_offset+r] = V3[j*3+k][r]
                for k in range(3):
                    for i in range(3):
                        W_total[k*3+i][col_offset+r] = W3[k*3+i][r]
            col_offset += R3_found
            
            # Block 2: k in [3, 6]
            for r in range(R3_found):
                for i in range(3):
                    for j in range(3):
                        U_total[i*3+j][col_offset+r] = U3[i*3+j][r]
                for j in range(3):
                    for k in range(3):
                        V_total[j*8+(3+k)][col_offset+r] = V3[j*3+k][r]
                for k in range(3):
                    for i in range(3):
                        W_total[(3+k)*3+i][col_offset+r] = W3[k*3+i][r]
            col_offset += R3_found
            
            # Block 3: k in [6, 8]
            for r in range(R2):
                for i in range(3):
                    for j in range(3):
                        U_total[i*3+j][col_offset+r] = U2[i*3+j][r]
                for j in range(3):
                    for k in range(2):
                        V_total[j*8+(6+k)][col_offset+r] = V2[j*3+k][r]
                for k in range(2):
                    for i in range(3):
                        W_total[(6+k)*3+i][col_offset+r] = W2[k*3+i][r]
                        
            if _verify(U_total, V_total, W_total, n, m, p):
                best_rank = total_rank
                best_result = {"U": U_total, "V": V_total, "W": W_total}
                
    # Try to reduce the best found rank further
    if _time_left() > 10:
        U_cur = best_result["U"]
        V_cur = best_result["V"]
        W_cur = best_result["W"]
        U_new, V_new, W_new, new_rank = _try_rank_reduction_optimization(U_cur, V_cur, W_cur, n, m, p, best_rank - 1, time_limit=_time_left() - 5)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_new, "V": V_new, "W": W_new}

    return best_result

