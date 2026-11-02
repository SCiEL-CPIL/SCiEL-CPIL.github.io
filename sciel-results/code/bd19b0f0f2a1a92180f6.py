"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank15(n=2, m=2, p=5):
    """
    Attempt to construct a rank-15 decomposition.
    Uses Strassen (rank 7) for cols 0,1 and a custom rank-8 for cols 2,3,4.
    """
    rank = 15
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # --- Part 1: Strassen for cols 0, 1 (7 terms, indices 0-6) ---
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=5, C11=6
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[6][0] = 1
    
    # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[5][1] = 1; W[6][1] = -1
    
    # M3: A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[6][2] = 1
    
    # M4: A11(B10-B00) -> C00 += M4, C10 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[5][3] = 1
    
    # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[6][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # --- Part 2: Custom rank-8 for cols 2,3,4 (indices 7-14) ---
    # B indices: B02=4, B03=5, B04=6, B12=7, B13=8, B14=9
    # C indices: C02=2, C03=3, C04=4, C12=7, C13=8, C14=9
    
    # This is a 2x2x3 problem. We need 8 terms.
    # Let's try a structured approach:
    
    # N1: A00*(B02+B13) -> C02 += N1, C13 += N1
    U[0][7] = 1
    V[4][7] = 1; V[8][7] = 1
    W[2][7] = 1; W[8][7] = 1
    
    # N2: (A10+A11)*B02 -> C12 += N2, C13 -= N2
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[7][8] = 1; W[8][8] = -1
    
    # N3: A00*(B03-B02) -> C03 += N3, C02 -= N3
    U[0][9] = 1
    V[5][9] = 1; V[4][9] = -1
    W[3][9] = 1; W[2][9] = -1
    
    # N4: A11*(B12-B13) -> C12 += N4, C13 += N4
    U[3][10] = 1
    V[7][10] = 1; V[8][10] = -1
    W[7][10] = 1; W[8][10] = 1
    
    # N5: (A00+A01)*(B13-B03) -> C03 += N5, C02 -= N5... let me reconsider
    # Actually let me try a different structure for the 2x2x3 part
    
    # Let me use a different approach: split into two 2x2x1 + one 2x2x1
    # Col 2: schoolbook rank 4
    # Col 3: schoolbook rank 4  
    # Col 4: schoolbook rank 4
    # That's 12, too many.
    
    # Let me try: use Strassen-like for cols 2,3 (rank 7) + 1 for col 4? That's 8.
    # But Strassen for 2x2x2 is rank 7, so cols 2,3 in rank 7 + col 4 in rank 1 = 8.
    # But col 4 needs rank 2 minimum (A00*B04 + A01*B14 for C04, A10*B04 + A11*B14 for C14)
    # So that doesn't work directly.
    
    # Let me try a hybrid: 6 terms for cols 2,3 and 2 terms for col 4
    # Or 5 terms for cols 2,3 and 3 terms for col 4
    
    # Actually, let me try to be more systematic. The 2x2x3 tensor has 12 nonzero entries.
    # Lower bound: need at least 6 terms (since each term can cover at most 2 of the 12 in a structured way)
    # But realistically, we need more.
    
    # Let me try rank 8 with this structure:
    # N1: A00*B02 -> C02
    U[0][7] = 1
    V[4][7] = 1
    W[2][7] = 1
    
    # N2: A01*B12 -> C02
    U[1][8] = 1
    V[7][8] = 1
    W[2][8] = 1
    
    # N3: A10*B02 -> C12
    U[2][9] = 1
    V[4][9] = 1
    W[7][9] = 1
    
    # N4: A11*B12 -> C12
    U[3][10] = 1
    V[7][10] = 1
    W[7][10] = 1
    
    # N5: A00*B03 -> C03
    U[0][11] = 1
    V[5][11] = 1
    W[3][11] = 1
    
    # N6: A01*B13 -> C03
    U[1][12] = 1
    V[8][12] = 1
    W[3][12] = 1
    
    # N7: A10*B03 -> C13
    U[2][13] = 1
    V[5][13] = 1
    W[8][13] = 1
    
    # N8: A11*B13 -> C13
    U[3][14] = 1
    V[8][14] = 1
    W[8][14] = 1
    
    # But we still need col 4! We have no more terms. This won't work.
    
    # Let me restart the rank-15 attempt with a different strategy.
    # Maybe I should try to share terms between the Strassen block and the extension.
    
    return None  # This naive approach doesn't work, need search

def _construct_rank16(n=2, m=2, p=5):
    """
    Construct a rank-16 decomposition:
    - Strassen (rank 7) for B columns 0,1
    - Rank-9 decomposition for B columns 2,3,4
    Uses structured terms that share factors across columns.
    """
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # --- Part 1: Strassen for cols 0, 1 (7 terms, indices 0-6) ---
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[6][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[5][1] = 1; W[6][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[6][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[5][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[6][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # --- Part 2: Rank-9 for cols 2,3,4 (indices 7-15) ---
    # B indices: B02=4, B03=5, B04=6, B12=7, B13=8, B14=9
    # C indices: C02=2, C03=3, C04=4, C12=7, C13=8, C14=9
    #
    # Strategy: Use 3 "Strassen-like" pairs for (col2,col3) and 3 for col4
    # Actually, let's use a direct approach with sharing:
    #
    # Term 7: A00*B02 -> C02
    U[0][7] = 1; V[4][7] = 1; W[2][7] = 1
    # Term 8: A01*B12 -> C02  
    U[1][8] = 1; V[7][8] = 1; W[2][8] = 1
    # Term 9: A10*B02 -> C12
    U[2][9] = 1; V[4][9] = 1; W[7][9] = 1
    # Term 10: A11*B12 -> C12
    U[3][10] = 1; V[7][10] = 1; W[7][10] = 1
    # Term 11: A00*B03 -> C03
    U[0][11] = 1; V[5][11] = 1; W[3][11] = 1
    # Term 12: A01*B13 -> C03
    U[1][12] = 1; V[8][12] = 1; W[3][12] = 1
    # Term 13: A10*B03 -> C13
    U[2][13] = 1; V[5][13] = 1; W[8][13] = 1
    # Term 14: A11*B13 -> C13
    U[3][14] = 1; V[8][14] = 1; W[8][14] = 1
    # Term 15: (A00*B04 + A01*B14) -> C04, (A10*B04 + A11*B14) -> C14
    # Can't do both in one term. We need 4 more terms for col4 but only have 1.
    # This doesn't work. Rank 16 = 7 + 9, but 2x2x3 needs at least 12 in schoolbook.
    # So rank 16 is NOT achievable with this block structure.
    # Return None to let the caller try other options.
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _construct_rank19(n=2, m=2, p=5):
    """
    Construct a rank-19 decomposition:
    - Strassen (rank 7) for B columns 0,1
    - Schoolbook (rank 12) for B columns 2,3,4
    """
    rank = 19
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # --- Part 1: Strassen for cols 0, 1 (7 terms, indices 0-6) ---
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[6][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[5][1] = 1; W[6][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[6][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[5][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[6][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # --- Part 2: Schoolbook for cols 2,3,4 (12 terms, indices 7-18) ---
    base = 7
    for k in range(3):  # columns 2,3,4
        col = k + 2
        # C0k = A00*B0k + A01*B1k
        U[0][base] = 1; V[0*p+col][base] = 1; W[col*n+0][base] = 1
        U[1][base+1] = 1; V[1*p+col][base+1] = 1; W[col*n+0][base+1] = 1
        # C1k = A10*B0k + A11*B1k
        U[2][base+2] = 1; V[0*p+col][base+2] = 1; W[col*n+1][base+2] = 1
        U[3][base+3] = 1; V[1*p+col][base+3] = 1; W[col*n+1][base+3] = 1
        base += 4
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
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
    return {"U": U, "V": V, "W": W}

def _als_refine(U, V, W, T, R, n, m, p, max_iter=300, tol=1e-14):
    """Run ALS iterations to refine the decomposition."""
    prev_err = np.inf
    for it in range(max_iter):
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n*m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
        
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
        
        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
        
        # Compute error
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and try to rationalize entries."""
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(result, T, atol=1e-9):
        return None
    
    # Try to rationalize with various denominators
    for denom in [2, 4, 8, 16]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, T, atol=1e-9):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return as floats if valid
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _search_rank15(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-15 decomposition using ALS with structured initializations.
    """
    start_time = time.time()
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 15
    
    for trial in range(30):
        if time.time() - start_time > timeout - 5:
            break
        
        if trial < 10:
            # Random initialization
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
        elif trial < 20:
            # Structured: start from rank-18 and drop 3 columns
            U, V, W = _get_rank18_np(n, m, p)
            # Keep first 15 columns
            U = U[:, :15] + np.random.randn(n*m, 15) * 0.1
            V = V[:, :15] + np.random.randn(m*p, 15) * 0.1
            W = W[:, :15] + np.random.randn(p*n, 15) * 0.1
        else:
            # Different random scales
            scale = 0.1 + 0.5 * (trial - 20) / 10
            U = np.random.randn(n*m, R) * scale
            V = np.random.randn(m*p, R) * scale
            W = np.random.randn(p*n, R) * scale
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=200)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _get_rank18_np(n=2, m=2, p=5):
    """Get rank-18 decomposition as numpy arrays."""
    rank = 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W

def _search_rank17(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-17 decomposition by perturbing the rank-18 structure.
    Key insight: drop one schoolbook term for col 4 and let ALS redistribute.
    Also try algebraic elimination approaches.
    """
    start_time = time.time()
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 17
    
    for trial in range(50):
        if time.time() - start_time > timeout - 5:
            break
        
        if trial < 10:
            # Start from rank-18, drop last column, add noise
            U18, V18, W18 = _get_rank18_np(n, m, p)
            U = U18[:, :17] + np.random.randn(n*m, 17) * 0.3
            V = V18[:, :17] + np.random.randn(m*p, 17) * 0.3
            W = W18[:, :17] + np.random.randn(p*n, 17) * 0.3
        elif trial < 20:
            # Drop a middle schoolbook term
            U18, V18, W18 = _get_rank18_np(n, m, p)
            cols = list(range(15)) + [16, 17]
            U = U18[:, cols] + np.random.randn(n*m, 17) * 0.2
            V = V18[:, cols] + np.random.randn(m*p, 17) * 0.2
            W = W18[:, cols] + np.random.randn(p*n, 17) * 0.2
        elif trial < 30:
            # Random perturbation of rank-18 with varying noise
            U18, V18, W18 = _get_rank18_np(n, m, p)
            noise_scale = 0.3 + 0.05 * (trial - 20)
            U = U18[:, :17] + np.random.randn(n*m, 17) * noise_scale
            V = V18[:, :17] + np.random.randn(m*p, 17) * noise_scale
            W = W18[:, :17] + np.random.randn(p*n, 17) * noise_scale
        elif trial < 40:
            # Structured: Strassen(7) + Strassen(7) + 3 terms for col4
            # Try to find 3 terms that handle col4 with help from other blocks
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            # First 14 terms: two Strassen blocks
            U18, V18, W18 = _get_rank18_np(n, m, p)
            U[:, :14] = U18[:, :14]
            V[:, :14] = V18[:, :14]
            W[:, :14] = W18[:, :14]
            # Last 3 terms: random initialization for col4
            U[:, 14:] = np.random.randn(n*m, 3) * 0.5
            V[:, 14:] = np.random.randn(m*p, 3) * 0.5
            W[:, 14:] = np.random.randn(p*n, 3) * 0.5
            # Force col4 entries in last 3 terms
            for r in range(14, 17):
                W[4*n+0, r] += 1.0  # C04
                W[4*n+1, r] += 1.0  # C14
        else:
            # Pure random
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=300)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _search_rank14(n=2, m=2, p=5, timeout=60):
    """
    Search for a rank-14 decomposition using ALS with structured initializations.
    """
    start_time = time.time()
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 14
    
    for trial in range(15):
        if time.time() - start_time > timeout - 3:
            break
        
        if trial < 5:
            # Random
            U = np.random.randn(n*m, R) * 0.3
            V = np.random.randn(m*p, R) * 0.3
            W = np.random.randn(p*n, R) * 0.3
        elif trial < 10:
            # From rank-18, keep 14 cols
            U18, V18, W18 = _get_rank18_np(n, m, p)
            U = U18[:, :14] + np.random.randn(n*m, 14) * 0.2
            V = V18[:, :14] + np.random.randn(m*p, 14) * 0.2
            W = W18[:, :14] + np.random.randn(p*n, 14) * 0.2
        else:
            # From rank-15 random
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=150)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _construct_rank15_structured(n=2, m=2, p=5):
    """
    Construct a rank-15 decomposition:
    - Strassen (rank 7) for B columns 0,1
    - 8 structured terms for B columns 2,3,4 using column sharing
    """
    rank = 15
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # --- Part 1: Strassen for cols 0, 1 (7 terms, indices 0-6) ---
    # B: B00=0, B01=1, B10=2, B11=3
    # C: C00=0, C01=1, C10=5, C11=6
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[6][0] = 1
    
    # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[5][1] = 1; W[6][1] = -1
    
    # M3: A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[6][2] = 1
    
    # M4: A11(B10-B00) -> C00 += M4, C10 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[5][3] = 1
    
    # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[6][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # --- Part 2: 8 terms for cols 2,3,4 (indices 7-14) ---
    # B: B02=4, B03=5, B04=6, B12=7, B13=8, B14=9
    # C: C02=2, C03=3, C04=4, C12=7, C13=8, C14=9
    
    # T1: A00*(B03+B04) -> C03 += A00*B03, C04 += A00*B04
    U[0][7] = 1
    V[5][7] = 1; V[6][7] = 1
    W[3][7] = 1; W[4][7] = 1
    
    # T2: A01*(B13+B14) -> C03 += A01*B13, C04 += A01*B14
    U[1][8] = 1
    V[8][8] = 1; V[9][8] = 1
    W[3][8] = 1; W[4][8] = 1
    
    # T3: A10*(B03+B04) -> C13 += A10*B03, C14 += A10*B04
    U[2][9] = 1
    V[5][9] = 1; V[6][9] = 1
    W[8][9] = 1; W[9][9] = 1
    
    # T4: A11*(B13+B14) -> C13 += A11*B13, C14 += A11*B14
    U[3][10] = 1
    V[8][10] = 1; V[9][10] = 1
    W[8][10] = 1; W[9][10] = 1
    
    # T5: A00*B02 -> C02
    U[0][11] = 1
    V[4][11] = 1
    W[2][11] = 1
    
    # T6: A01*B12 -> C02
    U[1][12] = 1
    V[7][12] = 1
    W[2][12] = 1
    
    # T7: A10*B02 -> C12
    U[2][13] = 1
    V[4][13] = 1
    W[7][13] = 1
    
    # T8: A11*B12 -> C12
    U[3][14] = 1
    V[7][14] = 1
    W[7][14] = 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try structured rank 15 first (deterministic)
    result = _construct_rank15_structured(n, m, p)
    if result is not None:
        return result
    
    # Try rank 14 search (short)
    if time.time() < deadline - 100:
        result = _search_rank14(n, m, p, timeout=min(20, deadline - time.time() - 100))
        if result is not None:
            return result
    
    # Try rank 15 search (in case structured fails)
    if time.time() < deadline - 70:
        result = _search_rank15(n, m, p, timeout=min(35, deadline - time.time() - 70))
        if result is not None:
            return result
    
    # Try rank 17 search
    if time.time() < deadline - 35:
        result = _search_rank17(n, m, p, timeout=min(30, deadline - time.time() - 35))
        if result is not None:
            return result
    
    # Try rank 18
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
