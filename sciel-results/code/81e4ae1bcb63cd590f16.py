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
    - Custom rank-9 decomposition for B columns 2,3,4
    """
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # --- Part 1: Strassen for cols 0, 1 (7 terms) ---
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=5, C11=6
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[6][0] = 1
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[5][1] = 1; W[6][1] = -1
    
    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[6][2] = 1
    
    # M4: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[5][3] = 1
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[6][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # --- Part 2: Custom rank-9 for cols 2,3,4 ---
    # B indices: B02=4, B03=5, B04=6, B12=7, B13=8, B14=9
    # C indices: C02=2, C03=3, C04=4, C12=7, C13=8, C14=9
    
    # N1: A00*(B02+B13) -> C02, C13
    U[0][7] = 1
    V[4][7] = 1; V[8][7] = 1
    W[2][7] = 1; W[8][7] = 1
    
    # N2: (A10+A11)*B02 -> C12, C13
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[7][8] = 1; W[8][8] = -1
    
    # N3: (A00+A01)*B13 -> C03, C13
    U[0][9] = 1; U[1][9] = 1
    V[8][9] = 1
    W[3][9] = 1; W[8][9] = 1
    
    # N4: A11*(B03-B02) -> C03
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[3][10] = 1
    
    # N5: A00*(B12-B13) -> C02, C12
    U[0][11] = 1
    V[7][11] = 1; V[8][11] = -1
    W[2][11] = -1; W[7][11] = 1
    
    # N6: (A10-A00)*B02 -> C12
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1
    W[7][12] = 1
    
    # N7: (A01-A11)*B13 -> C03, C13
    U[1][13] = 1; U[3][13] = -1
    V[8][13] = 1
    W[3][13] = 1; W[8][13] = -1
    
    # N8: A00*B04 -> C04
    U[0][14] = 1
    V[6][14] = 1
    W[4][14] = 1
    
    # N9: A10*B04 -> C14... wait, need A01*B14 and A11*B14 too
    # Let me fix this. C14 = A10*B04 + A11*B14
    # C04 = A00*B04 + A01*B14
    
    # N8: A00*B04 -> C04
    U[0][14] = 1
    V[6][14] = 1
    W[4][14] = 1
    
    # N9: A10*B04 -> C14
    U[2][15] = 1
    V[6][15] = 1
    W[9][15] = 1

    # But we're missing A01*B14 for C04 and A11*B14 for C14!
    # This rank-16 construction is incomplete. Let me use the verified one from the original.
    
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

def _rational_approx(x, max_denom=100):
    """Approximate x as a fraction with small denominator using continued fractions."""
    if abs(x) < 1e-12:
        return 0.0
    # Try simple denominators
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 100]:
        n_val = round(x * d)
        if abs(x - n_val / d) < 1e-10:
            return n_val / d
    # Continued fraction approach
    from fractions import Fraction
    try:
        frac = Fraction(x).limit_denominator(max_denom)
        if abs(x - float(frac)) < 1e-10:
            return float(frac)
    except:
        pass
    return x

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and try to rationalize entries."""
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(result, T, atol=1e-8):
        return None
    
    # Try to rationalize each entry individually
    U_s = np.array([[ _rational_approx(x) for x in row] for row in U])
    V_s = np.array([[ _rational_approx(x) for x in row] for row in V])
    W_s = np.array([[ _rational_approx(x) for x in row] for row in W])
    
    result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
    
    if np.allclose(result_s, T, atol=1e-7):
        U_list = [[x if isinstance(x, int) else float(x) for x in row] for row in U_s.tolist()]
        V_list = [[x if isinstance(x, int) else float(x) for x in row] for row in V_s.tolist()]
        W_list = [[x if isinstance(x, int) else float(x) for x in row] for row in W_s.tolist()]
        
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                        mat[i][j] = int(round(mat[i][j]))
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try with small denominator grid
    for denom in [2, 4, 8, 16, 32]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, T, atol=1e-7):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
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

def _search_rank16(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-16 decomposition.
    Uses Strassen for first 2 cols (rank 7) and random/structured for the rest (rank 9).
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
    
    R = 16
    R_strassen = 7
    
    for trial in range(30):
        if time.time() - start_time > timeout - 5:
            break
        
        # Initialize first 7 columns with Strassen for cols 0,1
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Strassen block
        U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
        U[:, :R_strassen] = U_str
        V[:, :R_strassen] = V_str
        W[:, :R_strassen] = W_str
        
        # Initialize remaining 9 columns randomly
        noise_scale = 0.5 + 0.1 * (trial % 5)
        U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
        V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
        W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=300)
        
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

def _algebraic_search_rank17(n=2, m=2, p=5, timeout=120):
    """
    Algebraic search for rank-17 decomposition.
    Uses Strassen for cols 0,1 (rank 7) and searches for 10 more terms
    with independent coefficients per factor.
    """
    start_time = time.time()
    R = 17
    n, m, p = 2, 2, 5
    
    # Get Strassen block for cols 0,1
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    
    # Compute what Strassen covers for each (i,j,k)
    # For cols 0,1: should give 1.0; for cols 2,3,4: should give 0.0
    strassen_val = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(7):
                    val += U_str[i*m+j, r] * V_str[j*p+k, r] * W_str[k*n+i, r]
                strassen_val[(i,j,k)] = val
    
    # Verify Strassen handles cols 0,1 correctly
    for i in range(n):
        for j in range(m):
            for k in [0, 1]:
                if abs(strassen_val[(i,j,k)] - 1.0) > 1e-10:
                    return None
            for k in [2, 3, 4]:
                if abs(strassen_val[(i,j,k)]) > 1e-10:
                    return None
    
    # For cols 2,3,4, we need 10 additional terms such that:
    # sum_{t=0}^{9} U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i] = 1 for all (i,j,k) with k in {2,3,4}
    # and = 0 for k in {0,1}
    
    # Strategy: use structured patterns where each term has specific support
    # and solve for coefficients using nonlinear least squares or direct elimination
    
    # Define candidate structured patterns. Each term t has:
    #   u_support: list of u-indices where U_t is nonzero
    #   v_support: list of v-indices where V_t is nonzero  
    #   w_support: list of w-indices where W_t is nonzero
    #   u_coeff, v_coeff, w_coeff: the actual values (to be determined)
    
    # For efficiency, we'll try patterns where each factor has at most 2 nonzeros
    # and use the fact that the tensor is very sparse (only 12 nonzero entries in the 2x2x3 subproblem)
    
    # The 12 target entries are:
    # C02 = A00*B02 + A01*B12  (u=0, v=4, w=2) and (u=1, v=7, w=2)
    # C03 = A00*B03 + A01*B13  (u=0, v=5, w=3) and (u=1, v=8, w=3)
    # C04 = A00*B04 + A01*B14  (u=0, v=6, w=4) and (u=1, v=9, w=4)
    # C12 = A10*B02 + A11*B12  (u=2, v=4, w=7) and (u=3, v=7, w=7)
    # C13 = A10*B03 + A11*B13  (u=2, v=5, w=8) and (u=3, v=8, w=8)
    # C14 = A10*B04 + A11*B14  (u=2, v=6, w=9) and (u=3, v=9, w=9)
    
    # Try a rank-10 decomposition of the 2x2x3 tensor
    # Use a hybrid of Strassen-like and schoolbook
    
    patterns = [
        # Pattern A: 6 schoolbook for cols 2,3 + 4 for col 4
        {
            'terms': [
                # Col 2 (schoolbook)
                {'u': [0], 'v': [4], 'w': [2]},   # A00*B02 -> C02
                {'u': [1], 'v': [7], 'w': [2]},   # A01*B12 -> C02
                {'u': [2], 'v': [4], 'w': [7]},   # A10*B02 -> C12
                {'u': [3], 'v': [7], 'w': [7]},   # A11*B12 -> C12
                # Col 3 (schoolbook)
                {'u': [0], 'v': [5], 'w': [3]},   # A00*B03 -> C03
                {'u': [1], 'v': [8], 'w': [3]},   # A01*B13 -> C03
                {'u': [2], 'v': [5], 'w': [8]},   # A10*B03 -> C13
                {'u': [3], 'v': [8], 'w': [8]},   # A11*B13 -> C13
                # Col 4 (2 terms with sharing)
                {'u': [0,1], 'v': [6,9], 'w': [4]},  # (a*A00+b*A01)*(c*B04+d*B14) -> C04
                {'u': [2,3], 'v': [6,9], 'w': [9]},  # (e*A10+f*A11)*(g*B04+h*B14) -> C14
            ]
        },
        # Pattern B: Strassen-like for cols 2,3 (7 terms) + 3 for col 4
        {
            'terms': [
                {'u': [0,3], 'v': [4,8], 'w': [2,8]},   # (A00+A11)(B02+B13)
                {'u': [2,3], 'v': [4], 'w': [7,8]},     # (A10+A11)B02
                {'u': [0], 'v': [5,8], 'w': [3,8]},     # A00(B03-B13)
                {'u': [3], 'v': [7,4], 'w': [7,2]},     # A11(B12-B02)
                {'u': [0,1], 'v': [8], 'w': [3,8]},     # (A00+A01)B13
                {'u': [2,0], 'v': [4,5], 'w': [7,3]},   # (A10-A00)(B02+B03)
                {'u': [1,3], 'v': [7,8], 'w': [2,3]},   # (A01-A11)(B12+B13)
                {'u': [0], 'v': [6], 'w': [4]},         # A00*B04
                {'u': [1], 'v': [9], 'w': [4]},         # A01*B14
                {'u': [2,3], 'v': [6,9], 'w': [9]},     # (A10+A11)(B04+B14)
            ]
        },
        # Pattern C: Mixed with more sharing
        {
            'terms': [
                {'u': [0,1], 'v': [4,7], 'w': [2]},   # (A00+A01)(B02+B12) -> C02
                {'u': [2,3], 'v': [4,7], 'w': [7]},   # (A10+A11)(B02+B12) -> C12
                {'u': [0,1], 'v': [5,8], 'w': [3]},   # (A00+A01)(B03+B13) -> C03
                {'u': [2,3], 'v': [5,8], 'w': [8]},   # (A10+A11)(B03+B13) -> C13
                {'u': [0], 'v': [6], 'w': [4]},       # A00*B04 -> C04
                {'u': [1], 'v': [9], 'w': [4]},       # A01*B14 -> C04
                {'u': [2], 'v': [6], 'w': [9]},       # A10*B04 -> C14
                {'u': [3], 'v': [9], 'w': [9]},       # A11*B14 -> C14
                {'u': [0,3], 'v': [4,5], 'w': [2,3]}, # Correction terms
                {'u': [1,2], 'v': [7,8], 'w': [7,8]}, # Correction terms
            ]
        },
        # Pattern D: Diagonal grouping
        {
            'terms': [
                {'u': [0], 'v': [4,5,6], 'w': [2,3,4]},   # A00*(B02+B03+B04)
                {'u': [1], 'v': [7,8,9], 'w': [2,3,4]},   # A01*(B12+B13+B14)
                {'u': [2], 'v': [4,5,6], 'w': [7,8,9]},   # A10*(B02+B03+B04)
                {'u': [3], 'v': [7,8,9], 'w': [7,8,9]},   # A11*(B12+B13+B14)
                {'u': [0,1], 'v': [4,7], 'w': [2,7]},     # Cross correction
                {'u': [2,3], 'v': [5,8], 'w': [3,8]},     # Cross correction
                {'u': [0,3], 'v': [6,9], 'w': [4,9]},     # Cross correction
                {'u': [1,2], 'v': [4,8], 'w': [7,2]},     # Cross correction
                {'u': [0,1], 'v': [5,9], 'w': [3,4]},     # Cross correction
                {'u': [2,3], 'v': [7,6], 'w': [8,9]},     # Cross correction
            ]
        },
        # Pattern E: Try exploiting the structure more
        {
            'terms': [
                {'u': [0,1], 'v': [4,5,6], 'w': [2,3,4]},
                {'u': [2,3], 'v': [4,5,6], 'w': [7,8,9]},
                {'u': [0], 'v': [4], 'w': [2]},
                {'u': [1], 'v': [7], 'w': [2]},
                {'u': [2], 'v': [5], 'w': [8]},
                {'u': [3], 'v': [8], 'w': [8]},
                {'u': [0,3], 'v': [6,9], 'w': [4,9]},
                {'u': [1,2], 'v': [6,9], 'w': [9,4]},
                {'u': [0,1], 'v': [4,8], 'w': [2,3]},
                {'u': [2,3], 'v': [7,5], 'w': [7,8]},
            ]
        },
    ]
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    for pat_idx, pattern in enumerate(patterns):
        if time.time() - start_time > timeout - 5:
            break
        
        terms = pattern['terms']
        num_terms = len(terms)
        
        # For each term, we need to determine coefficients for each nonzero entry
        # Total unknowns: sum over terms of (|u_support| + |v_support| + |w_support|)
        
        # Build the system: for each (i,j,k) with k in {2,3,4}:
        #   sum_t [sum_{u in u_supp_t} U_t[u] * sum_{v in v_supp_t} V_t[v] * sum_{w in w_supp_t} W_t[w] * indicator(i,j,k in supports)] = 1
        # For k in {0,1}: same sum = 0
        
        # This is nonlinear. Use scipy.optimize or manual elimination.
        # For small systems, try grid search over rational values.
        
        # Alternative: use the fact that we can factor each term's contribution
        # and set up a system where each term has 3 scalar coefficients (one per factor)
        # if each factor has a single nonzero, or more if multiple.
        
        # Simplified approach: assume each term has the form:
        #   c_u * u_pattern * c_v * v_pattern * c_w * w_pattern
        # where c_u, c_v, c_w are scalars and u_pattern, v_pattern, w_pattern are 0/1 vectors
        
        # For terms with |u_supp|=1, |v_supp|=1, |w_supp|=1: 3 unknowns (but can normalize to 1)
        # For terms with multiple nonzeros: more unknowns
        
        # Let's just try: for each term, assign a single coefficient to all its u-entries,
        # another to all v-entries, another to all w-entries.
        # This gives 3*num_terms unknowns, but we can normalize one per term.
        
        # Actually, let's try a different approach: use scipy's least squares on the
        # bilinear form by fixing two sets and solving for the third, iterating.
        
        U_extra = np.zeros((n*m, num_terms))
        V_extra = np.zeros((m*p, num_terms))
        W_extra = np.zeros((p*n, num_terms))
        
        # Initialize with the pattern structure
        for t, term in enumerate(terms):
            for u in term['u']:
                U_extra[u, t] = 1.0
            for v in term['v']:
                V_extra[v, t] = 1.0
            for w in term['w']:
                W_extra[w, t] = 1.0
        
        # Now optimize the scaling factors
        # Each term t has scaling factors: su_t, sv_t, sw_t
        # We can normalize: set su_t = 1, optimize sv_t and sw_t
        
        # Total 2*num_terms = 20 unknowns
        
        from scipy.optimize import minimize
        
        def objective(params):
            if time.time() - start_time > timeout - 3:
                return 1e10
            sv = params[:num_terms]
            sw = params[num_terms:]
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target = 1.0 if k >= 2 else 0.0
                        computed = 0.0
                        for t in range(num_terms):
                            u_val = U_extra[i*m+j, t]
                            v_val = V_extra[j*p+k, t] * sv[t]
                            w_val = W_extra[k*n+i, t] * sw[t]
                            computed += u_val * v_val * w_val
                        err += (computed - target) ** 2
            return err
        
        x0 = np.ones(2 * num_terms)
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < 1e-10:
                sv = res.x[:num_terms]
                sw = res.x[num_terms:]
                
                # Build full decomposition
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                U[:, :7] = U_str
                V[:, :7] = V_str
                W[:, :7] = W_str
                
                for t in range(num_terms):
                    r = 7 + t
                    for u in terms[t]['u']:
                        U[u, r] = 1.0
                    for v in terms[t]['v']:
                        V[v, r] = sv[t]
                    for w in terms[t]['w']:
                        W[w, r] = sw[t]
                
                result = _check_and_rationalize(U, V, W, T, R, n, m, p)
                if result is not None:
                    return result
        except:
            continue
    
    return None

def _search_rank17(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-17 decomposition.
    First tries algebraic elimination, then falls back to ALS-based search.
    """
    start_time = time.time()
    
    # Try algebraic approach first
    if time.time() - start_time < timeout - 20:
        result = _algebraic_search_rank17(n, m, p, timeout=min(20, timeout - (time.time() - start_time)))
        if result is not None:
            return result
    
    # Fallback to ALS-based search with more trials
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 17
    R_strassen = 7
    U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
    
    for trial in range(50):
        if time.time() - start_time > timeout - 5:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        U[:, :R_strassen] = U_str
        V[:, :R_strassen] = V_str
        W[:, :R_strassen] = W_str
        
        # Use structured initialization: start from schoolbook-like and perturb
        noise_scale = 0.3 + 0.1 * (trial % 7)
        U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
        V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
        W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=200)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _get_strassen_block_np(n=2, m=2, p=5, col0=0, col1=1):
    """Get Strassen block for two columns as numpy arrays."""
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0*p+col0, 0] = 1; V[1*p+col1, 0] = 1
    W[col0*n+0, 0] = 1; W[col1*n+1, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0*p+col0, 1] = 1
    W[col0*n+1, 1] = 1; W[col1*n+1, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[0*p+col1, 2] = 1; V[1*p+col1, 2] = -1
    W[col1*n+0, 2] = 1; W[col1*n+1, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[1*p+col0, 3] = 1; V[0*p+col0, 3] = -1
    W[col0*n+0, 3] = 1; W[col0*n+1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[1*p+col1, 4] = 1
    W[col0*n+0, 4] = -1; W[col1*n+0, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0*p+col0, 5] = 1; V[0*p+col1, 5] = 1
    W[col1*n+1, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1*p+col0, 6] = 1; V[1*p+col1, 6] = 1
    W[col0*n+0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # First, get a valid baseline (rank 18) quickly
    baseline = _construct_rank18(n, m, p)
    
    # Try rank 17 search (includes algebraic + ALS)
    if time.time() < deadline - 15:
        result = _search_rank17(n, m, p, timeout=min(40, deadline - time.time() - 15))
        if result is not None:
            return result
    
    # Try rank 16 search
    if time.time() < deadline - 10:
        result = _search_rank16(n, m, p, timeout=min(10, deadline - time.time() - 10))
        if result is not None:
            return result
    
    # Return baseline rank 18
    if baseline is not None:
        return baseline
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
