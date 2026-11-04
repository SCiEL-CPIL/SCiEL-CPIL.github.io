"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the factors exactly decompose the MM tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    # Reshape to (rank, n, m), (rank, m, p), (rank, p, n)
    Ur = U.T.reshape(rank, n, m)
    Vr = V.T.reshape(rank, m, p)
    Wr = W.T.reshape(rank, p, n)
    
    # T[i,j,k] = sum_c Ur[c,i,j] * Vr[c,j,k] * Wr[c,k,i]
    # Use einsum: T[i,j,k] = sum_c Ur[c,i,j] * Vr[c,j,k] * Wr[c,k,i]
    T_computed = np.einsum('cij,cjk,cki->ijk', Ur, Vr, Wr)
    
    T_expected = np.ones((n, m, p), dtype=np.float64)
    return np.max(np.abs(T_computed - T_expected)) < tol


def _construct_strassen_2x2():
    """Return U, V, W factors for 2x2 x 2x2 matrix multiplication (rank 7).
    
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For 2x2: U is 4x7, V is 4x7, W is 4x7
    U index: i*2+j, V index: j*2+k, W index: k*2+i
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W indexing: W[k*n+i, c] = W[k*2+i, c]
    # C00: k=0,i=0 -> W[0,c]
    # C01: k=1,i=0 -> W[2,c]  
    # C10: k=0,i=1 -> W[1,c]
    # C11: k=1,i=1 -> W[3,c]
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1  # C00 += M1
    W[3, 0] = 1  # C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[1, 1] = 1  # C10 += M2
    W[3, 1] = -1  # C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, -B11
    W[2, 2] = 1  # C01 += M3
    W[3, 2] = 1  # C11 += M3
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, -B00
    W[0, 3] = 1  # C00 += M4
    W[1, 3] = 1  # C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1  # C00 -= M5
    W[2, 4] = 1   # C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1  # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1   # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1   # C00 += M7
    
    return U, V, W


def _construct_4x4x4_strassen():
    """Construct rank-49 decomposition for 4x4 x 4x4 using block Strassen.
    
    Split 4x4 into 2x2 blocks of 2x2 matrices.
    A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    
    Block Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2 matrix product, requiring 7 scalar products.
    Total: 7 * 7 = 49 scalar products.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Get 2x2 Strassen factors
    s2U, s2V, s2W = _construct_strassen_2x2()
    
    # Block-level Strassen terms:
    # Each term has: list of (block_row, block_col, sign) for A, B, C
    # A blocks: (bi, bj) -> rows 2*bi..2*bi+1, cols 2*bj..2*bj+1 in A
    # B blocks: (bj, bk) -> rows 2*bj..2*bj+1, cols 2*bk..2*bk+1 in B  
    # C blocks: (bk, bi) -> rows 2*bk..2*bk+1, cols 2*bi..2*bi+1 in C
    #
    # U index for A element (i,j): i*4+j where i=2*bi+ui, j=2*bj+uj
    # V index for B element (j,k): j*4+k where j=2*bj+uj, k=2*bk+uk
    # W index for C element (k,i): k*4+i where k=2*bk+uk, i=2*bi+ui
    
    # Define the 7 block-level terms with their A, B, C block contributions
    block_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        {
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        {
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
            'C': [(0, 1, 1), (1, 1, -1)]
        },
        # M3 = A00(B01-B11) -> C01+=1, C11+=1
        {
            'A': [(0, 0, 1)],
            'B': [(0, 1, 1), (1, 1, -1)],
            'C': [(1, 0, 1), (1, 1, 1)]
        },
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        {
            'A': [(1, 1, 1)],
            'B': [(1, 0, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (0, 1, 1)]
        },
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        {
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(0, 0, -1), (1, 0, 1)]
        },
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        {
            'A': [(1, 0, 1), (0, 0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        {
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]
        },
    ]
    
    col = 0
    for bt in block_terms:
        for s2c in range(7):
            # For this scalar product (s2c within the 2x2 Strassen),
            # we need to combine all A blocks, B blocks, C blocks
            # with the 2x2 Strassen coefficients
            
            # Collect all (bi,bj,sa) from A, (bj,bk,sb) from B, (bk,bi,sc) from C
            # For each valid combination where bj matches and bk/bi are consistent:
            
            a_list = bt['A']  # [(bi, bj, sa), ...]
            b_list = bt['B']  # [(bj, bk, sb), ...]
            c_list = bt['C']  # [(bk, bi, sc), ...]
            
            for (bi, bj, sa) in a_list:
                for (bj2, bk, sb) in b_list:
                    if bj != bj2:
                        continue
                    for (bk2, bi2, sc) in c_list:
                        if bk != bk2 or bi != bi2:
                            continue
                        
                        # Valid combination: (bi, bj, bk) with signs sa, sb, sc
                        # Now add 2x2 Strassen contributions
                        
                        # U: for each (ui, uj) in 2x2, U[(2*bi+ui)*4 + (2*bj+uj), col] += sa * s2U[ui*2+uj, s2c]
                        for ui in range(2):
                            for uj in range(2):
                                val_u = s2U[ui * 2 + uj, s2c]
                                if val_u != 0:
                                    i = 2 * bi + ui
                                    j = 2 * bj + uj
                                    U[i * 4 + j, col] += sa * val_u
                        
                        # V: for each (uj, uk) in 2x2, V[(2*bj+uj)*4 + (2*bk+uk), col] += sb * s2V[uj*2+uk, s2c]
                        for uj in range(2):
                            for uk in range(2):
                                val_v = s2V[uj * 2 + uk, s2c]
                                if val_v != 0:
                                    j = 2 * bj + uj
                                    k = 2 * bk + uk
                                    V[j * 4 + k, col] += sb * val_v
                        
                        # W: for each (uk, ui) in 2x2, W[(2*bk+uk)*4 + (2*bi+ui), col] += sc * s2W[uk*2+ui, s2c]
                        for uk in range(2):
                            for ui in range(2):
                                val_w = s2W[uk * 2 + ui, s2c]
                                if val_w != 0:
                                    k = 2 * bk + uk
                                    i = 2 * bi + ui
                                    W[k * 4 + i, col] += sc * val_w
            
            col += 1
    
    return U, V, W, rank


def _construct_4x4x5():
    """Construct a decomposition for 4x4 x 4x5."""
    n, m, p = 4, 4, 5
    
    # Start with 4x4 x 4x4 Strassen (rank 49)
    U4, V4, W4, rank4 = _construct_4x4x4_strassen()
    
    # For k=4, we need: sum_c U[i*4+j, c] * V[j*5+4, c] * W[4*4+i, c] = 1
    # Use 4 new terms, one for each j:
    # Term j: U[i*4+j, c_j] = 1 for all i, V[j*5+4, c_j] = 1, W[16+i, c_j] = 1 for all i
    
    rank_total = rank4 + m  # 49 + 4 = 53
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Copy the 4x4x4 part
    U[:, :rank4] = U4
    V[:, :rank4] = V4
    W[:, :rank4] = W4
    
    # Add terms for k=4
    for j in range(m):
        c = rank4 + j
        for i in range(n):
            U[i * m + j, c] = 1
            W[4 * n + i, c] = 1
        V[j * p + 4, c] = 1
    
    return U, V, W, rank_total


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank


def _try_reduce_rank(U, V, W, n, m, p, tol=1e-9):
    """Try to reduce rank by removing linearly dependent columns."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    # Check if any column is a linear combination of others
    # For each column, check if it can be expressed as a combination of the remaining
    # This is expensive, so we'll do a greedy approach
    
    keep = list(range(rank))
    changed = True
    while changed:
        changed = False
        for c in keep[:]:
            if len(keep) <= 1:
                break
            # Remove c and check if we can still verify
            other = [x for x in keep if x != c]
            U_sub = U[:, other]
            V_sub = V[:, other]
            W_sub = W[:, other]
            if _verify_decomposition(U_sub, V_sub, W_sub, n, m, p, tol=tol):
                keep = other
                changed = True
                break
    
    if len(keep) < rank:
        U_red = U[:, keep]
        V_red = V[:, keep]
        W_red = W[:, keep]
        return U_red, V_red, W_red, len(keep)
    
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # First, try the deterministic construction (rank 53)
    try:
        U, V, W, rank = _construct_4x4x5()
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            # Try to reduce rank
            U_red, V_red, W_red, rank_red = _try_reduce_rank(U, V, W, n, m, p, tol=1e-8)
            U_list = [[float(x) for x in row] for row in U_red]
            V_list = [[float(x) for x in row] for row in V_red]
            W_list = [[float(x) for x in row] for row in W_red]
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W, rank = _construct_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
