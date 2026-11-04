"""Matrix multiplication 4x4 by 4x8 rank-minimal search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def _build_tensor():
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _schoolbook():
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

def _strassen_2x2x2():
    """
    Correct Strassen 2x2x2 decomposition.
    U: 4x7, V: 4x7, W: 4x7
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 0, 1, 1]
    ]
    return U, V, W

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    n1 = len(U1)
    n2 = len(U2)
    m1 = len(V1)
    m2 = len(V2)
    p1 = len(W1)
    p2 = len(W2)
    U = [[0] * r for _ in range(n1 * n2)]
    V = [[0] * r for _ in range(m1 * m2)]
    W = [[0] * r for _ in range(p1 * p2)]
    for i1 in range(n1):
        for i2 in range(n2):
            row = i1 * n2 + i2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    U[row][col] = U1[i1][c1] * U2[i2][c2]
    for j1 in range(m1):
        for j2 in range(m2):
            row = j1 * m2 + j2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    V[row][col] = V1[j1][c1] * V2[j2][c2]
    for k1 in range(p1):
        for k2 in range(p2):
            row = k1 * p2 + k2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    W[row][col] = W1[k1][c1] * W2[k2][c2]
    return U, V, W

def _strassen_4x4x4():
    U2, V2, W2 = _strassen_2x2x2()
    return _kron_decomp(U2, V2, W2, U2, V2, W2)

def _split_4x4x8(U44, V44, W44):
    """
    Construct 4x4x8 decomposition from 4x4x4.
    A is 4x4. B is 4x8 = [B1, B2] where B1, B2 are 4x4.
    C is 4x8 = [C1, C2].
    C1 = A B1, C2 = A B2.
    We use the same decomposition for both, but shift the W indices for the second block.
    """
    r = len(U44[0])
    # U is just duplicated for the two blocks? No, U depends on A, which is the same.
    # But the decomposition terms are independent.
    # We need to create a new set of rank-1 terms.
    # Term c in first block: U44[i,c], V44[j,c], W44[k,c] (k in 0..3)
    # Term c in second block: U44[i,c], V44[j,c], W44[k,c] (k in 4..7)
    
    U_new = []
    V_new = []
    W_new = []
    
    # First block (B1, C1)
    for i in range(len(U44)):
        U_new.append(U44[i][:])
    for j in range(len(V44)):
        V_new.append(V44[j][:])
    for k in range(len(W44)):
        W_new.append(W44[k][:])
        
    # Second block (B2, C2)
    for i in range(len(U44)):
        U_new.append(U44[i][:])
    for j in range(len(V44)):
        V_new.append(V44[j][:])
    for k in range(len(W44)):
        # Shift W indices by 4
        W_new.append(W44[k][:])
        
    # Now we have 2*r columns.
    # But wait, the W indices for the second block should be 4..7, not 0..3.
    # The W matrix rows are indexed by k*n+i.
    # For 4x4x4, k is 0..3.
    # For 4x4x8, k is 0..7.
    # The W matrix in the 4x4x4 decomposition has rows 0..15 (k*n+i).
    # k=0..3.
    # For the second block, we need W rows 16..31 (k=4..7).
    # So we need to construct W_new with zeros in the first half for the second block terms,
    # and zeros in the second half for the first block terms.
    
    r_total = 2 * r
    U_final = [[0] * r_total for _ in range(n * m)]
    V_final = [[0] * r_total for _ in range(m * p)]
    W_final = [[0] * r_total for _ in range(p * n)]
    
    # Fill first half of columns (0..r-1) with first block
    for i in range(n * m):
        for c in range(r):
            U_final[i][c] = U44[i][c]
            
    for j in range(m * p):
        # V44 has rows 0..15 (j*p+k for p=4).
        # For 4x4x8, V has rows 0..31.
        # First block uses j in 0..3, k in 0..3.
        # V44 row index is j*4+k.
        # In 4x4x8, this corresponds to j*8+k.
        # So we need to map V44 rows to V_final rows.
        # V44 row j4 corresponds to j_final = j4 * 2? No.
        # V44 is 4x4 (m=4, p=4). Rows are 0..15.
        # V_final is 4x8 (m=4, p=8). Rows are 0..31.
        # V44[j4, c] corresponds to V_final[j4*2, c]? No.
        # j4 = j*4+k.
        # j_final = j*8+k.
        # If j4 = j*4+k, then j = j4//4, k = j4%4.
        # j_final = (j4//4)*8 + (j4%4).
        
        pass # This mapping is complex. Let's build it directly.
        
    # Re-implementing _split_4x4x8 more carefully.
    
    # U44 is 16x49. U_final is 16x98.
    # First 49 cols of U_final = U44.
    # Next 49 cols of U_final = U44.
    for i in range(16):
        for c in range(49):
            U_final[i][c] = U44[i][c]
            U_final[i][c + 49] = U44[i][c]
            
    # V44 is 16x49. V_final is 32x98.
    # V44 row j4 (0..15) corresponds to V_final row j8 (0..31).
    # j4 = j*4+k. j8 = j*8+k.
    # First block (k in 0..3): V_final[j8, c] = V44[j4, c].
    # Second block (k in 4..7): V_final[j8, c+49] = V44[j4, c].
    for j4 in range(16):
        j = j4 // 4
        k = j4 % 4
        j8_1 = j * 8 + k
        j8_2 = j * 8 + (k + 4)
        for c in range(49):
            V_final[j8_1][c] = V44[j4][c]
            V_final[j8_2][c + 49] = V44[j4][c]
            
    # W44 is 16x49. W_final is 32x98.
    # W44 row k4 (0..15) corresponds to W_final row k8 (0..31).
    # k4 = k*4+i. k8 = k*4+i.
    # Wait, W index is k*n+i.
    # For 4x4x4, k is 0..3. k4 = k*4+i.
    # For 4x4x8, k is 0..7. k8 = k*4+i.
    # First block (k in 0..3): W_final[k8, c] = W44[k4, c].
    # Second block (k in 4..7): W_final[k8, c+49] = W44[k4, c].
    for k4 in range(16):
        k = k4 // 4
        i = k4 % 4
        k8_1 = k * 4 + i
        k8_2 = (k + 4) * 4 + i
        for c in range(49):
            W_final[k8_1][c] = W44[k4][c]
            W_final[k8_2][c + 49] = W44[k4][c]
            
    return U_final, V_final, W_final

def _verify_exact(U, V, W, T):
    rank = len(U[0]) if U else 0
    # Use numpy for speed, then check integers
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    
    # T_rec[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # This is a tensor contraction.
    # We can loop over c and add outer products.
    T_rec = np.zeros_like(T)
    for c in range(rank):
        T_rec += np.einsum('i,j,k->ijk', Ua[:, c], Va[:, c], Wa[:, c])
        
    return np.array_equal(T_rec, T)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    T = _build_tensor()
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try Strassen-based construction
    try:
        U44, V44, W44 = _strassen_4x4x4()
        U, V, W = _split_4x4x8(U44, V44, W44)
        
        # Verify
        if _verify_exact(U, V, W, T):
            # Convert to Fractions for the required interface
            U_frac = [[Fraction(x) for x in row] for row in U]
            V_frac = [[Fraction(x) for x in row] for row in V]
            W_frac = [[Fraction(x) for x in row] for row in W]
            
            # Double check with Fraction
            # (This is slow, so we only do it if float/int check passed)
            # The int check is sufficient for integer tensors.
            
            best_rank = len(U_frac[0])
            best = {"U": U_frac, "V": V_frac, "W": W_frac}
            print(f"Strassen 4x4x8: rank {best_rank}")
    except Exception as e:
        print(f"Strassen failed: {e}")
        
    return best
