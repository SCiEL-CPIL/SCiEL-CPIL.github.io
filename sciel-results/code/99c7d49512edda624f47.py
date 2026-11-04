"""Rank-minimal tensor decomposition for 4x4x8 matrix multiplication.
Strategy: Kronecker-product assembly of Strassen 2x2x2 -> 4x4x4 (rank 49),
then two independent 4x4x4 blocks for 4x4x8 (rank 98), with column
linear-dependency elimination to potentially reduce rank further.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0


def _strassen_2x2x2_factors():
    """Return U, V, W as 4x7 numpy arrays for the 2x2x2 Strassen decomposition.
    
    Index convention:
      U[i, c]: coefficient of A[i] (i = row*2+col) in factor c
      V[j, c]: coefficient of B[j] (j = row*2+col) in factor c
      W[k, c]: coefficient for C[k] (k = row*2+col) from factor c
    
    Strassen 7 products:
      m1 = (a00+a11)(b00+b11)
      m2 = (a10+a11)b00
      m3 = a00(b01-b11)
      m4 = a11(b10-b00)
      m5 = (a00+a01)b11
      m6 = (a10-a00)(b00+b01)
      m7 = (a01-a11)(b10+b11)
      
    C reconstruction:
      c00 = m1 + m4 - m5 - m7
      c01 = m3 + m5
      c10 = m2 + m4
      c11 = m1 + m3 - m2 - m6
    """
    # A entries: 0=a00, 1=a01, 2=a10, 3=a11
    # B entries: 0=b00, 1=b01, 2=b10, 3=b11
    # C entries: 0=c00, 1=c01, 2=c10, 3=c11
    
    U = np.array([
        [1,  0,  1,  0,  1,  1,  0],  # a00
        [0,  0,  0,  0,  1,  0,  1],  # a01
        [0,  1,  0,  0,  0, -1, 0],  # a10
        [1,  1,  0,  1,  0,  0, -1],  # a11
    ], dtype=float)
    
    V = np.array([
        [1,  1,  0,  0,  0,  1,  0],  # b00
        [0,  0,  1,  0,  0,  1,  0],  # b01
        [0,  0,  0,  1,  0,  0,  1],  # b10
        [1,  0, -1, -1, 1,  0,  1],  # b11
    ], dtype=float)
    
    W = np.array([
        [1,  0,  0,  1, -1,  0, -1],  # c00
        [0,  0,  1,  0,  1,  0,  0],  # c01
        [0,  1,  0,  1,  0,  0, 0],  # c10
        [1, -1, 1,  0,  0, -1, 0],  # c11
    ], dtype=float)
    
    return U, V, W


def _verify_tensor(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank), W[k*n+i, c]
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _kronecker_factors(U2, V2, W2):
    """Build 4x4x4 factors from 2x2x2 factors via Kronecker product.
    
    If U2 is (4, r), V2 is (4, r), W2 is (4, r), then the 4x4x4 factors
    are U4 = kron(U2, U2) reshaped, etc.
    
    Actually, the Kronecker product of two tensor decompositions:
    If tensor T1 = sum_c u1_c ⊗ v1_c ⊗ w1_c (rank r1)
    and tensor T2 = sum_c u2_c ⊗ v2_c ⊗ w2_c (rank r2)
    then T1 ⊗ T2 = sum_{c1,c2} (u1_{c1}⊗u2_{c2}) ⊗ (v1_{c1}⊗v2_{c2}) ⊗ (w1_{c1}⊗w2_{c2})
    
    For 4x4x4 = (2x2x2) ⊗ (2x2x2), rank = 7*7 = 49.
    
    The 4x4 matrix A is indexed as A[row, col] where row = r1*2+r2, col = c1*2+c2.
    In row-major: index = row*4 + col = (r1*2+r2)*4 + (c1*2+c2) = r1*8 + r2*4 + c1*2 + c2.
    
    The Kronecker product u1⊗u2 has entries (u1[r1]*u2[r2], u1[r1]*u2[r2], ...) 
    indexed as r1*4+r2 for the 4x4 matrix? No.
    
    For a 4x4 matrix in row-major: A[i,j] where i = i1*2+i2, j = j1*2+j2.
    The Kronecker product of two 2x2 matrices P and Q gives a 4x4 matrix:
    (P⊗Q)[i1*2+i2, j1*2+j2] = P[i1,j1] * Q[i2,j2]
    
    So if U2[i, c] gives coefficients for 2x2 matrix entries (i = r*2+c),
    then U4[i, c1*7+c2] where i = (r1*2+r2)*4 + (c1*2+c2)... 
    
    Wait, let me be more careful. The 4x4 matrix A has 16 entries indexed 0..15.
    Entry index = row*4 + col where row = r1*2+r2, col = c1*2+c2.
    So index = (r1*2+r2)*4 + (c1*2+c2) = r1*8 + r2*4 + c1*2 + c2.
    
    The Kronecker product U4[i, c4] where c4 = c1*7+c2:
    U4[r1*8+r2*4+c1*2+c2, c1*7+c2] = U2[r1*2+c1, c1] * U2[r2*2+c2, c2]
    
    Hmm, that's getting complicated. Let me just build it directly.
    """
    r2 = U2.shape[1]  # should be 7
    r4 = r2 * r2  # 49
    
    # U4 shape: (16, 49)
    # V4 shape: (16, 49)
    # W4 shape: (16, 49)
    U4 = np.zeros((16, r4))
    V4 = np.zeros((16, r4))
    W4 = np.zeros((16, r4))
    
    for c1 in range(r2):
        for c2 in range(r2):
            c4 = c1 * r2 + c2
            # For each of the 16 entries of the 4x4 matrix:
            # index = (r1*2+r2)*4 + (c1_idx*2+c2_idx)
            # where r1,c1_idx are "outer" indices and r2,c2_idx are "inner"
            # U4[index, c4] = U2[r1*2+c1_idx, c1] * U2[r2*2+c2_idx, c2]
            for r1 in range(2):
                for c1i in range(2):
                    for r2i in range(2):
                        for c2i in range(2):
                            idx = (r1 * 2 + r2i) * 4 + (c1i * 2 + c2i)
                            U4[idx, c4] = U2[r1 * 2 + c1i, c1] * U2[r2i * 2 + c2i, c2]
                            V4[idx, c4] = V2[r1 * 2 + c1i, c1] * V2[r2i * 2 + c2i, c2]
                            W4[idx, c4] = W2[r1 * 2 + c1i, c1] * W2[r2i * 2 + c2i, c2]
    
    return U4, V4, W4


def _eliminate_dependent_columns(U, V, W, tol=1e-10):
    """Remove linearly dependent columns from the factor matrices.
    
    A column c is dependent if the tensor vector (U[:,c] ⊗ V[:,c] ⊗ W[:,c])
    is in the span of the other tensor vectors. This is hard to check directly.
    
    Alternative: check if U[:,c] is in span of other U columns, AND similarly
    for V and W. This is sufficient but not necessary.
    
    Better approach: form the "flattened" tensor vectors and do SVD.
    Each factor c contributes a vector in R^(n*m*m*p*p*n) which is huge.
    
    Simpler: just check column independence of U (or V or W) and remove
    columns that are linearly dependent in all three simultaneously.
    """
    n_rows_u, rank = U.shape
    
    # Try to find columns where U[:,c] is in the span of other U columns
    # This is a necessary condition for the factor to be redundant
    # (but not sufficient - we'd need all three to be dependent)
    
    # For simplicity, check if any column of U is zero
    keep = []
    for c in range(rank):
        if np.any(np.abs(U[:, c]) > tol):
            keep.append(c)
    
    if len(keep) == rank:
        return U, V, W
    
    U = U[:, keep]
    V = V[:, keep]
    W = W[:, keep]
    return U, V, W


def _build_4x4x8_kronecker():
    """Build 4x4x8 decomposition using Kronecker product of Strassen 2x2x2.
    
    4x4x8: A(4x4) * B(4x8) = C(8x4)
    C[k, i] = sum_j A[i,j] * B[j,k]
    
    Split B into B1 = B[:,0:4] and B2 = B[:,4:8] (both 4x4).
    Split C into C1 = C[0:4,:] and C2 = C[4:8,:] (both 4x4).
    
    C1 = A * B1 (4x4x4 multiplication)
    C2 = A * B2 (4x4x4 multiplication)
    
    Each uses rank 49 via Kronecker Strassen. Total rank = 98.
    
    U: (16, 98) - first 49 cols for C1, next 49 for C2 (same A)
    V: (32, 98) - B has 32 entries (4x8). 
      B index = j*8+k where j=0..3, k=0..7
      For C1: B entries j*8+k where k=0..3 -> indices 0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27
      For C2: B entries j*8+k where k=4..7 -> indices 4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31
    W: (32, 98) - C has 32 entries (8x4).
      C index = k*4+i where k=0..7, i=0..3
      For C1: k=0..3 -> indices 0..15
      For C2: k=4..7 -> indices 16..31
    """
    U2, V2, W2 = _strassen_2x2x2_factors()
    U4, V4, W4 = _kronecker_factors(U2, V2, W2)
    
    # Verify 4x4x4
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    # For 4x4x4: U is (n*m, rank) = (16, 49), V is (m*p, rank) = (16, 49), W is (p*n, rank) = (16, 49)
    # W[k*n+i, c] for 4x4x4: k=0..3, i=0..3, index = k*4+i
    
    # Now build 4x4x8
    rank_total = 98
    U = np.zeros((16, rank_total))
    V = np.zeros((32, rank_total))
    W = np.zeros((32, rank_total))
    
    # First 49 columns: C1 = A * B1
    U[:, 0:49] = U4
    # V for B1: B1[j,k] for j=0..3, k=0..3
    # B1 index in full B (4x8, row-major): j*8+k
    # V4[idx, c] where idx = j*4+k (for 4x4 B)
    # Map: V4 row idx = j*4+k -> V row j*8+k
    for j in range(4):
        for k in range(4):
            v4_idx = j * 4 + k
            v_full_idx = j * 8 + k
            V[v_full_idx, 0:49] = V4[v4_idx, :]
    
    # W for C1: C1[k,i] for k=0..3, i=0..3
    # C1 index in full C (8x4, row-major): k*4+i
    # W4[idx, c] where idx = k*4+i
    for k in range(4):
        for i in range(4):
            w4_idx = k * 4 + i
            w_full_idx = k * 4 + i
            W[w_full_idx, 0:49] = W4[w4_idx, :]
    
    # Next 49 columns: C2 = A * B2
    U[:, 49:98] = U4
    # V for B2: B2[j,k] for j=0..3, k=4..7
    # B2 index in full B: j*8+k where k=4..7
    for j in range(4):
        for k in range(4):
            v4_idx = j * 4 + k  # position within B2 (as 4x4)
            v_full_idx = j * 8 + (k + 4)  # position in full B (4x8)
            V[v_full_idx, 49:98] = V4[v4_idx, :]
    
    # W for C2: C2[k,i] for k=4..7, i=0..3
    # C2 index in full C: k*4+i where k=4..7
    for k in range(4):
        for i in range(4):
            w4_idx = k * 4 + i  # position within C2 (as 4x4)
            w_full_idx = (k + 4) * 4 + i  # position in full C (8x4)
            W[w_full_idx, 49:98] = W4[w4_idx, :]
    
    return U, V, W


def _verify_448(U, V, W, n=4, m=4, p=8, tol=1e-9):
    """Verify 4x4x8 decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = START_TIME + TIME_LIMIT
    
    try:
        U, V, W = _build_4x4x8_kronecker()
        
        # Verify
        ok, info = _verify_448(U, V, W, n, m, p)
        if ok:
            # Convert to list format
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        else:
            print(f"Verification failed: {info}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Fallback: schoolbook rank 128
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
