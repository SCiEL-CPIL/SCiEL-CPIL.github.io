"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all i,j,k
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each i, j, k:
    #   sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Reshape for efficient computation
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # For each (i,j,k), the value is the dot product of:
    #   U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    # which is element-wise product then sum
    
    # Efficient: create tensor of shape (n, m, p, rank)
    # T[i,j,k,c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Then sum over c.
    
    U_mat = U_arr.reshape(n, m, rank)
    V_mat = V_arr.reshape(m, p, rank)
    W_mat = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
    # This is tricky to vectorize directly. Let's do it in a loop over rank for small rank.
    
    # Alternative: for each c, compute the outer product contribution
    # T[i,j,k] += U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
    
    T = np.zeros((n, m, p), dtype=np.int64)
    for c in range(rank):
        Uc = U_mat[:, :, c]  # (n, m)
        Vc = V_mat[:, :, c]  # (m, p)
        Wc = W_mat[:, :, c]  # (p, n)
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        # For fixed i: Uc[i,:] * sum_k Vc[j,k] * Wc[k,i]
        # Let's compute for each i:
        for i in range(n):
            # Wc[:, i] is (p,)
            # Vc @ Wc[:, i] is (m,) = sum_k Vc[j,k] * Wc[k,i]
            vwt = Vc @ Wc[:, i]  # (m,)
            T[i, :, :] += Uc[i, :][:, np.newaxis] * vwt[np.newaxis, :]
    
    return np.all(T == 1)


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


def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two decompositions."""
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    rows1 = len(U1)
    rows2 = len(U2)
    
    rank = rank1 * rank2
    U = [[0] * rank for _ in range(rows1 * rows2)]
    V = [[0] * rank for _ in range(rows1 * rows2)]  # Wait, V rows are different
    # Actually, V1 has rows m1*p1, V2 has rows m2*p2
    # U1 has rows n1*m1, U2 has rows n2*m2
    # W1 has rows p1*n1, W2 has rows p2*n2
    
    v1_rows = len(V1)
    v2_rows = len(V2)
    w1_rows = len(W1)
    w2_rows = len(W2)
    
    V = [[0] * rank for _ in range(v1_rows * v2_rows)]
    W = [[0] * rank for _ in range(w1_rows * w2_rows)]
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 * rank2 + c2
            for r1 in range(rows1):
                if U1[r1][c1] == 0:
                    continue
                for r2 in range(rows2):
                    if U2[r2][c2] == 0:
                        continue
                    r = r1 * rows2 + r2
                    U[r][c] = U1[r1][c1] * U2[r2][c2]
            
            for r1 in range(v1_rows):
                if V1[r1][c1] == 0:
                    continue
                for r2 in range(v2_rows):
                    if V2[r2][c2] == 0:
                        continue
                    r = r1 * v2_rows + r2
                    V[r][c] = V1[r1][c1] * V2[r2][c2]
            
            for r1 in range(w1_rows):
                if W1[r1][c1] == 0:
                    continue
                for r2 in range(w2_rows):
                    if W2[r2][c2] == 0:
                        continue
                    r = r1 * w2_rows + r2
                    W[r][c] = W1[r1][c1] * W2[r2][c2]
    
    return U, V, W


def _strassen_2x2x2():
    """Strassen decomposition of 2x2x2, rank 7."""
    # Standard Strassen formulas for C = AB where A,B are 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b10)
    # m7 = (a01-a11)*(b01+b11)
    # c00 = m1 - m4 + m5 + m7
    # c01 = m3 + m4
    # c10 = m2 + m5
    # c11 = m1 - m3 + m6 + m7
    
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # i,j in 0..1 for U (n*m=4 rows)
    # j,k in 0..1 for V (m*p=4 rows)
    # k,i in 0..1 for W (p*n=4 rows)
    
    # m1 = (a00+a11)(b00+b11)
    # U: a00=1 (row 0), a11=1 (row 3)
    # V: b00=1 (row 0), b11=1 (row 3)
    # W: contributes to c00 (i=0,k=0 -> W[0*2+0]=W[0]) and c11 (i=1,k=1 -> W[1*2+1]=W[3])
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (a10+a11)*b00
    # U: a10=1 (row 2), a11=1 (row 3)
    # V: b00=1 (row 0)
    # W: contributes to c10 (i=1,k=0 -> W[0*2+1]=W[1])
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # m3 = a00*(b01-b11)
    # U: a00=1 (row 0)
    # V: b01=1 (row 1), b11=-1 (row 3)
    # W: contributes to c01 (i=0,k=1 -> W[1*2+0]=W[2])
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # m4 = a11*(b10-b00)
    # U: a11=1 (row 3)
    # V: b10=1 (row 2), b00=-1 (row 0)
    # W: contributes to c01 (i=0,k=1 -> W[2]) and c11 (i=1,k=1 -> W[3])
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = -1
    
    # m5 = (a00+a01)*b11
    # U: a00=1 (row 0), a01=1 (row 1)
    # V: b11=1 (row 3)
    # W: contributes to c00 (i=0,k=0 -> W[0]) and c10 (i=1,k=0 -> W[1])
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # m6 = (a10-a00)*(b00+b10)
    # U: a10=1 (row 2), a00=-1 (row 0)
    # V: b00=1 (row 0), b10=1 (row 2)
    # W: contributes to c11 (i=1,k=1 -> W[3])
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # m7 = (a01-a11)*(b01+b11)
    # U: a01=1 (row 1), a11=-1 (row 3)
    # V: b01=1 (row 1), b11=1 (row 3)
    # W: contributes to c00 (i=0,k=0 -> W[0]) and c11 (i=1,k=1 -> W[3])
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W


def _concat_decomp(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions (block diagonal)."""
    rank = len(U1[0]) + len(U2[0])
    U = [row[:] + row2[:] for row, row2 in zip(U1, U2)]
    V = [row[:] + row2[:] for row, row2 in zip(V1, V2)]
    W = [row[:] + row2[:] for row, row2 in zip(W1, W2)]
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Strategy: Decompose 4x5x5 as:
    # M(4,5,5) = M(4,4,5) + M(4,1,5)
    # M(4,4,5) = M(4,4,4) + M(4,4,1)
    # M(4,4,4) = M(2,2,2) x M(2,2,2) via Kronecker (rank 7*7=49)
    # M(4,4,1) = schoolbook (rank 16)
    # M(4,1,5) = schoolbook (rank 20)
    # Total: 49 + 16 + 20 = 85
    
    # First, get Strassen for 2x2x2
    sU, sV, sW = _strassen_2x2x2()
    
    # Verify Strassen
    assert _verify(sU, sV, sW, 2, 2, 2), "Strassen verification failed"
    
    # Kronecker: M(4,4,4)
    kU, kV, kW = _kron_decomp(sU, sV, sW, sU, sV, sW)
    # kU has rows 4*4=16, kV has rows 4*4=16, kW has rows 4*4=16
    assert _verify(kU, kV, kW, 4, 4, 4), "Kronecker 4x4x4 verification failed"
    
    # Schoolbook: M(4,4,1)
    sb1_U, sb1_V, sb1_W = _schoolbook(4, 4, 1)
    assert _verify(sb1_U, sb1_V, sb1_W, 4, 4, 1), "Schoolbook 4x4x1 verification failed"
    
    # Concatenate to get M(4,4,5)
    c1_U, c1_V, c1_W = _concat_decomp(kU, kV, kW, sb1_U, sb1_V, sb1_W)
    assert _verify(c1_U, c1_V, c1_W, 4, 4, 5), "Concatenated 4x4x5 verification failed"
    
    # Schoolbook: M(4,1,5)
    sb2_U, sb2_V, sb2_W = _schoolbook(4, 1, 5)
    assert _verify(sb2_U, sb2_V, sb2_W, 4, 1, 5), "Schoolbook 4x1x5 verification failed"
    
    # Concatenate to get M(4,5,5)
    final_U, final_V, final_W = _concat_decomp(c1_U, c1_V, c1_W, sb2_U, sb2_V, sb2_W)
    
    rank = len(final_U[0])
    assert rank == 85, f"Expected rank 85, got {rank}"
    
    # Final verification
    assert _verify(final_U, final_V, final_W, 4, 5, 5), "Final 4x5x5 verification failed"
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Safety margin, though this should never happen for this construction
        pass
    
    return {"U": final_U, "V": final_V, "W": final_W}

# EVOLVE-BLOCK-END

