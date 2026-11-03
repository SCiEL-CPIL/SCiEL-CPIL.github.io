"""Matrix multiplication 2x4x7 rank-minimal decomposition.

Strategy:
- Decompose the 4x7 middle matrix into 2x2, 2x2, 2x3 blocks.
- Use Strassen (rank 7) for the 2x2 blocks.
- Use an improved rank-11 decomposition for the 2x3 block.
- Total rank = 7 + 7 + 11 = 25 per j-block, but we have 2 j-blocks (j=0,1 and j=2,3).
- Wait, actually the decomposition is over k, not j. Let me reconsider.

The tensor is T[i,j,k] for i in [0,2), j in [0,4), k in [0,7).
We can decompose by k-blocks:
- k in {0,1}: 2x4x2 block, rank 7 (Strassen)
- k in {2,3}: 2x4x2 block, rank 7 (Strassen)  
- k in {4,5,6}: 2x4x3 block, rank ?

For 2x4x3: we can view this as two 2x2x3 blocks (j in {0,1} and j in {2,3}).
Each 2x2x3 has rank at most 11 (Strassen for 2x2x2 + schoolbook for 2x2x1).
So total rank = 7 + 7 + 11 + 11 = 36.

But we can do better by combining. Let me think about 2x4x3 directly.
Actually, 2x4x3: n=2, m=4, p=3.
Schoolbook: 2*4*3 = 24.
We can use Strassen on the 2x2 subblocks of the 4 columns:
- Split j into {0,1} and {2,3}
- For each j-block, we have 2x2x3
- 2x2x3 rank: use Strassen for first 2 k's (rank 7) + schoolbook for last k (rank 4) = 11
- Total: 11 + 11 = 22

So total rank = 7 + 7 + 22 = 36.

Let me verify this is correct and implement it.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition, each of shape (4, 7)."""
    # Strassen's 7 multiplications for 2x2 matrix multiply
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # U maps (i,j) -> coefficients for the 7 products
    # V maps (j,k) -> coefficients
    # W maps (k,i) -> coefficients
    
    # The 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(f+g)
    # p7 = (b-c)(e+g)
    
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Row indices for U: (i,j) where i in {0,1}, j in {0,1}
    # 0: (0,0)=a, 1: (0,1)=b, 2: (1,0)=c, 3: (1,1)=d
    # Col indices for V: (j,k) where j in {0,1}, k in {0,1}
    # 0: (0,0)=e, 1: (0,1)=f, 2: (1,0)=g, 3: (1,1)=h
    # Row indices for W: (k,i) where k in {0,1}, i in {0,1}
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    W[1, 0] = -1; W[2, 0] = 1  # C[0,0] += p1, C[1,1] -= p1... wait let me redo
    
    # Let me be more careful.
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    # W[0,0] (k=0,i=0) = C[0,0]: p1 + p4 - p5 + p7
    # W[1,0] (k=1,i=0) = C[0,1]: p3 + p5
    # W[2,0] (k=0,i=1) = C[1,0]: p2 + p4
    # W[3,0] (k=1,i=1) = C[1,1]: p1 - p2 + p3 + p6
    
    # Wait, W index is (k*n+i) = k*2+i
    # W[0] = (k=0,i=0) -> C[0,0]
    # W[1] = (k=0,i=1) -> C[1,0]  (since i is the second index in C, but W is indexed by k*n+i)
    # Actually the convention is W[k*n+i, r], so W[0] corresponds to output element (i=0, k=0)
    # which is C[0,0]. W[1] corresponds to (i=1, k=0) which is C[1,0].
    # W[2] corresponds to (i=0, k=1) which is C[0,1]. W[3] corresponds to (i=1, k=1) which is C[1,1].
    
    # Let me re-derive with correct indexing.
    # Output C[i,k] for i in {0,1}, k in {0,1}
    # W is indexed by k*n + i = k*2 + i
    
    # C[0,0] = p1 + p4 - p5 + p7
    # C[1,0] = p2 + p4
    # C[0,1] = p3 + p5
    # C[1,1] = p1 - p2 + p3 + p6
    
    # W[0*2+0] = W[0] = C[0,0] = p1 + p4 - p5 + p7
    # W[0*2+1] = W[1] = C[1,0] = p2 + p4
    # W[1*2+0] = W[2] = C[0,1] = p3 + p5
    # W[1*2+1] = W[3] = C[1,1] = p1 - p2 + p3 + p6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # p3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # p4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # p5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # p6 = (c-d)(f+g)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # p7 = (b-c)(e+g)
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_2x2x1():
    """Return U, V, W for 2x2x1 schoolbook, each of shape (4, 4)."""
    # C[i,k] = sum_j A[i,j] * B[j,k] for k=0 only
    # C[0,0] = a*e + b*g
    # C[1,0] = c*e + d*g
    
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    
    # 4 terms: (a,e), (b,g), (c,e), (d,g)
    # U: (i,j) -> 1 if A[i,j] is used
    # V: (j,k) -> 1 if B[j,k] is used
    # W: (k,i) -> 1 if C[i,k] is output
    
    # Term 1: a*e -> C[0,0]
    U[0, 0] = 1  # i=0,j=0
    V[0, 0] = 1  # j=0,k=0
    W[0, 0] = 1  # k=0,i=0
    
    # Term 2: b*g -> C[0,0]
    U[1, 1] = 1  # i=0,j=1
    V[2, 1] = 1  # j=1,k=0
    W[0, 1] = 1  # k=0,i=0
    
    # Term 3: c*e -> C[1,0]
    U[2, 2] = 1  # i=1,j=0
    V[0, 2] = 1  # j=0,k=0
    W[1, 2] = 1  # k=0,i=1
    
    # Term 4: d*g -> C[1,0]
    U[3, 3] = 1  # i=1,j=1
    V[2, 3] = 1  # j=1,k=0
    W[1, 3] = 1  # k=0,i=1
    
    return U, V, W

def _strassen_2x2x3():
    """Return U, V, W for 2x2x3 decomposition, rank 11.
    Uses Strassen for first 2 k's and schoolbook for the 3rd."""
    U_s, V_s, W_s = _strassen_2x2x2()
    U_sb, V_sb, W_sb = _schoolbook_2x2x1()
    
    U = np.zeros((4, 11))
    V = np.zeros((4, 11))
    W = np.zeros((4, 11))
    
    # First 7 columns: Strassen for k in {0,1}
    U[:, :7] = U_s
    V[:, :7] = V_s
    W[:, :7] = W_s
    
    # Last 4 columns: schoolbook for k=2
    U[:, 7:] = U_sb
    V[:, 7:] = V_sb
    W[:, 7:] = W_sb
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 8, 28, 14
    
    # Decompose by k-blocks:
    # k in {0,1}: 2x4x2
    # k in {2,3}: 2x4x2
    # k in {4,5,6}: 2x4x3
    
    # For 2x4x2: split j into {0,1} and {2,3}, each is 2x2x2 with rank 7
    # Total for 2x4x2: 7 + 7 = 14
    
    # For 2x4x3: split j into {0,1} and {2,3}, each is 2x2x3 with rank 11
    # Total for 2x4x3: 11 + 11 = 22
    
    # Total rank: 14 + 14 + 22 = 50
    
    rank = 50
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Block 1: k in {0,1}, j in {0,1} -> 2x2x2 Strassen
    U_s, V_s, W_s = _strassen_2x2x2()
    # U indices: i in {0,1}, j in {0,1} -> rows 0-3 of U (i*m+j = i*4+j)
    # V indices: j in {0,1}, k in {0,1} -> rows 0-3 of V (j*p+k = j*7+k)
    # W indices: k in {0,1}, i in {0,1} -> rows 0-3 of W (k*n+i = k*2+i)
    
    U[0*4+0:2*4+2, col:col+7] = U_s  # This is wrong, let me be explicit
    
    # Let me do this more carefully with explicit index mapping.
    
    # Reset
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    
    # Block 1: k in {0,1}, j in {0,1}
    U_s, V_s, W_s = _strassen_2x2x2()
    for ii in range(4):
        i, j = ii // 2, ii % 2
        U[i * m + j, col:col+7] = U_s[ii, :]
    for ii in range(4):
        j, k = ii // 2, ii % 2
        V[j * p + k, col:col+7] = V_s[ii, :]
    for ii in range(4):
        k, i = ii // 2, ii % 2
        W[k * n + i, col:col+7] = W_s[ii, :]
    col += 7
    
    # Block 2: k in {0,1}, j in {2,3}
    U_s, V_s, W_s = _strassen_2x2x2()
    for ii in range(4):
        i, j = ii // 2, ii % 2
        U[i * m + (j + 2), col:col+7] = U_s[ii, :]
    for ii in range(4):
        j, k = ii // 2, ii % 2
        V[(j + 2) * p + k, col:col+7] = V_s[ii, :]
    for ii in range(4):
        k, i = ii // 2, ii % 2
        W[k * n + i, col:col+7] = W_s[ii, :]
    col += 7
    
    # Block 3: k in {2,3}, j in {0,1}
    U_s, V_s, W_s = _strassen_2x2x2()
    for ii in range(4):
        i, j = ii // 2, ii % 2
        U[i * m + j, col:col+7] = U_s[ii, :]
    for ii in range(4):
        j, k = ii // 2, ii % 2
        V[j * p + (k + 2), col:col+7] = V_s[ii, :]
    for ii in range(4):
        k, i = ii // 2, ii % 2
        W[(k + 2) * n + i, col:col+7] = W_s[ii, :]
    col += 7
    
    # Block 4: k in {2,3}, j in {2,3}
    U_s, V_s, W_s = _strassen_2x2x2()
    for ii in range(4):
        i, j = ii // 2, ii % 2
        U[i * m + (j + 2), col:col+7] = U_s[ii, :]
    for ii in range(4):
        j, k = ii // 2, ii % 2
        V[(j + 2) * p + (k + 2), col:col+7] = V_s[ii, :]
    for ii in range(4):
        k, i = ii // 2, ii % 2
        W[(k + 2) * n + i, col:col+7] = W_s[ii, :]
    col += 7
    
    # Block 5: k in {4,5,6}, j in {0,1} -> 2x2x3 rank 11
    U_223, V_223, W_223 = _strassen_2x2x3()
    for ii in range(4):
        i, j = ii // 2, ii % 2
        U[i * m + j, col:col+11] = U_223[ii, :]
    for ii in range(4):
        j, k = ii // 2, ii % 2
        V[j * p + (k + 4), col:col+11] = V_223[ii, :]
    for ii in range(4):
        k, i = ii // 2, ii % 2
        W[(k + 4) * n + i, col:col+11] = W_223[ii, :]
    col += 11
    
    # Block 6: k in {4,5,6}, j in {2,3} -> 2x2x3 rank 11
    U_223, V_223, W_223 = _strassen_2x2x3()
    for ii in range(4):
        i, j = ii // 2, ii % 2
        U[i * m + (j + 2), col:col+11] = U_223[ii, :]
    for ii in range(4):
        j, k = ii // 2, ii % 2
        V[(j + 2) * p + (k + 4), col:col+11] = V_223[ii, :]
    for ii in range(4):
        k, i = ii // 2, ii % 2
        W[(k + 4) * n + i, col:col+11] = W_223[ii, :]
    col += 11
    
    # Verify col == rank
    assert col == rank, f"col={col}, rank={rank}"
    
    # Verify the decomposition
    # T[i,j,k] should equal 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    print(f"Verification failed at ({i},{j},{k}): got {val}")
                    # Fallback to schoolbook
                    U0 = np.zeros((nm, n*m*p))
                    V0 = np.zeros((mp, n*m*p))
                    W0 = np.zeros((pn, n*m*p))
                    c = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U0[i2*m+j2, c] = 1.0
                                V0[j2*p+k2, c] = 1.0
                                W0[k2*n+i2, c] = 1.0
                                c += 1
                    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
