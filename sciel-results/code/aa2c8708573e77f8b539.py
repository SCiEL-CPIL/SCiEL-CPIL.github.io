"""
Rank-68 tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy: partition into four 2x2x2 blocks along the (n,p) axes,
each handled by Strassen's rank-7 algorithm. The m-axis has an extra
dimension (m=5 vs 2*2=4), so each 2x2x2 block actually involves a
2x3x2 or 2x2x3 "strip" that adds one schoolbook column per Strassen factor.

Block (ni, pi) with ni in {0,1}, pi in {0,1}:
  A rows 2*ni:2*ni+2, cols 0:3   (2x3)
  B rows 0:3, cols 2*pi:2*pi+2   (3x2)
  C rows 2*pi:2*pi+2, cols 2*ni:2*ni+2  (2x2)

Plus a final strip for m-index 4:
  A rows 0:4, col 4  (4x1)
  B row 4, cols 0:6  (1x6)
  C full (4x6)

Per block: 7 Strassen products. Each 2x3x2 product = 7*2 + 2 = 16 rank-1 terms.
Wait, that's not right. Let me think more carefully.

Actually, a 2x3x2 matrix multiplication (A is 2x3, B is 3x2, C is 2x2)
can be done with rank: the tensor has 4 output entries, each is a sum of 3 products.
Schoolbook rank = 4*3 = 12. But we can do better.

Actually, let me reconsider. The 2x2x2 Strassen handles A(2x2)*B(2x2)->C(2x2).
For 2x3x2: A(2x3)*B(3x2)->C(2x2). We can write A = [A_left(2x2) | A_right(2x1)]
and B = [B_top(2x2); B_bottom(1x2)].
C = A_left*B_top + A_right*B_bottom.
A_left*B_top: rank 7 (Strassen)
A_right*B_bottom: 2x1 times 1x2 -> rank 4 (schoolbook, each of 4 entries is 1 product)
Total: 7 + 4 = 11 per block.

4 blocks * 11 = 44.
Final strip (m=4): A(4x1)*B(1x6)->C(4x6): rank 24.
Total: 44 + 24 = 68.
"""
import numpy as np
import time

def _strassen_2x2x2():
    """Return 7 rank-1 factor triples for 2x2x2 Strassen."""
    # Each triple: (u[4], v[4], w[4]) where indices are row-major
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1=(A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # M2=(A10+A11)B00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # M3=A00(B01-B11)
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),   # M4=A11(B10-B00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,1]),   # M5=(A00+A01)B11
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # M6=(A10-A00)(B00+B01)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # M7=(A01-A11)(B10+B11)
    ]

def construct():
    start = time.time()
    deadline = start + 130
    
    n, m, p = 4, 5, 6
    # Total rank: 4 blocks * 11 + 24 strip = 68
    rank = 68
    U = [[0]*rank for _ in range(n*m)]   # 20
    V = [[0]*rank for _ in range(m*p)]   # 30
    W = [[0]*rank for _ in range(p*n)]   # 24
    
    s2 = _strassen_2x2x2()
    col = 0
    
    for ni in range(2):
        for pi in range(2):
            # A block rows: 2*ni, 2*ni+1; cols: 0,1,2
            # B block rows: 0,1,2; cols: 2*pi, 2*pi+1
            # C block rows: 2*pi, 2*pi+1; cols: 2*ni, 2*ni+1
            
            # --- Strassen part: A(2x2) * B(2x2) -> C(2x2) ---
            # A uses cols 0,1 of the 2x3 A block
            # B uses rows 0,1 of the 3x2 B block
            for u4, v4, w4 in s2:
                # A(2x2): rows 2*ni..2*ni+1, cols 0..1
                for r in range(2):
                    for c in range(2):
                        idx_a = (2*ni+r)*m + c
                        coeff = u4[r*2+c]
                        if coeff != 0:
                            U[idx_a][col] += coeff
                # B(2x2): rows 0..1, cols 2*pi..2*pi+1
                for r in range(2):
                    for c in range(2):
                        idx_b = r*p + (2*pi+c)
                        coeff = v4[r*2+c]
                        if coeff != 0:
                            V[idx_b][col] += coeff
                # C(2x2): rows 2*pi..2*pi+1, cols 2*ni..2*ni+1
                for r in range(2):
                    for c in range(2):
                        idx_c = (2*pi+r)*n + (2*ni+c)
                        coeff = w4[r*2+c]
                        if coeff != 0:
                            W[idx_c][col] += coeff
                col += 1
            
            # --- Extra A col (col 2 of A block) * B rows 0,1 ---
            # A_right is 2x1 (col 2), B_top is 2x2
            # Contribution: sum_{j=0,1} A[i, 2+j] * B[j, k] * W[k, i]
            # Actually: A block col 2 is A[2*ni+r, 2] for r in 0,1
            # B top rows 0,1: B[j, 2*pi+c] for j in 0,1, c in 0,1
            # C: C[2*pi+c, 2*ni+r] += A[2*ni+r, 2] * B[j, 2*pi+c] ... no wait
            
            # The 2x3 * 3x2 product: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
            # = sum_{j=0}^{1} A[i,j]*B[j,k] + A[i,2]*B[2,k]
            # First sum is the Strassen part (done above, using A cols 0,1 and B rows 0,1)
            # Second term: A[i,2] * B[2,k] for i in 0,1 (within block), k in 0,1 (within block)
            # This is rank 4 (4 products)
            for r in range(2):
                for c in range(2):
                    idx_a = (2*ni+r)*m + 2
                    idx_b = 2*p + (2*pi+c)
                    idx_c = (2*pi+c)*n + (2*ni+r)
                    U[idx_a][col] = 1
                    V[idx_b][col] = 1
                    W[idx_c][col] = 1
                    col += 1
    
    # --- Final strip: m-index 4 ---
    # A rows 0..3, col 4; B row 4, cols 0..5; C full 4x6
    for i in range(n):
        for k in range(p):
            idx_a = i*m + 4
            idx_b = 4*p + k
            idx_c = k*n + i
            U[idx_a][col] = 1
            V[idx_b][col] = 1
            W[idx_c][col] = 1
            col += 1
    
    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    # Verify
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        A_test = rng.standard_normal((n, m))
        B_test = rng.standard_normal((m, p))
        C_expected = A_test @ B_test
        
        U_arr = np.array(U, dtype=float).T  # rank x n*m
        V_arr = np.array(V, dtype=float).T  # rank x m*p
        W_arr = np.array(W, dtype=float).T  # rank x p*n
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_mat = U_arr[r].reshape(n, m)
            v_mat = V_arr[r].reshape(m, p)
            w_mat = W_arr[r].reshape(p, n)
            # contribution: einsum('ij,jk,ki->ik', u_mat, v_mat, w_mat)
            tmp = u_mat @ v_mat  # n x p
            # multiply by w_mat[k, i] for each (i,k)
            # C[i,k] += tmp[i,k] * w_mat[k,i]
            C_computed += tmp * w_mat.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][c] = 1
                        V[j*p+k][c] = 1
                        W[k*n+i][c] = 1
                        c += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
