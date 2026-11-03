import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = 3, 4, 4
    
    # Strategy: Use a rank-11 decomposition for 3x2x2 blocks, tiled over 2 column-blocks
    # For 3x2x2: rows 0,1 use Strassen (rank 7), row 2 uses trivial (rank 4)
    # Total per 3x2x2 block: rank 11
    # Two such blocks (columns 0-1 and 2-3): total rank 22... wait, that's for 3x2x2
    # But we need 3x4x4, so we need 4 blocks of 3x2x2? No...
    # 3x4x4: partition columns into 2 groups of 2, so 2 blocks of 3x2x2? No, 3x4x4 means
    # A is 3x4, B is 4x4, C is 3x4. Partition B's columns into 2 groups of 2.
    # Then we have 2 sub-problems: 3x4x2 each.
    # For 3x4x2: partition A's columns into 2 groups of 2.
    # Then 4 sub-problems: 3x2x2 each, each rank 11, total 44.
    
    # Let me try to find a rank-10 decomposition for 3x2x2
    # 3x2x2 tensor: T[i,j,k] where i in {0,1,2}, j in {0,1}, k in {0,1}
    # T[i,j,k] = delta_{i,k} * (A[i,j] * B[j,k])
    # So T[i,j,k] = 1 if i==k, else 0 (for the appropriate indices)
    
    # Actually, the tensor for matrix multiplication 3x2x2:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to C[i,k]. So T[i,j,k] = 1 always (for valid i,j,k), but the "output"
    # index is (i,k) not (i,k) as a separate dimension.
    
    # Wait, I need to be more careful. The tensor for n x m x p matrix multiplication:
    # T[a, b, c] where a indexes A (n*m entries), b indexes B (m*p entries), c indexes C (n*p entries)
    # T[a,b,c] = 1 if a corresponds to A[i,j], b to B[j,k], c to C[i,k] for some i,j,k
    # Otherwise T[a,b,c] = 0
    
    # For 3x2x2: n=3, m=2, p=2
    # A has 6 entries: A[0,0], A[0,1], A[1,0], A[1,1], A[2,0], A[2,1]
    # B has 4 entries: B[0,0], B[0,1], B[1,0], B[1,1]
    # C has 6 entries: C[0,0], C[0,1], C[1,0], C[1,1], C[2,0], C[2,1]
    
    # T[A[i,j], B[j,k], C[i,k]] = 1
    
    # Trivial rank: 3*2 = 6 (one per (i,k) pair, summing over j)
    # Wait no, trivial is n*m*p = 12? No, trivial is sum over j, so for each (i,k) we need
    # m terms, total n*m*p = 3*2*2 = 12.
    
    # But Strassen for the 2x2 part (i in {0,1}) gives rank 7 for the 2x2x2 sub-tensor.
    # Plus 4 more for i=2 (since 1x2x2 has trivial rank 2*1*... wait)
    
    # For i=2: C[2,k] = sum_j A[2,j] * B[j,k], k in {0,1}
    # This is a 1x2x2 tensor with trivial rank 2*2 = 4? No, it's 1*2*2 = 4 terms.
    # Actually for 1xm x mxp, trivial rank is m*p. So 1x2x2 has trivial rank 4.
    
    # So total for 3x2x2: 7 (Strassen for rows 0,1) + 4 (trivial for row 2) = 11.
    
    # Can we do better? The lower bound for 3x2x2 is... let me think.
    # The trivial rank is 12. Strassen+trivial gives 11.
    
    # Let me try to find if rank 10 is possible by sharing some terms.
    # In the Strassen decomposition for rows 0,1:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For row 2, we need:
    # C20 = A20*B00 + A21*B10
    # C21 = A20*B01 + A21*B11
    
    # Can we express these using some of the M's plus a few extra terms?
    # A20*B00 is not directly in the M's. A21*B10 is not either.
    
    # It seems hard to share. Let me just go with rank 44 and verify it works.
    
    # Build rank-44 decomposition
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x 44
    V = [[0]*rank for _ in range(m*p)]  # 16 x 44
    W = [[0]*rank for _ in range(p*n)]  # 12 x 44
    
    col = 0
    for b_r in range(2):  # row block: 0 or 1 (each covers 2 rows)
        for b_c in range(2):  # col block: 0 or 1 (each covers 2 cols)
            # Strassen for the 2x2x2 part (rows b_r*2, b_r*2+1; cols b_c*2, b_c*2+1)
            r0 = b_r * 2
            r1 = b_r * 2 + 1
            c0 = b_c * 2
            c1 = b_c * 2 + 1
            
            # M1 = (A[r0,c0]+A[r1,c1])(B[c0,r0]+B[c1,r1])
            U[r0*m + c0][col] = 1
            U[r1*m + c1][col] = 1
            V[c0*p + r0][col] = 1
            V[c1*p + r1][col] = 1
            W[r0*n + c0][col] = 1
            W[r1*n + c1][col] = 1
            col += 1
            
            # M2 = (A[r1,c0]+A[r1,c1])B[c0,r0]
            U[r1*m + c0][col] = 1
            U[r1*m + c1][col] = 1
            V[c0*p + r0][col] = 1
            W[r0*n + c0][col] = 1
            W[r1*n + c0][col] = -1
            col += 1
            
            # M3 = A[r0,c0](B[c1,r0]-B[c0,r0])
            U[r0*m + c0][col] = 1
            V[c1*p + r0][col] = 1
            V[c0*p + r0][col] = -1
            W[r0*n + c1][col] = 1
            W[r1*n + c1][col] = 1
            col += 1
            
            # M4 = A[r1,c1](B[c0,r1]-B[c1,r1])
            U[r1*m + c1][col] = 1
            V[c0*p + r1][col] = 1
            V[c1*p + r1][col] = -1
            W[r1*n + c0][col] = 1
            W[r0*n + c0][col] = 1
            col += 1
            
            # M5 = (A[r0,c0]+A[r0,c1])B[c1,r1]
            U[r0*m + c0][col] = 1
            U[r0*m + c1][col] = 1
            V[c1*p + r1][col] = 1
            W[r0*n + c0][col] = -1
            W[r0*n + c1][col] = 1
            col += 1
            
            # M6 = (A[r1,c0]-A[r0,c0])(B[c0,r0]+B[c0,r1])
            U[r0*m + c0][col] = -1
            U[r1*m + c0][col] = 1
            V[c0*p + r0][col] = 1
            V[c0*p + r1][col] = 1
            W[r1*n + c1][col] = 1
            col += 1
            
            # M7 = (A[r0,c1]-A[r1,c1])(B[c1,r0]+B[c1,r1])
            U[r0*m + c1][col] = 1
            U[r1*m + c1][col] = -1
            V[c1*p + r0][col] = 1
            V[c1*p + r1][col] = 1
            W[r0*n + c0][col] = 1
            col += 1
            
            # Trivial for row 2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (c0 + j_local)][col] = 1
                    V[(c0 + j_local)*p + (c0 + k_local)][col] = 1
                    W[(c0 + k_local)*n + 2][col] = 1
                    col += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = T[i*m+j, j*p+k, k*n+i]
                max_err = max(max_err, abs(actual - 1))
    
    for idx_i in range(n*m):
        for idx_j in range(m*p):
            for idx_k in range(p*n):
                i = idx_i // m
                j = idx_i % m
                j2 = idx_j // p
                k = idx_j % p
                k2 = idx_k // n
                i2 = idx_k % n
                if j == j2 and i == i2 and k == k2:
                    continue
                max_err = max(max_err, abs(T[idx_i, idx_j, idx_k]))
    
    if max_err > 1e-10:
        # Fallback to trivial rank 48
        rank = 48
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
