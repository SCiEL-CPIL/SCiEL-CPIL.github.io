"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Decomposition strategy:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Split into:
    #   Core: j=0..3, k=0..3 -> 4x4x4 (Strassen, rank 49)
    #   Right strip: j=0..3, k=4 -> 4x4x1 (schoolbook, rank 16)
    #   Bottom strip: j=4, k=0..3 -> 4x1x4 (schoolbook, rank 16)
    #   Corner: j=4, k=4 -> 4x1x1 (schoolbook, rank 4)
    # Total rank: 49 + 16 + 16 + 4 = 85
    
    rank = 85
    U = np.zeros((n * m, rank))  # (20, 85)
    V = np.zeros((m * p, rank))  # (25, 85)
    W = np.zeros((p * n, rank))  # (20, 85)
    
    col = 0
    
    # Part 1: 4x4x4 Strassen decomposition (rank 49)
    # Using recursive Strassen: 7 products of 2x2x2, each with rank 7
    # 2x2x2 Strassen decomposition
    def strassen_2x2x2():
        """Returns U, V, W for 2x2x2 Strassen (rank 7)."""
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        u[0,0]=1; u[3,0]=1; v[0,0]=1; v[3,0]=1; w[0,0]=1; w[3,0]=1
        # M2 = (A10+A11)B00
        u[2,1]=1; u[3,1]=1; v[0,1]=1; w[2,1]=1; w[3,1]=-1
        # M3 = A00(B01-B11)
        u[0,2]=1; v[1,2]=1; v[3,2]=-1; w[1,2]=1; w[3,2]=1
        # M4 = A11(B10-B00)
        u[3,3]=1; v[2,3]=1; v[0,3]=-1; w[0,3]=1; w[2,3]=1
        # M5 = (A00+A01)B11
        u[0,4]=1; u[1,4]=1; v[3,4]=1; w[0,4]=-1; w[1,4]=1
        # M6 = (A10-A00)(B00+B01)
        u[0,5]=-1; u[2,5]=1; v[0,5]=1; v[1,5]=1; w[3,5]=1
        # M7 = (A01-A11)(B10+B11)
        u[1,6]=1; u[3,6]=-1; v[2,6]=1; v[3,6]=1; w[0,6]=1
        return u, v, w
    
    su, sv, sw = strassen_2x2x2()
    
    # Strassen outer structure for 4x4x4
    # A blocks: A00(i=0,1;k=0,1), A01(i=0,1;k=2,3), A10(i=2,3;k=0,1), A11(i=2,3;k=2,3)
    # B blocks: B00(k=0,1;j=0,1), B01(k=0,1;j=2,3), B10(k=2,3;j=0,1), B11(k=2,3;j=2,3)
    # C blocks: C00(i=0,1;j=0,1), C01(i=0,1;j=2,3), C10(i=2,3;j=0,1), C11(i=2,3;j=2,3)
    
    # Each Strassen product M_t = X_t * Y_t where X_t, Y_t are 2x2 matrices
    # formed from blocks. For each (local_i, local_k, local_j) in 2x2x2:
    #   U[A_global(block_a, local_i, local_k)] += coeff_a * su[local_i*2+local_k, t2]
    #   V[B_global(block_b, local_k, local_j)] += coeff_b * sv[local_k*2+local_j, t2]
    #   W[C_global(block_c, local_i, local_j)] += coeff_c * sw[local_i*2+local_j, t2]
    
    # Define the 7 Strassen products with their block compositions
    # Each product: list of (block_id, sign) for A, B, and C contributions
    strassen_products = [
        # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        # M2 = (A10+A11)B00 -> C10(+1), C11(-1)
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
        # M3 = A00(B01-B11) -> C01(+1), C11(+1)
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        # M4 = A11(B10-B00) -> C00(+1), C10(+1)
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        # M5 = (A00+A01)B11 -> C00(-1), C01(+1)
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        # M6 = (A10-A00)(B00+B01) -> C11(+1)
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        # M7 = (A01-A11)(B10+B11) -> C00(+1)
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
    ]
    
    # Helper to get global index from block and local indices
    def a_global(block, li, lk):
        """Get global (i, k) index for A from block and local indices."""
        if block == 0:  # A00
            return li, lk
        elif block == 1:  # A01
            return li, lk + 2
        elif block == 2:  # A10
            return li + 2, lk
        elif block == 3:  # A11
            return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        """Get global (k, j) index for B from block and local indices."""
        if block == 0:  # B00
            return lk, lj
        elif block == 1:  # B01
            return lk, lj + 2
        elif block == 2:  # B10
            return lk + 2, lj
        elif block == 3:  # B11
            return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        """Get global (i, j) index for C from block and local indices."""
        if block == 0:  # C00
            return li, lj
        elif block == 1:  # C01
            return li, lj + 2
        elif block == 2:  # C10
            return li + 2, lj
        elif block == 3:  # C11
            return li + 2, lj + 2
    
    # For each of the 7 Strassen products, add 7 rank-1 terms (from 2x2x2 Strassen)
    for t in range(7):
        prod = strassen_products[t]
        for t2 in range(7):  # 7 terms from 2x2x2 Strassen
            c_idx = col + t2
            # For each local index combination in 2x2x2
            for li in range(2):
                for lk in range(2):
                    for lj in range(2):
                        # U contribution: from A blocks
                        su_val = su[li * 2 + lk, t2]
                        if su_val != 0:
                            for block_a, sign_a in prod['A']:
                                gi, gk = a_global(block_a, li, lk)
                                U[gi * m + gk, c_idx] += sign_a * su_val
                        
                        # V contribution: from B blocks
                        sv_val = sv[lk * 2 + lj, t2]
                        if sv_val != 0:
                            for block_b, sign_b in prod['B']:
                                gk, gj = b_global(block_b, lk, lj)
                                V[gk * p + gj, c_idx] += sign_b * sv_val
                        
                        # W contribution: from C blocks
                        sw_val = sw[li * 2 + lj, t2]
                        if sw_val != 0:
                            for block_c, sign_c in prod['C']:
                                gi, gj = c_global(block_c, li, lj)
                                W[gj * n + gi, c_idx] += sign_c * sw_val
            col += 1
    
    # col should now be 49
    
    # Part 2: 4x4x1 right strip (rank 16)
    # C[i,4] += A[i,j]*B[j,4] for i=0..3, j=0..3
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
    
    # Part 3: 4x1x4 bottom strip (rank 16)
    # C[i,k] += A[i,4]*B[4,k] for i=0..3, k=0..3
    for i in range(4):
        for k in range(4):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Part 4: 4x1x1 corner (rank 4)
    # C[i,4] += A[i,4]*B[4,4] for i=0..3
    for i in range(4):
        U[i * m + 4, col] = 1
        V[4 * p + 4, col] = 1
        W[4 * n + i, col] = 1
        col += 1
    
    # Verify col == rank
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Quick verification
    def verify(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
