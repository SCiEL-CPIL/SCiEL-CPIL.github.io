"""Matrix multiplication 4x4 by 4x6: search for low-rank decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 6
    schoolbook_rank = n * m * p  # 96
    
    def make_schoolbook():
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
    
    def strassen_decomp():
        """Strassen-based decomposition: 12 products of (2x2)*(2x2), each rank 7. Total rank 84."""
        
        def strassen_2x2_2x2():
            """Return (U,V,W) for 2x2*2x2 with rank 7."""
            R = 7
            U = [[0]*R for _ in range(4)]  # 2*2
            V = [[0]*R for _ in range(4)]  # 2*2
            W = [[0]*R for _ in range(4)]  # 2*2, W[k*2+i]
            
            # M1: (A11+A22)(B11+B22)
            U[0][0] = 1  # A11
            U[3][0] = 1  # A22
            V[0][0] = 1  # B11
            V[3][0] = 1  # B22
            W[0][0] = 1  # C11
            W[1][0] = 1  # C12
            W[2][0] = 1  # C21
            W[3][0] = 1  # C22
            
            # M2: (A21+A22)B11
            U[2][1] = 1  # A21
            U[3][1] = 1  # A22
            V[0][1] = 1  # B11
            W[2][1] = 1  # C21
            W[3][1] = 1  # C22
            
            # M3: A11(B12-B22)
            U[0][2] = 1  # A11
            V[1][2] = 1  # B12
            V[3][2] = -1  # B22
            W[1][2] = 1  # C12
            W[3][2] = 1  # C22
            
            # M4: A22(B21-B11)
            U[3][3] = 1  # A22
            V[2][3] = 1  # B21
            V[0][3] = -1  # B11
            W[2][3] = 1  # C21
            W[3][3] = 1  # C22
            
            # M5: (A11+A12)B22
            U[0][4] = 1  # A11
            U[1][4] = 1  # A12
            V[3][4] = 1  # B22
            W[1][4] = 1  # C12
            W[3][4] = 1  # C22
            
            # M6: (A21-A11)(B11+B12)
            U[2][5] = 1  # A21
            U[0][5] = -1  # A11
            V[0][5] = 1  # B11
            V[1][5] = 1  # B12
            W[2][5] = 1  # C21
            W[3][5] = 1  # C22
            
            # M7: (A12-A22)(B21+B22)
            U[1][6] = 1  # A12
            U[3][6] = -1  # A22
            V[2][6] = 1  # B21
            V[3][6] = 1  # B22
            W[0][6] = 1  # C11
            W[1][6] = 1  # C12
            W[2][6] = 1  # C21
            W[3][6] = 1  # C22
            
            return U, V, W
        
        U_s, V_s, W_s = strassen_2x2_2x2()
        
        # Build the full 4x4*4x6 decomposition
        # A: 2x2 blocks of 2x2, indexed by (ib, jb) for ib,jb in {0,1}
        # B: 2x3 blocks of 2x2, indexed by (jb, kb) for jb in {0,1}, kb in {0,1,2}
        # C: 2x3 blocks of 2x2, indexed by (ib, kb) for ib in {0,1}, kb in {0,1,2}
        
        # For each (ib, kb), C[ib,kb] = A[ib,0]*B[0,kb] + A[ib,1]*B[1,kb]
        # Each product is 2x2*2x2, done with Strassen (rank 7).
        # Total: 2*3*2 = 12 Strassen products, each rank 7, total rank 84.
        
        total_rank = 12 * 7
        U_full = [[0]*total_rank for _ in range(16)]  # 4*4
        V_full = [[0]*total_rank for _ in range(24)]  # 4*6
        W_full = [[0]*total_rank for _ in range(24)]  # 6*4
        
        col_offset = 0
        for ib in range(2):
            for kb in range(3):
                for jb in range(2):
                    # Product: A[ib,jb] * B[jb,kb] contributes to C[ib,kb]
                    for r in range(7):
                        # U: A indices
                        for ai in range(2):
                            for aj in range(2):
                                a_idx = ai*2 + aj
                                if U_s[a_idx][r] != 0:
                                    global_i = 2*ib + ai
                                    global_j = 2*jb + aj
                                    U_full[global_i*4 + global_j][col_offset + r] += U_s[a_idx][r]
                        
                        # V: B indices
                        for bj in range(2):
                            for bk in range(2):
                                b_idx = bj*2 + bk
                                if V_s[b_idx][r] != 0:
                                    global_j = 2*jb + bj
                                    global_k = 2*kb + bk
                                    V_full[global_j*6 + global_k][col_offset + r] += V_s[b_idx][r]
                        
                        # W: C indices
                        for ck in range(2):
                            for ci in range(2):
                                c_idx = ck*2 + ci
                                if W_s[c_idx][r] != 0:
                                    global_k = 2*kb + ck
                                    global_i = 2*ib + ci
                                    W_full[global_k*4 + global_i][col_offset + r] += W_s[c_idx][r]
                    
                    col_offset += 7
        
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify the decomposition is correct
    def verify(U, V, W):
        """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k."""
        n, m, p = 4, 4, 6
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    expected = 1
                    if abs(s - expected) > 1e-9:
                        return False, (i,j,k,s,expected)
        return True, None
    
    # Try Strassen (rank 84)
    result = strassen_decomp()
    valid, err = verify(result["U"], result["V"], result["W"])
    
    if valid:
        return result
    
    # If Strassen fails, fall back to schoolbook
    return make_schoolbook()
