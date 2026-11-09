"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if j is the shared index, else 0.
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 if the indices form a valid multiplication term.
    # More precisely, the multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
    # and the decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (1 if valid)
    
    # Let's check: for each (i,j,k), the sum should be 1 if this is a valid
    # term in matrix multiplication, which is always 1 for all (i,j,k) in range.
    # Actually no - the tensor for matrix multiplication M_{n,m,p} has
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # And the decomposition must satisfy:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    
    # Construct rank-18 decomposition for 2x3 x 3x4
    # Split into two 2x3 x 3x2 blocks
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, r):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val
    
    # For each block (cols 0,1 and cols 2,3), use rank-9 for M_{2,3,2}
    # M_{2,3,2}: A is 2x3, B is 3x2, C is 2x2
    # We use the isomorphism M_{2,3,2} ~ M_{2,2,3}
    
    # Rank-9 decomposition for M_{2,2,3}:
    # A: 2x2, B: 2x3, C: 2x3
    # 7 Strassen terms for first 2 cols + 2 terms for 3rd col
    
    # For M_{2,3,2}, we map:
    # A_{i,j} (i=0,1; j=0,1,2) corresponds to rows of A in M_{2,2,3}
    # B_{j,k} (j=0,1,2; k=0,1) corresponds to cols of B in M_{2,2,3}
    
    # Let's use a direct construction for M_{2,3,2} with rank 9.
    # A = [[a0, a1, a2], [a3, a4, a5]]
    # B = [[b0, b1], [b2, b3], [b4, b5]]
    # C = A*B
    
    # 9 products:
    # p0 = (a0+a1+a2)*(b0+b2+b4)
    # p1 = (a0+a1)*(b0+b2)  
    # p2 = (a0+a2)*(b0+b4)
    # p3 = (a1+a2)*(b2+b4)
    # p4 = a0*(b2+b4-b0)
    # p5 = a1*(b0+b4-b2)
    # p6 = a2*(b0+b2-b4)
    # p7 = (a3+a4+a5)*(b1+b3+b5)
    # p8 = (a3+a4)*(b1+b3)
    
    # C[0,0] = p0 - p1 - p2 - p3 + p4 + p5 + p6
    # C[0,1] = ?
    # C[1,0] = ?
    # C[1,1] = ?
    
    # This doesn't seem right. Let me use a verified construction.
    
    # Actually, let me use the following known rank-9 for M_{2,2,3} and map it.
    # M_{2,2,3}: A(2x2), B(2x3), C(2x3)
    # Strassen 7 for A(2x2)*B[:,0:2] (2x2x2)
    # Plus 2 more for B[:,2]
    
    # For the 2x2x2 part (cols 0,1 of B in M_{2,2,3}):
    # s0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,0])
    # s1 = (A[1,0]+A[1,1])*B[0,0]
    # s2 = A[0,0]*(B[1,0]-B[0,0])
    # s3 = A[1,1]*(B[0,1]-B[1,0])
    # s4 = (A[0,0]+A[0,1])*B[1,0]
    # s5 = (A[1,0]+A[1,1])*B[0,1]
    # s6 = (A[0,1]-A[1,0])*(B[0,0]+B[1,0])
    
    # C[0,0] = s0 - s1 - s3 + s6
    # C[0,1] = s2 + s3
    # C[1,0] = s0 - s2 - s4 + s5
    # C[1,1] = s1 - s0 + s4 + s5
    
    # For col 2 of B in M_{2,2,3}:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # This needs 4 terms in schoolbook, but we claim rank 9 total.
    # So we need 2 more terms. This suggests a different construction.
    
    # Given the difficulty, let me just return schoolbook rank 24.
    # This is safe and correct.
    
    if time.time() > deadline - 5:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Return schoolbook decomposition (rank 24)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Verify
    if not _check_decomposition(U_sb, V_sb, W_sb, n, m, p):
        # Should never happen for schoolbook
        pass
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
