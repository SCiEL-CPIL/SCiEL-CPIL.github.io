import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def verify_block(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Rank 55: Strassen for 2x2x2 blocks + schoolbook for remainder
    # A = [A1 | A2], A1 is 2x2 (cols 0,1), A2 is 2x1 (col 2)
    # B = [B1; B2], B1 is 2x10 (rows 0,1), B2 is 1x10 (row 2)
    # AB = A1*B1 + A2*B2
    # A1*B1: 5 blocks of 2x2x2, each rank 7 (Strassen) -> rank 35
    # A2*B2: schoolbook 2x1x10 -> rank 20
    # Total: rank 55
    def make_rank55():
        rank = 55
        U = [[0] * rank for _ in range(n * m)]  # 6 x 55
        V = [[0] * rank for _ in range(m * p)]  # 30 x 55
        W = [[0] * rank for _ in range(p * n)]  # 20 x 55
        
        col = 0
        # Strassen for each 2x2x2 block (5 blocks, each using 7 terms)
        for block_idx in range(5):
            k1 = block_idx * 2      # first output column of this block
            k2 = block_idx * 2 + 1  # second output column of this block
            
            # A1 = [[a00, a01], [a10, a11]] (rows 0,1; cols 0,1 of A)
            # B1_block = [[b0k1, b0k2], [b1k1, b1k2]] (rows 0,1 of B; cols k1,k2)
            
            # Strassen:
            # m1 = (a00+a10)(b0k1+b1k1)  -- but we need to handle the 2x2x2 case
            # For 2x2x2: A=[[a,b],[c,d]], B=[[e,f],[g,h]]
            # C = [[ae+cg, af+dh], [ce+dg, cf+eh]]
            # m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
            # C00=m1+m4+m5+m6, C01=m3+m5, C10=m2+m4, C11=m1-m2+m3+m7
            
            # In our case:
            # a = a00, b = a01, c = a10, d = a11
            # e = b0k1, f = b0k2, g = b1k1, h = b1k2
            
            # m1 = (a00+a11)(b0k1+b1k2)
            # m2 = (a10+a11)*b0k1
            # m3 = a00*(b0k2-b1k2)
            # m4 = a11*(b1k1-b0k1)
            # m5 = (a00+a01)*b1k2
            # m6 = (a10-a00)*(b0k1+b0k2)
            # m7 = (a01-a11)*(b1k1+b1k2)
            
            # C00 (row 0, col k1 of output) = m1+m4+m5+m6
            # C01 (row 0, col k2 of output) = m3+m5
            # C10 (row 1, col k1 of output) = m2+m4
            # C11 (row 1, col k2 of output) = m1-m2+m3+m7
            
            # For each Strassen term, we need U, V, W entries
            # U is for A factors, V is for B factors, W is for output factors
            
            # m1: U has (a00+a11), V has (b0k1+b1k2), W contributes to C00 and C11
            # But wait, in tensor decomposition, each rank-1 term is U_r * V_r * W_r
            # where U_r is a vector in A-space, V_r in B-space, W_r in output-space
            
            # For m1 = (a00+a11)(b0k1+b1k2):
            # U_r: a00 and a11 get +1 (indices 0 and 3 in 2x3 A, i.e., row0*col0 and row1*col1)
            # V_r: b0k1 and b1k2 get +1 (indices 0*k1 and 1*k2 in 3x10 B)
            # W_r: contributes to output[0,k1] and output[1,k2]
            # But W_r is a single vector, so it can only have one non-zero pattern per term.
            
            # I think I'm confusing myself. Let me reconsider the tensor decomposition.
            
            # The tensor T has T[i,j,k] = 1 if output[i,k] = sum_j A[i,j]*B[j,k]
            # A rank-r decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
            # where U_r[i,j], V_r[j,k], W_r[k,i]
            
            # For Strassen on 2x2x2:
            # T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1}
            # m1: U has U[0,0]=1, U[1,1]=1; V has V[0,0]=1, V[1,1]=1; W has W[0,0]=1, W[1,1]=1
            # But that's not right either, because W is indexed by [k,i], not [i,k].
            
            # Let me be very explicit. The output C[i,k] = sum_j A[i,j] B[j,k]
            # The tensor T[i,j,k] = 1 for all i,j,k (in the 2x2x2 case)
            # A rank-1 term contributes U[i,j] * V[j,k] * W[k,i] to T[i,j,k]
            
            # For m1 = (a0+a1)(b00+b11) [using 0-indexed: a00,a01,a10,a11 for A; b00,b01,b10,b11 for B]:
            # Actually, let me use the standard Strassen for 2x2:
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # m1=(a+d)(e+h): U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,1]=1; W[0,0]=1,W[1,1]=1
            #   Check: T[0,0,0] += 1*1*1=1, T[0,1,1] += 1*1*1=1, T[1,0,0] += 1*1*1=1, T[1,1,1] += 1*1*1=1
            #   But we need T[i,j,k]=1 for all i,j,k. This only covers 4 of the 8 entries.
            
            # I think the issue is that a single rank-1 term can't cover all 8 entries.
            # The 7 Strassen terms together cover all 8 entries with the right coefficients.
            
            # Let me look up the exact Strassen tensor decomposition.
            # The 2x2x2 tensor has rank 7. The decomposition is:
            # T = sum_{r=1}^{7} U_r ⊗ V_r ⊗ W_r
            # where:
            # r=1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]  -- (a+d)(e+h)
            # r=2: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,0],[0,1]]  -- (c+d)e
            # r=3: U=[[1,0],[0,0]], V=[[0,1],[0,-1]], W=[[0,1],[0,0]]  -- a(f-h)
            # r=4: U=[[0,0],[0,1]], V=[[0,-1],[1,0]], W=[[0,0],[1,0]]  -- d(g-e)
            # r=5: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[0,1],[0,1]]  -- (a+b)h
            # r=6: U=[[0,-1],[1,0]], V=[[1,1],[0,0]], W=[[1,0],[0,0]]  -- (c-a)(e+f)
            # r=7: U=[[0,1],[0,-1]], V=[[0,0],[1,1]], W=[[0,0],[1,1]]  -- (b-d)(g+h)
            
            # Let me verify r=1:
            # T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
            # T[0,0,1] += U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
            # T[0,1,0] += U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0
            # T[0,1,1] += U[0,1]*V[1,1]*W[1,0] = 0*1*0 = 0
            # T[1,0,0] += U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
            # T[1,0,1] += U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
            # T[1,1,0] += U[1,1]*V[1,0]*W[0,1] = 1*0*0 = 0
            # T[1,1,1] += U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1
            # So r=1 contributes to T[0,0,0] and T[1,1,1].
            
            # This is getting very complex. Let me just hardcode the known rank-55 decomposition.
            
            # Actually, I realize I should just implement this carefully.
            # For each 2x2x2 block (using A cols 0,1 and B rows 0,1, output cols k1,k2):
            # The 7 Strassen terms have specific U, V, W patterns.
            
            # Let me define the 7 Strassen patterns for 2x2x2:
            # Each pattern is (U_pattern, V_pattern, W_pattern)
            # U is 2x2, V is 2x2, W is 2x2 (W[k,i] convention)
            
            strassen_patterns = [
                # (U, V, W) where U[i,j], V[j,k], W[k,i]
                ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),  # m1=(a+d)(e+h)
                ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[0,1]]),  # m2=(c+d)e
                ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),  # m3=a(f-h)
                ([[0,0],[0,1]], [[0,-1],[1,0]], [[0,0],[1,0]]),  # m4=d(g-e)
                ([[1,1],[0,0]], [[0,0],[0,1]], [[0,1],[0,1]]),  # m5=(a+b)h
                ([[0,-1],[1,0]], [[1,1],[0,0]], [[1,0],[0,0]]),  # m6=(c-a)(e+f)
                ([[0,1],[0,-1]], [[0,0],[1,1]], [[0,0],[1,1]]),  # m7=(b-d)(g+h)
            ]
            
            for pat_idx in range(7):
                U_pat, V_pat, W_pat = strassen_patterns[pat_idx]
                
                # Map local indices to global indices
                # A: local (li, lj) -> global (li, lj) for li in {0,1}, lj in {0,1}
                # B: local (lj, lk) -> global (lj, k1+lk) for lj in {0,1}, lk in {0,1}
                # W: local (lk, li) -> global (k1+lk, li) for lk in {0,1}, li in {0,1}
                
                for li in range(2):
                    for lj in range(2):
                        if U_pat[li][lj] != 0:
                            U[li * m + lj][col] += U_pat[li][lj]
                
                for lj in range(2):
                    for lk in range(2):
                        if V_pat[lj][lk] != 0:
                            V[lj * p + (k1 + lk)][col] += V_pat[lj][lk]
                
                for lk in range(2):
                    for li in range(2):
                        if W_pat[lk][li] != 0:
                            W[(k1 + lk) * n + li][col] += W_pat[lk][li]
                
                col += 1
        
        # Schoolbook for A2*B2: A col 2 (2x1) times B row 2 (1x10)
        # 2*1*10 = 20 terms
        for i in range(n):  # i in {0,1}
            for k in range(p):  # k in {0,...,9}
                U[i * m + 2][col] = 1  # A[i,2]
                V[2 * p + k][col] = 1  # B[2,k]
                W[k * n + i][col] = 1  # output[i,k]
                col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        
        # Verify
        if verify_block(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        return None

    # Try rank 55 first
    result = make_rank55()
    if result:
        return result
    
    # Fallback: Schoolbook (Rank 60)
    def make_schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    return make_schoolbook()
