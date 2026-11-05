import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def check_decomposition(U, V, W, n=2, m=3, p=4):
    """
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    # Construct T from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The tensor for matrix multiplication is defined such that T[i, j, k] = 1.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    u_val = U[i*m+j][r] if not isinstance(U, np.ndarray) else U[i*m+j, r]
                    v_val = V[j*p+k][r] if not isinstance(V, np.ndarray) else V[j*p+k, r]
                    w_val = W[k*n+i][r] if not isinstance(W, np.ndarray) else W[k*n+i, r]
                    s += u_val * v_val * w_val
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    max_time = 135.0  # Safety margin before 150s kill

    def build_rank_22():
        """
        Construct a rank-22 decomposition using Strassen for 2x2 blocks
        and outer product for the remaining 2x1x4 and 1x1x4 parts.
        """
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Indices for A (n x m = 2 x 3):
        # A[i, j] -> U[i*m + j]
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        
        # Indices for B (m x p = 3 x 4):
        # B[j, k] -> V[j*p + k]
        # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
        # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
        # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
        
        # Indices for C (p x n = 4 x 2) via W[k*n + i]:
        # C[k, i] -> W[k*n + i]
        # C[0,0]=0, C[0,1]=1
        # C[1,0]=2, C[1,1]=3
        # C[2,0]=4, C[2,1]=5
        # C[3,0]=6, C[3,1]=7
        
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # Uses Strassen (7 terms).
        # A submatrix:
        #   A00 (0), A01 (1)
        #   A10 (3), A11 (4)
        # B submatrix:
        #   B00 (0), B01 (1)
        #   B10 (4), B11 (5)
        # C submatrix:
        #   C00 (0), C01 (1)
        #   C10 (2), C11 (3)
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
        # Standard Strassen for C = A*B:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's map these to columns 0-6.
        
        # Col 0: M1
        # U: A00(0), A11(4) -> 1, 1
        # V: B00(0), B11(5) -> 1, 1
        # W: C00(0), C01(1), C10(2), C11(3) -> 1, 1, 1, 1? 
        # Wait, M1 contributes to C00, C01, C10, C11 with coefficients?
        # C00 = M1 + ... => coeff 1
        # C01 = ... + M5? No C01 = M3+M5. M1 is not in C01.
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 is in C00 (coeff 1) and C11 (coeff 1).
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        
        # Col 1: M2
        # U: A10(3), A11(4) -> 1, 1
        # V: B00(0) -> 1
        # W: C10(2) (coeff 1), C11(3) (coeff -1)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(2, 1), (3, -1)], 1)
        
        # Col 2: M3
        # U: A00(0) -> 1
        # V: B01(1), B11(5) -> 1, -1
        # W: C01(1) (coeff 1), C11(3) (coeff 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(1, 1), (3, 1)], 2)
        
        # Col 3: M4
        # U: A11(4) -> 1
        # V: B10(4), B00(0) -> 1, -1
        # W: C00(0) (coeff 1), C10(2) (coeff 1)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (2, 1)], 3)
        
        # Col 4: M5
        # U: A00(0), A10(3) -> 1, 1
        # V: B11(5) -> 1
        # W: C00(0) (coeff -1), C01(1) (coeff 1)
        add_term([(0, 1), (3, 1)], [(5, 1)], [(0, -1), (1, 1)], 4)
        
        # Col 5: M6
        # U: A10(3), A00(0) -> 1, -1
        # V: B00(0), B10(4) -> 1, 1
        # W: C11(3) (coeff 1)
        add_term([(3, 1), (0, -1)], [(0, 1), (4, 1)], [(3, 1)], 5)
        
        # Col 6: M7
        # U: A01(1), A11(4) -> 1, -1
        # V: B01(1), B11(5) -> 1, 1
        # W: C00(0) (coeff 1)
        add_term([(1, 1), (4, -1)], [(1, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        # A is same as Block 1.
        # B submatrix:
        #   B02 (2), B03 (3)
        #   B12 (6), B13 (7)
        # C submatrix:
        #   C02 (4), C03 (5)
        #   C12 (6), C13 (7)
        
        # Col 7: M1'
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        
        # Col 8: M2'
        add_term([(3, 1), (4, 1)], [(2, 1)], [(6, 1), (7, -1)], 8)
        
        # Col 9: M3'
        add_term([(0, 1)], [(3, 1), (7, -1)], [(5, 1), (7, 1)], 9)
        
        # Col 10: M4'
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (6, 1)], 10)
        
        # Col 11: M5'
        add_term([(0, 1), (3, 1)], [(7, 1)], [(4, -1), (5, 1)], 11)
        
        # Col 12: M6'
        add_term([(3, 1), (0, -1)], [(2, 1), (6, 1)], [(7, 1)], 12)
        
        # Col 13: M7'
        add_term([(1, 1), (4, -1)], [(3, 1), (7, 1)], [(4, 1)], 13)

        # Block 3: A[0:2, 2:3] * B[0:3, 0:4] -> C[0:4, 0:2]
        # This is the remaining part involving A[:, 2] and B[2, :].
        # A[0,2]=2, A[1,2]=5
        # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
        # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3, C[2,0]=4, C[2,1]=5, C[3,0]=6, C[3,1]=7
        
        # We need to compute:
        # C[0,0] += A[0,2]B[2,0] => 2*8
        # C[0,1] += A[0,2]B[2,1] => 2*9
        # C[0,2] += A[0,2]B[2,2] => 2*10
        # C[0,3] += A[0,2]B[2,3] => 2*11
        # C[1,0] += A[1,2]B[2,0] => 5*8
        # C[1,1] += A[1,2]B[2,1] => 5*9
        # C[1,2] += A[1,2]B[2,2] => 5*10
        # C[1,3] += A[1,2]B[2,3] => 5*11
        
        # We can do this with 4 terms if we group by B column?
        # Or 2 terms if we group by A row?
        # Since it's a rank-1 slice of the tensor (fixed j=2), we can use:
        # Term for each k? No, that's 8 terms.
        # Term for each i? No.
        
        # Let's just use outer product for the 2x1x4 and 1x2x4 parts?
        # Actually, the "outer product" inspiration used 8 terms here.
        # Can we do better?
        # A[0,2]B[2,0] -> U[2]=1, V[8]=1, W[0]=1
        # A[0,2]B[2,1] -> U[2]=1, V[9]=1, W[1]=1
        # ...
        # This is 8 terms.
        
        # Is there a Strassen-like trick for 2x1x4?
        # No, 2x1x4 tensor is just a vector of 4 entries for each of 2 rows.
        # It's essentially a 2x4 matrix multiplication by a 1x1? No.
        # It's A (2x1) * B (1x4).
        # This is just element-wise multiplication of a 2x1 vector and 1x4 vector?
        # No, it's C[i,k] = A[i,2] * B[2,k].
        # This is an outer product of vector A[:,2] and vector B[2,:].
        # Rank of outer product of two vectors is 1?
        # No, the tensor T[i,k] = u_i v_k.
        # Decomposition: sum_r U[i,r] V[k,r] W[i,k,r]?
        # Here indices are i, k. j is fixed at 2.
        # T[i, 2, k] = A[i,2] B[2,k].
        # We need sum_r U[i*3+2, r] V[2*4+k, r] W[k*2+i, r] = A[i,2] B[2,k].
        
        # If we set:
        # r=0: U[2,0]=1, V[8,0]=1, W[0,0]=1 => contributes 1*1*1 to T[0,2,0]
        # We need to separate A[i,2] and B[2,k].
        
        # Let's use 4 terms, one for each k?
        # Term k:
        # U[2, col] = 1
        # V[8+k, col] = 1
        # W[k*2+0, col] = A[0,2]? No, A[0,2] is a value, not 1.
        # We can scale U or V or W.
        
        # Let's try 4 terms:
        # Col 14: k=0
        # U[2] = 1
        # V[8] = 1
        # W[0] = 1 (for i=0), W[2] = 1 (for i=1)
        # This gives T[0,2,0] = 1*1*1 = 1. But we need A[0,2]B[2,0].
        # Since A and B entries are 1 in the tensor definition?
        # Wait, the tensor T[i,j,k] is 1 for all i,j,k.
        # The decomposition must satisfy sum U V W = 1.
        # The values of A and B are not fixed, they are variables.
        # The tensor T is the structure constant.
        # T[i,j,k] = 1.
        # So we just need to match the structure.
        
        # So for the block A[:,2] * B[2,:], we need:
        # T[i, 2, k] = 1.
        # This is satisfied if we have terms that cover all (i,k) pairs.
        # Since it's a full 2x4 grid of 1s, we need rank at least 4?
        # Actually, the tensor T[i, 2, k] for i in 0..1, k in 0..3 is all 1s.
        # This is a 2x4 matrix of 1s.
        # The rank of a matrix of 1s is 1.
        # But this is a tensor slice.
        # T[i, 2, k] = sum_r U[i*3+2, r] V[2*4+k, r] W[k*2+i, r].
        
        # Can we do it with 1 term?
        # U[2, r] = 1
        # V[8+k, r] = 1
        # W[k*2+i, r] = 1
        # Then sum = 1*1*1 = 1.
        # Yes!
        # So we only need 1 term for the whole 2x4 block?
        # Let's check.
        # If we set U[2, r] = 1, V[8, r]=V[9, r]=V[10, r]=V[11, r]=1, W[0, r]=W[2, r]=W[4, r]=W[6, r]=1, W[1, r]=W[3, r]=W[5, r]=W[7, r]=1.
        # Then for any i, k:
        # U[i*3+2, r] is 1 if i=0 (idx 2) or i=1 (idx 5)?
        # Wait, U index is i*m + j. j=2.
        # i=0 -> U[2]. i=1 -> U[5].
        # So U[2, r]=1 and U[5, r]=1.
        # V index is j*p + k. j=2.
        # k=0 -> V[8]. k=1 -> V[9]. k=2 -> V[10]. k=3 -> V[11].
        # So V[8, r]=V[9, r]=V[10, r]=V[11, r]=1.
        # W index is k*n + i.
        # i=0 -> W[k*2]. i=1 -> W[k*2+1].
        # So W[0, r]=W[2, r]=W[4, r]=W[6, r]=1 (for i=0)
        # and W[1, r]=W[3, r]=W[5, r]=W[7, r]=1 (for i=1).
        
        # Then T[0, 2, 0] = U[2, r] V[8, r] W[0, r] = 1*1*1 = 1.
        # T[1, 2, 3] = U[5, r] V[11, r] W[7, r] = 1*1*1 = 1.
        
        # So we can do the entire A[:,2] * B[2,:] block with just 1 term!
        # This reduces the rank from 7+7+8 = 22 to 7+7+1 = 15?
        # Wait, did I miss something?
        # The tensor T[i,j,k] is 1 for ALL i,j,k.
        # My Strassen blocks only cover j=0,1.
        # The block j=2 is covered by this 1 term.
        # So total rank = 7 (Block 1) + 7 (Block 2) + 1 (Block 3) = 15.
        
        # Let's verify this logic.
        # Block 1: j=0,1. k=0,1. i=0,1.
        # Block 2: j=0,1. k=2,3. i=0,1.
        # Block 3: j=2. k=0,1,2,3. i=0,1.
        
        # Yes, this covers all i, j, k.
        # So rank 15 is possible?
        
        # Let's add the term for Block 3.
        add_term([(2, 1), (5, 1)], [(8, 1), (9, 1), (10, 1), (11, 1)], 
                 [(0, 1), (2, 1), (4, 1), (6, 1), (1, 1), (3, 1), (5, 1), (7, 1)], 14)
        
        # Total rank used: 15.
        # We declared rank=22, but we only used 15 columns.
        # We should trim the rank to 15.
        # But the function returns rank=22.
        # We can just return the matrices with 15 columns.
        
        # Let's re-verify the "1 term" claim.
        # T[i, 2, k] = U[i*3+2, 14] * V[2*4+k, 14] * W[k*2+i, 14]
        # U[2, 14] = 1, U[5, 14] = 1.
        # V[8, 14] = 1, V[9, 14] = 1, V[10, 14] = 1, V[11, 14] = 1.
        # W[0, 14] = 1, W[1, 14] = 1, ..., W[7, 14] = 1.
        
        # For i=0, k=0: U[2,14]*V[8,14]*W[0,14] = 1*1*1 = 1. Correct.
        # For i=1, k=3: U[5,14]*V[11,14]*W[7,14] = 1*1*1 = 1. Correct.
        
        # It works!
        
        # So we have a rank 15 decomposition.
        # Let's create the trimmed matrices.
        U_trim = [row[:15] for row in U]
        V_trim = [row[:15] for row in V]
        W_trim = [row[:15] for row in W]
        
        return U_trim, V_trim, W_trim, 15

    # Try to find lower rank decompositions using ALS
    # But rank 15 is already very good.
    # Lower bound for 2x3x4 matrix multiplication?
    # R(2x3x4) >= max(2*3, 3*4, 4*2) = 12? No, that's not a lower bound.
    # R(n x m x p) >= nm, mp, pn.
    # R >= 6, 12, 8. So R >= 12.
    # So 15 is close to the lower bound.
    
    # Let's try to verify rank 15.
    U15, V15, W15, r15 = build_rank_22()
    if check_decomposition(U15, V15, W15, n, m, p):
        return {"U": U15, "V": V15, "W": W15}
    
    # If verification fails (it shouldn't), fallback to schoolbook rank 24
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

# EVOLVE-BLOCK-END
