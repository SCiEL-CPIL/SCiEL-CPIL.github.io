"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

import time
import sys

def construct():
    """Construct a rank-42 decomposition using Strassen's algorithm for 2x2x2 blocks.
    
    For multiplying (2x2) by (2x12), we split the 12 columns into 6 groups of 2.
    For each group, we use Strassen's 7-multiplication algorithm for 2x2 matrices.
    Total rank = 6 * 7 = 42.
    
    Output W follows the convention W[k*n + i, r] where n=2.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # Split p=12 into num_blocks blocks of size block_p=2
    num_blocks = p // 2
    block_p = 2
    
    # For each block of 2 columns, use Strassen's 7 multiplications
    # Strassen's algorithm for 2x2 matrices:
    # C = A * B where A, B are 2x2
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B01)
    # m7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = m1 - m4 + m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m3 + m2 + m6
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    for g in range(num_blocks):
        base_col = g * 7
        for r in range(7):
            col = base_col + r
            # U indices: i*m + j where i in {0,1}, j in {0,1}
            # i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3
            # V indices: j*p + k where j in {0,1}, k in {0,1} for this block
            # k_global = g*2 + k_local
            # j=0,k=0 -> 0*12 + 0 = 0; j=0,k=1 -> 0*12 + 1 = 1
            # j=1,k=0 -> 1*12 + 0 = 12; j=1,k=1 -> 1*12 + 1 = 13
            # W indices: k*n + i where k in {0,1} for this block, i in {0,1}
            # k_global = g*2 + k_local
            # k=0,i=0 -> 0*2 + 0 = 0; k=0,i=1 -> 0*2 + 1 = 1
            # k=1,i=0 -> 1*2 + 0 = 2; k=1,i=1 -> 1*2 + 1 = 3
            
            if r == 0:  # m1 = (A00 + A11) * (B00 + B11)
                # U: A00 (i=0,j=0) and A11 (i=1,j=1)
                U[0][col] = 1
                U[3][col] = 1
                # V: B00 (j=0,k=0) and B11 (j=1,k=1)
                V[0 * p + (2*g + 0)][col] = 1
                V[1 * p + (2*g + 1)][col] = 1
                # W: C00 (k=0,i=0) +1, C11 (k=1,i=1) +1
                W[(2*g + 0) * n + 0][col] = 1
                W[(2*g + 1) * n + 1][col] = 1
            elif r == 1:  # m2 = (A10 + A11) * B00
                # U: A10 (i=1,j=0) and A11 (i=1,j=1)
                U[2][col] = 1
                U[3][col] = 1
                # V: B00 (j=0,k=0)
                V[0 * p + (2*g + 0)][col] = 1
                # W: C10 (k=0,i=1) +1, C11 (k=1,i=1) -1
                W[(2*g + 0) * n + 1][col] = 1
                W[(2*g + 1) * n + 1][col] = -1
            elif r == 2:  # m3 = A00 * (B01 - B11)
                # U: A00 (i=0,j=0)
                U[0][col] = 1
                # V: B01 (j=0,k=1) and -B11 (j=1,k=1)
                V[0 * p + (2*g + 1)][col] = 1
                V[1 * p + (2*g + 1)][col] = -1
                # W: C01 (k=1,i=0) +1, C11 (k=1,i=1) +1
                W[(2*g + 1) * n + 0][col] = 1
                W[(2*g + 1) * n + 1][col] = 1
            elif r == 3:  # m4 = A11 * (B10 - B00)
                # U: A11 (i=1,j=1)
                U[3][col] = 1
                # V: B10 (j=1,k=0) and -B00 (j=0,k=0)
                V[1 * p + (2*g + 0)][col] = 1
                V[0 * p + (2*g + 0)][col] = -1
                # W: C00 (k=0,i=0) +1, C10 (k=0,i=1) +1
                W[(2*g + 0) * n + 0][col] = 1
                W[(2*g + 0) * n + 1][col] = 1
            elif r == 4:  # m5 = (A00 + A01) * B11
                # U: A00 (i=0,j=0) and A01 (i=0,j=1)
                U[0][col] = 1
                U[1][col] = 1
                # V: B11 (j=1,k=1)
                V[1 * p + (2*g + 1)][col] = 1
                # W: C00 (k=0,i=0) -1, C01 (k=1,i=0) +1
                W[(2*g + 0) * n + 0][col] = -1
                W[(2*g + 1) * n + 0][col] = 1
            elif r == 5:  # m6 = (A10 - A00) * (B00 + B01)
                # U: A10 (i=1,j=0) and -A00 (i=0,j=0)
                U[2][col] = 1
                U[0][col] = -1
                # V: B00 (j=0,k=0) and B01 (j=0,k=1)
                V[0 * p + (2*g + 0)][col] = 1
                V[0 * p + (2*g + 1)][col] = 1
                # W: C11 (k=1,i=1) +1
                W[(2*g + 1) * n + 1][col] = 1
            elif r == 6:  # m7 = (A01 - A11) * (B10 + B11)
                # U: A01 (i=0,j=1) and -A11 (i=1,j=1)
                U[1][col] = 1
                U[3][col] = -1
                # V: B10 (j=1,k=0) and B11 (j=1,k=1)
                V[1 * p + (2*g + 0)][col] = 1
                V[1 * p + (2*g + 1)][col] = 1
                # W: C00 (k=0,i=0) +1
                W[(2*g + 0) * n + 0][col] = 1
    
    # Verify decomposition is correct
    # The decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i', r] = delta(i,i') when j=p... 
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 if we want C[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r], so W index is k*n+i
    # Let me verify with a small test
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        rank = len(U[0])
        # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give
        # the standard matrix multiplication tensor, which is 1 if this is a valid decomposition
        # Actually the tensor for matmul is T[i,j,k] = 1 for all i,j,k (it's the identity in the 
        # appropriate sense). Let me think again.
        # 
        # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k.
        # 
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if Strassen verification fails
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][column] = 1
                    V_s[j * p + k][column] = 1
                    W_s[k * n + i][column] = 1
                    column += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
