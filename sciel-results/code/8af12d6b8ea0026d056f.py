import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 8
    
    # Build the target tensor
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1
    
    # Strategy: try to find a rank-40 decomposition by improving the 2x3x3x2 block
    # Current: 4 blocks x rank 11 = 44
    # Target: 4 blocks x rank 10 = 40, or better
    
    # Let's try to find a rank-10 decomposition for 2x3x3x2
    # A is 2x3, B is 3x2, C is 2x2
    # Tensor T[a,b,c] where a=i*3+j, b=j*2+k, c=k*2+i
    
    T_block = np.zeros((6, 6, 4), dtype=np.int64)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                T_block[i*3+j, j*2+k, k*2+i] = 1
    
    # Try rank 10: set up the system
    # We want U (6,10), V (6,10), W (4,10) such that sum_d U[:,d] outer V[:,d] outer W[:,d] = T_block
    # This is a nonlinear system. Let's try a structured approach.
    
    # Alternative: try rank 9
    # Let's try a direct construction.
    
    # For 2x3x3x2, let's try to use the following idea:
    # Split B into B0 (row 0) and B1 (rows 1-2)
    # C = A[:,0] * B0 + A[:,1:3] * B1
    # A[:,0] * B0 is rank 4 (2x1 * 1x2)
    # A[:,1:3] * B1 is 2x2 * 2x2, rank 7 (Strassen)
    # Total: 11
    
    # Can we do better? Let's try a different split.
    # What if we split A instead?
    # C = [A[0,:], A[1,:]] * B
    # This doesn't help directly.
    
    # Let's try to find if rank 10 is possible by solving the system.
    # For small ranks, we can try a grid search or use the fact that entries are integers.
    
    # Actually, let's try a smarter approach: use the known result that
    # rank(2x3x3x2) might be 10 or 9.
    
    # Let me try to construct a rank-10 solution directly.
    # 
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    #
    # Try to share:
    # P1 = (a00+a10) * b00
    # P2 = (a00+a10) * b01
    # P3 = a01*b10 + a11*b10 = (a01+a11)*b10
    # P4 = a01*b11 + a11*b11 = (a01+a11)*b11
    # P5 = a02*b20 + a12*b20 = (a02+a12)*b20
    # P6 = a02*b21 + a12*b21 = (a02+a12)*b21
    #
    # C[0,0] = P1 + P3 + P5 - a10*b00
    # This doesn't work cleanly.
    
    # Let me try a computational search for rank 10.
    # We'll parameterize and try to solve.
    
    # Given time constraints, let's try a different strategy:
    # Use the rank-44 solution but try to find a rank-40 by improving the block.
    
    # Let's try rank 10 for the block using a linear algebra approach.
    # If we fix U and V, W is determined (if the system is consistent).
    
    # For now, let's stick with rank 44 but try to verify if a lower rank is feasible.
    
    # Actually, let's try a completely different approach:
    # Use the fact that the full 2x3x3x8 tensor might have a lower rank than 4*11.
    # The rank of the full tensor is at most 4*11 = 44, but could be lower if there's
    # sharing across blocks.
    
    # Let's try to find a rank-40 decomposition of the full tensor.
    # We'll set up the problem and try to solve it.
    
    # For efficiency, let's try a structured ansatz.
    # Assume the solution has a block structure with 4 blocks, each of rank 10.
    
    # Try to construct a rank-10 block:
    # We need U (6,10), V (6,10), W (4,10) for the 2x3x3x2 block.
    
    # Let's try the following ansatz:
    # Use 4 products for the "diagonal" part and 6 for the "off-diagonal" part.
    
    # Actually, let me just try to solve the rank-10 system numerically.
    
    # Given the complexity, let's return the rank-44 solution for now,
    # but try to improve if time permits.
    
    U = np.zeros((n*m, 44), dtype=np.int64)
    V = np.zeros((m*p, 44), dtype=np.int64)
    W = np.zeros((p*n, 44), dtype=np.int64)
    
    col_offset = 0
    
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Block 1: A[:, 0] * B[0, k0:k1]  (2x1 * 1x2) - rank 4
        U[0*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1
        
        U[0*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        # Block 2: A[:, 1:3] * B[1:3, k0:k1]  (2x2 * 2x2) - Strassen rank 7
        U[0*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        W[k1*n + 1, col_offset] = -1
        col_offset += 1
        
        U[0*m + 1, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        V[2*p + k1, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 2, col_offset] = 1
        V[2*p + k0, col_offset] = 1
        V[1*p + k0, col_offset] = -1
        W[k0*n + 0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        U[0*m + 1, col_offset] = 1
        U[0*m + 2, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        U[1*m + 1, col_offset] = 1
        U[0*m + 1, col_offset] = -1
        V[1*p + k0, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[0*m + 2, col_offset] = 1
        U[1*m + 2, col_offset] = -1
        V[2*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1

    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(T_check, T_exp)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, col] = 1
                    V_sb[b_idx, col] = 1
                    W_sb[c_idx, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
