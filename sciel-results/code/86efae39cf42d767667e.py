import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    start_time = time.time()
    deadline = start_time + 140

    def make_schoolbook():
        rank = n * m * p  # 64
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def numeric_check(U, V, W, A, B, tol=1e-6):
        C = A @ B
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for t in range(U.shape[1]):
                        total += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(total - C[i,k]) > tol:
                    return False
        return True

    def make_strassen_block():
        """2x2*2x2 Strassen, rank 7. U indexed [i*2+j], V [j*2+k], W [k*2+i] (n=2)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1
        U[0,4]=1; U[2,4]=1; V[3,4]=1
        U[1,5]=1; U[3,5]=-1; V[0,5]=1; V[2,5]=1
        U[2,6]=1; U[0,6]=-1; V[1,6]=1; V[3,6]=1
        W[0,0]=1; W[0,4]=-1; W[0,3]=1; W[0,6]=1
        W[1,2]=1; W[1,4]=1
        W[2,1]=1; W[2,3]=1
        W[3,0]=1; W[3,2]=-1; W[3,5]=1; W[3,1]=-1
        return U, V, W

    def make_2x2_by_2x8_block():
        """
        2x2 A * 2x8 B -> 2x8 C, rank 16.
        A indices: i in {0,1}, j in {0,1} -> U index i*2+j
        B indices: j in {0,1}, k in {0..7} -> V index j*8+k
        C indices: k in {0..7}, i in {0,1} -> W index k*2+i
        Built via Kronecker product of Strassen 2x2*2x2 (rank 7) and schoolbook 2x4 (rank 4).
        """
        U_s, V_s, W_s = make_strassen_block()  # shapes (4,7), (4,7), (4,7)
        
        # Schoolbook for 2x2 * 2x4: rank 8
        # A2: i in {0,1}, j in {0,1} -> U2 index i*2+j
        # B2: j in {0,1}, k in {0..3} -> V2 index j*4+k
        # C2: k in {0..3}, i in {0,1} -> W2 index k*2+i
        U2 = np.zeros((4, 8))
        V2 = np.zeros((8, 8))
        W2 = np.zeros((8, 8))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    U2[i*2+j, col] = 1
                    V2[j*4+k, col] = 1
                    W2[k*2+i, col] = 1
                    col += 1
        
        # Now combine: the 2x2*2x8 can be seen as doing two independent 2x2*2x4 multiplications
        # Actually, let's think differently.
        # A is 2x2, B is 2x8. We can split B into two 2x4 blocks: B_left (cols 0-3) and B_right (cols 4-7).
        # C = A * B = [A*B_left | A*B_right]
        # Each A*B is 2x4. 
        # So we need two independent 2x2*2x4 multiplications.
        # 2x2*2x4 schoolbook rank is 8. So total rank would be 16.
        # Let's build it properly.
        
        # For the left half (k in 0..3):
        U_left = np.zeros((4, 8))
        V_left = np.zeros((8, 8))
        W_left = np.zeros((8, 8))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    U_left[i*2+j, col] = 1
                    V_left[j*4+k, col] = 1
                    W_left[k*2+i, col] = 1
                    col += 1
        
        # For the right half (k in 4..7):
        U_right = np.zeros((4, 8))
        V_right = np.zeros((8, 8))
        W_right = np.zeros((8, 8))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    U_right[i*2+j, col] = 1
                    V_right[j*4+(k+4), col] = 1
                    W_right[(k+4)*2+i, col] = 1
                    col += 1
        
        # Combine: total rank 16
        U_block = np.hstack([U_left, U_right])  # (4, 16)
        V_block = np.hstack([V_left, V_right])  # (8, 16)
        W_block = np.hstack([W_left, W_right])  # (8, 16)
        
        return U_block, V_block, W_block

    # Now build the full 2x4 * 4x8 as two independent 2x2 * 2x8 blocks
    # Split A (2x4) into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # Split B (4x8) into B_top (2x8, rows 0-1) and B_bottom (2x8, rows 2-3)
    # C = A_left * B_top + A_right * B_bottom
    # Each term is 2x8.
    
    U_block, V_block, W_block = make_2x2_by_2x8_block()
    rank_total = 2 * 16  # 32
    
    U_full = np.zeros((U_dim, rank_total))
    V_full = np.zeros((V_dim, rank_total))
    W_full = np.zeros((W_dim, rank_total))
    
    # Block 0: A_left (cols 0-1) * B_top (rows 0-1)
    # A_left indices: i in {0,1}, j in {0,1} -> U_full index i*4 + j
    # B_top indices: j in {0,1}, k in {0..7} -> V_full index j*8 + k
    # C contribution: k in {0..7}, i in {0,1} -> W_full index k*2 + i
    for t in range(16):
        for i in range(2):
            for j in range(2):
                U_full[i*4 + j, t] += U_block[i*2+j, t]
        for j in range(2):
            for k in range(8):
                V_full[j*8 + k, t] += V_block[j*8+k, t]
        for i in range(2):
            for k in range(8):
                W_full[k*2 + i, t] += W_block[k*2+i, t]
    
    # Block 1: A_right (cols 2-3) * B_bottom (rows 2-3)
    # A_right indices: i in {0,1}, j in {0,1} -> U_full index i*4 + (2+j)
    # B_bottom indices: j in {0,1}, k in {0..7} -> V_full index (2+j)*8 + k
    # C contribution: k in {0..7}, i in {0,1} -> W_full index k*2 + i
    for t in range(16):
        for i in range(2):
            for j in range(2):
                U_full[i*4 + (2+j), 16+t] += U_block[i*2+j, t]
        for j in range(2):
            for k in range(8):
                V_full[(2+j)*8 + k, 16+t] += V_block[j*8+k, t]
        for i in range(2):
            for k in range(8):
                W_full[k*2 + i, 16+t] += W_block[k*2+i, t]
    
    # Verify with random matrices
    rng = np.random.RandomState(0)
    A = rng.randint(-5, 6, size=(n, m)).astype(float)
    B = rng.randint(-5, 6, size=(m, p)).astype(float)
    if numeric_check(U_full, V_full, W_full, A, B, tol=1e-6):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback schoolbook
    U0, V0, W0, r0 = make_schoolbook()
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
