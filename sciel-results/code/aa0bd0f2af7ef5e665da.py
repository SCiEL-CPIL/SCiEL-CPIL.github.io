"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 7]

def _build_2x2x2_strassen():
    # Strassen's 7-term algorithm for 2x2x2
    # C = A * B, A is 2x2, B is 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # We need to return U, V, W for the 4x4x4 tensor (indices 0..3 for each)
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i,j] corresponds to A_{i,j}, V[i,j] to B_{i,j}, W[i,j] to C_{i,j}
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Indices: A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 3] += 1
    
    # M2
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 2] += 1; W[3, 3] += 1
    
    # M3
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] += -1
    W[0, 1] += 1; W[3, 3] += 1
    
    # M4
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] += -1
    W[0, 0] += 1; W[2, 2] += 1
    
    # M5
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 0] += -1; W[0, 1] += 1
    
    # M6
    U[2, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 3] += 1
    
    # M7
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 0] += 1; W[2, 2] += -1
    
    return U, V, W

def _build_block(n, m, p):
    # Builds a decomposition for n x m x p multiplication
    # Uses block decomposition recursively if possible
    # For now, we'll just use schoolbook for small blocks or Strassen for 2x2x2
    # This function is a placeholder for a more general recursive builder.
    # For the specific case of 3x4x7, we will build it manually in construct().
    pass

def construct():
    # We want to find a decomposition for 3x4x7.
    # Strategy: Use the known rank 17 decomposition for 3x3x3? No, rank 3x3x3 is 19 or 20?
    # Actually, the best known rank for 3x3x3 is 19 (Landsberg) or 20? 
    # Wait, the best known upper bound for 3x3x3 is 19, and lower bound is 19.
    # So R(3,3,3) = 19.
    # We can use a 3x3x3 decomposition to help.
    # Let's try to build a 3x4x7 decomposition using blocks.
    # Split 4 into 1+3, 7 into 3+4?
    # Or use the fact that 3x4x7 can be embedded in 4x4x4?
    # No, that's not helpful for rank reduction.
    
    # Let's stick to a constructive approach.
    # We know R(2,2,2) = 7.
    # We can build a 3x4x7 decomposition by splitting the dimensions.
    # Let's try to build it using the Strassen algorithm for the 2x2x2 blocks.
    
    # Actually, let's just use the schoolbook for now to ensure correctness, 
    # but the directive asks for a stronger algorithm.
    # The previous attempt tried to use block decomposition to get rank 80.
    # Let's try to implement the rank 80 solution.
    
    # Split m=4 into 2+2.
    # For each m-block of size 2, we have a 3x2x7 multiplication.
    # Split p=7 into 3+3+1.
    # For each p-block, we have a 3x2x3 or 3x2x1 multiplication.
    # R(3,2,3) <= R(2,2,3) + R(1,2,3)
    # R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # R(1,2,3) <= R(1,2,2) + R(1,2,1) = 4 + 2 = 6.
    # So R(3,2,3) <= 17.
    # R(3,2,1) <= R(3,2,1) = 3*2*1 = 6.
    # So for one m-block: 17 + 17 + 6 = 40.
    # Total for m=4: 40 + 40 = 80.
    
    # Let's implement this.
    # We need to build the 3x2x3, 3x2x1, etc. decompositions.
    
    def build_2x2x2():
        # Returns U, V, W for 2x2x2 Strassen
        U, V, W = _build_2x2x2_strassen()
        return U, V, W

    def build_2x2x1():
        # 2x2x1: C = A * B, A is 2x2, B is 2x1, C is 2x1.
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # This is 4 multiplications.
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # M1: A00 * B0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # M2: A01 * B1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # M3: A10 * B0
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # M4: A11 * B1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    def build_1x2x2():
        # 1x2x2: C = A * B, A is 1x2, B is 2x2, C is 1x2.
        # C0 = A0*B00 + A1*B10
        # C1 = A0*B01 + A1*B11
        # This is 4 multiplications.
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        # M1: A0 * B00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # M2: A1 * B10
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # M3: A0 * B01
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        # M4: A1 * B11
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    def build_3x2x3():
        # 3x2x3: C = A * B, A is 3x2, B is 2x3, C is 3x3.
        # We can split A into 2+1 rows.
        # A1 = A[0:2, :], A2 = A[2:3, :]
        # C1 = A1 * B (2x3), C2 = A2 * B (1x3)
        # R(2,2,3) + R(1,2,3)
        U1, V1, W1 = build_2x2x3()
        U2, V2, W2 = build_1x2x3()
        # Concatenate
        U = np.vstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W

    def build_2x2x3():
        # 2x2x3: C = A * B, A is 2x2, B is 2x3, C is 2x3.
        # Split B into 2+1 columns.
        # B1 = B[:, 0:2], B2 = B[:, 2:3]
        # C1 = A * B1 (2x2), C2 = A * B2 (2x1)
        # R(2,2,2) + R(2,2,1)
        U1, V1, W1 = build_2x2x2()
        U2, V2, W2 = build_2x2x1()
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W

    def build_1x2x3():
        # 1x2x3: C = A * B, A is 1x2, B is 2x3, C is 1x3.
        # Split B into 2+1 columns.
        # B1 = B[:, 0:2], B2 = B[:, 2:3]
        # C1 = A * B1 (1x2), C2 = A * B2 (1x1)
        # R(1,2,2) + R(1,2,1)
        U1, V1, W1 = build_1x2x2()
        U2, V2, W2 = build_1x2x1()
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W

    def build_1x2x1():
        # 1x2x1: C = A * B, A is 1x2, B is 2x1, C is 1x1.
        # C0 = A0*B0 + A1*B1
        # This is 2 multiplications.
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    def build_3x2x1():
        # 3x2x1: C = A * B, A is 3x2, B is 2x1, C is 3x1.
        # This is 6 multiplications.
        U = np.zeros((6, 6))
        V = np.zeros((2, 6))
        W = np.zeros((3, 6))
        for i in range(3):
            for j in range(2):
                col = i * 2 + j
                U[i * 2 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
        return U, V, W

    # Now build the 3x4x7 decomposition.
    # Split m=4 into 2+2.
    # For each m-block, split p=7 into 3+3+1.
    U_total = []
    V_total = []
    W_total = []
    
    for m_block in range(2):
        # For each m-block, we have a 3x2x7 multiplication.
        for p_block in range(3):
            if p_block < 2:
                # 3x2x3 block
                U_b, V_b, W_b = build_3x2x3()
            else:
                # 3x2x1 block
                U_b, V_b, W_b = build_3x2x1()
            
            # We need to map these local indices to the global indices.
            # Local A is 3x2, global A is 3x4.
            # Local A indices: (i, j) where i in 0..2, j in 0..1.
            # Global A indices: (i, j_global) where i in 0..2, j_global in 0..3.
            # j_global = m_block * 2 + j.
            
            # Local B is 2x3 (or 2x1), global B is 4x7.
            # Local B indices: (j, k) where j in 0..1, k in 0..2 (or 0..0).
            # Global B indices: (j_global, k_global) where j_global in 0..3, k_global in 0..6.
            # j_global = m_block * 2 + j.
            # k_global = p_block * 3 + k (if p_block < 2) or p_block * 3 + k (if p_block == 2, k is 0).
            # Actually, for p_block == 2, the size is 1, so k is 0.
            # Let's just use the local indices and map them.
            
            # Let's construct the global U, V, W by placing the local blocks.
            # This is a bit tricky because the local U, V, W are already in a specific format.
            # We will just create new global matrices and fill them in.
            
            n, m, p = 3, 4, 7
            rank_total = 80
            if not U_total:
                U_total = np.zeros((n * m, rank_total))
                V_total = np.zeros((m * p, rank_total))
                W_total = np.zeros((p * n, rank_total))
            
            # We need to know the current column offset in the global matrices.
            # This is getting complicated. Let's just build the whole thing at once.
            pass

    # This manual construction is error-prone. Let's just use the schoolbook for now.
    # The directive says to use a stronger algorithm, but if it's too complex, 
    # it's better to be correct with schoolbook than incorrect with a heuristic.
    # However, the previous attempt was rejected for being empty or low quality.
    # Let's try to implement the rank 80 solution correctly.
    
    # We will build the global matrices directly.
    n, m, p = 3, 4, 7
    rank_total = 80
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col_offset = 0
    for m_block in range(2):
        for p_block in range(3):
            if p_block < 2:
                # 3x2x3 block, rank 17
                U_b, V_b, W_b = build_3x2x3()
                r_b = U_b.shape[1]
            else:
                # 3x2x1 block, rank 6
                U_b, V_b, W_b = build_3x2x1()
                r_b = U_b.shape[1]
            
            # Map local indices to global indices.
            for i in range(3):
                for j in range(2):
                    j_global = m_block * 2 + j
                    for r in range(r_b):
                        U[i * m + j_global, col_offset + r] = U_b[i * 2 + j, r]
            
            for j in range(2):
                j_global = m_block * 2 + j
                if p_block < 2:
                    for k in range(3):
                        k_global = p_block * 3 + k
                        for r in range(r_b):
                            V[j_global * p + k_global, col_offset + r] = V_b[j * 3 + k, r]
                else:
                    k_global = p_block * 3
                    for r in range(r_b):
                        V[j_global * p + k_global, col_offset + r] = V_b[j * 1 + 0, r]
            
            if p_block < 2:
                for k in range(3):
                    k_global = p_block * 3 + k
                    for i in range(3):
                        for r in range(r_b):
                            W[k_global * n + i, col_offset + r] = W_b[k * 3 + i, r]
            else:
                k_global = p_block * 3
                for i in range(3):
                    for r in range(r_b):
                        W[k_global * n + i, col_offset + r] = W_b[0 * 3 + i, r]
            
            col_offset += r_b
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
