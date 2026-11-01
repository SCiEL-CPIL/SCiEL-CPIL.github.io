"""Rewritten program for matrix multiplication tensor rank minimization (2x4 * 4x4)."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0  # seconds (safety margin under 150 s wall-clock)

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n x m by m x p multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, c]
    T[i,j,k] = 1 for all i,j,k
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    T_rec = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        for k in range(p):
            Wck = Wc[k, :]  # shape (n,)
            Vck = Vc[:, k]  # shape (m,)
            T_rec[:, :, k] += Uc * np.outer(Wck, Vck)
    T_true = np.ones((n, m, p))
    return np.allclose(T_rec, T_true, atol=tol)


def _build_strassen_decomposition():
    """Build a rank-28 decomposition using 4 independent Strassen 2x2x2 blocks.
    
    Partition:
    - A is 2x4: rows i in {0,1}, cols j in {0,1,2,3}
    - B is 4x4: rows j in {0,1,2,3}, cols k in {0,1,2,3}
    - C is 2x4: rows i in {0,1}, cols k in {0,1,2,3}
    
    Split j into {0,1} and {2,3}, split k into {0,1} and {2,3}.
    This gives 4 sub-products of size 2x2 * 2x2 = 2x2:
      Product 0: A[:,0:2] * B[0:2,0:2] -> C[:,0:2] (contribution to cols 0,1)
      Product 1: A[:,2:4] * B[2:4,0:2] -> C[:,0:2] (contribution to cols 0,1)
      Product 2: A[:,0:2] * B[0:2,2:4] -> C[:,2:4] (contribution to cols 2,3)
      Product 3: A[:,2:4] * B[2:4,2:4] -> C[:,2:4] (contribution to cols 2,3)
    
    Each 2x2*2x2->2x2 uses Strassen with rank 7.
    Total rank: 4 * 7 = 28.
    """
    n, m, p = 2, 4, 4
    rank_total = 28
    
    U = np.zeros((n * m, rank_total))  # (8, 28)
    V = np.zeros((m * p, rank_total))  # (16, 28)
    W = np.zeros((p * n, rank_total))  # (8, 28)
    
    # Strassen formulas for 2x2 matrix multiplication:
    # Given A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a10)*b11
    # M6 = (b00+b01)*(a10-a00)
    # M7 = (b10+b11)*(a01-a11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each of the 4 blocks, we need to map:
    # Block b: A_sub is 2x2 (rows i=0,1; cols j=base_j+0, base_j+1)
    #           B_sub is 2x2 (rows j=base_j+0, base_j+1; cols k=base_k+0, base_k+1)
    #           C_sub is 2x2 (rows i=0,1; cols k=base_k+0, base_k+1)
    
    # Strassen terms for a single 2x2 block:
    # Term t=0 (M1): U contributes to (a00+a11), V contributes to (b00+b11), W distributes to C
    #   U[i, base_j + 0] += 1, U[i, base_j + 1] += 1 for i in {0,1}
    #   V[base_j + 0, base_k + 0] += 1, V[base_j + 0, base_k + 1] += 1
    #   V[base_j + 1, base_k + 0] += 1, V[base_j + 1, base_k + 1] += 1
    #   W[base_k + 0, 0] += 1, W[base_k + 1, 0] += 1  (C00 gets +M1)
    #   W[base_k + 0, 1] += 0, W[base_k + 1, 1] += 0  (C01 gets 0 from M1)
    #   Wait, W index is W[k*n + i, c], so W[base_k+0, 0] means k=base_k, i=0
    #   C00 contribution: W[base_k+0, 0] for i=0, W[base_k+1, 0] for i=0? No.
    #   W[k*n + i, c]: for k=base_k+0, i=0: W[base_k*2+0, c]
    #   for k=base_k+0, i=1: W[base_k*2+1, c]
    #   for k=base_k+1, i=0: W[(base_k+1)*2+0, c]
    #   for k=base_k+1, i=1: W[(base_k+1)*2+1, c]
    
    # Let me redo this more carefully.
    # W has shape (p*n, rank) = (8, 28), indexed as W[k*n + i, c] = W[k*2 + i, c]
    
    # For a Strassen term computing M = (sum of A entries) * (sum of B entries):
    # The contribution to C[i, k] from this term is:
    #   U_contrib[i, j_local] * V_contrib[j_local, k_local] * W_contrib[k_local, i]
    # where j_local and k_local are local indices within the 2x2 block.
    
    # For M1 = (a00+a11)*(b00+b11):
    #   U: for i in {0,1}, U[i, j_base+0] += 1, U[i, j_base+1] += 1
    #   V: for j_local in {0,1}, V[j_base+j_local, k_base+0] += 1, V[j_base+j_local, k_base+1] += 1
    #   W: contributes to C00 and C11 (both get +M1)
    #     C00: i=0, k_local=0 -> W[k_base*2 + 0, c] += 1
    #     C11: i=1, k_local=1 -> W[(k_base+1)*2 + 1, c] += 1
    #     Also C00 for i=0: W[k_base*2+0, c] += 1 (already counted)
    #     Wait, C[i, k] where i is row of C (0 or 1), k is col of C.
    #     In the block, C_local[i_local, k_local]:
    #       C_local[0,0] = M1+M4-M5+M7
    #       C_local[0,1] = M3+M5
    #       C_local[1,0] = M2+M4
    #       C_local[1,1] = M1-M2+M3+M6
    #     So M1 contributes +1 to C_local[0,0] and +1 to C_local[1,1].
    #     C_local[0,0] corresponds to i=0, k=k_base+0 -> W[(k_base+0)*2 + 0, c] += 1
    #     C_local[1,1] corresponds to i=1, k=k_base+1 -> W[(k_base+1)*2 + 1, c] += 1
    
    # Let me define the Strassen template:
    # For each term t in 0..6, define:
    #   U_pattern: 2x2 matrix (i_local, j_local)
    #   V_pattern: 2x2 matrix (j_local, k_local)  
    #   W_pattern: 2x2 matrix (k_local, i_local)
    
    # M1 = (a00+a11)*(b00+b11)
    #   U: [[1,1],[1,1]]
    #   V: [[1,1],[1,1]]
    #   W: [[1,0],[0,1]]  (C00 and C11)
    
    # M2 = (a10+a11)*b00
    #   U: [[0,0],[1,1]]
    #   V: [[1,0],[0,0]]
    #   W: [[0,0],[1,0]]  (C10)
    
    # M3 = a00*(b01-b11)
    #   U: [[1,0],[0,0]]
    #   V: [[0,1],[0,-1]]
    #   W: [[0,1],[0,0]]  (C01)
    
    # M4 = a11*(b10-b00)
    #   U: [[0,0],[0,1]]
    #   V: [[-1,0],[1,0]]
    #   W: [[1,0],[0,1]]  (C00 and C10)
    
    # M5 = (a00+a10)*b11
    #   U: [[1,1],[0,1]]
    #   V: [[0,0],[0,1]]
    #   W: [[-1,0],[0,1]]  (C00 gets -M5, C11 gets +M5)
    
    # M6 = (b00+b01)*(a10-a00)
    #   U: [[-1,0],[1,0]]
    #   V: [[1,1],[0,0]]
    #   W: [[0,0],[0,1]]  (C11)
    
    # M7 = (b10+b11)*(a01-a11)
    #   U: [[0,1],[0,-1]]
    #   V: [[0,0],[1,1]]
    #   W: [[1,0],[0,0]]  (C00)
    
    strassen_U = [
        [[1,1],[1,1]],   # M1
        [[0,0],[1,1]],   # M2
        [[1,0],[0,0]],   # M3
        [[0,0],[0,1]],   # M4
        [[1,1],[0,1]],   # M5
        [[-1,0],[1,0]],  # M6
        [[0,1],[0,-1]],  # M7
    ]
    strassen_V = [
        [[1,1],[1,1]],   # M1
        [[1,0],[0,0]],   # M2
        [[0,1],[0,-1]],  # M3
        [[-1,0],[1,0]],  # M4
        [[0,0],[0,1]],   # M5
        [[1,1],[0,0]],   # M6
        [[0,0],[1,1]],   # M7
    ]
    strassen_W = [
        [[1,0],[0,1]],   # M1: C00, C11
        [[0,0],[1,0]],   # M2: C10
        [[0,1],[0,0]],   # M3: C01
        [[1,0],[0,1]],   # M4: C00, C10
        [[-1,0],[0,1]],  # M5: -C00, +C11
        [[0,0],[0,1]],   # M6: C11
        [[1,0],[0,0]],   # M7: C00
    ]
    
    # The 4 blocks:
    # Block 0: j_base=0, k_base=0 (A[:,0:2] * B[0:2,0:2] -> C[:,0:2])
    # Block 1: j_base=2, k_base=0 (A[:,2:4] * B[2:4,0:2] -> C[:,0:2])
    # Block 2: j_base=0, k_base=2 (A[:,0:2] * B[0:2,2:4] -> C[:,2:4])
    # Block 3: j_base=2, k_base=2 (A[:,2:4] * B[2:4,2:4] -> C[:,2:4])
    
    blocks = [(0, 0), (2, 0), (0, 2), (2, 2)]
    
    for block_idx, (j_base, k_base) in enumerate(blocks):
        for t in range(7):
            col = block_idx * 7 + t
            
            # Fill U: shape (n*m, rank) = (8, 28)
            # U[i*m + j, col] where i in {0,1}, j = j_base + j_local
            for i_local in range(2):
                for j_local in range(2):
                    val = strassen_U[t][i_local][j_local]
                    if val != 0:
                        U[i_local * m + (j_base + j_local), col] = val
            
            # Fill V: shape (m*p, rank) = (16, 28)
            # V[j*p + k, col] where j = j_base + j_local, k = k_base + k_local
            for j_local in range(2):
                for k_local in range(2):
                    val = strassen_V[t][j_local][k_local]
                    if val != 0:
                        V[(j_base + j_local) * p + (k_base + k_local), col] = val
            
            # Fill W: shape (p*n, rank) = (8, 28)
            # W[k*n + i, col] where k = k_base + k_local, i = i_local
            for k_local in range(2):
                for i_local in range(2):
                    val = strassen_W[t][k_local][i_local]
                    if val != 0:
                        W[(k_base + k_local) * n + i_local, col] = val
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    
    # First, try the structural Strassen decomposition (rank 28)
    U_str, V_str, W_str = _build_strassen_decomposition()
    
    if _verify(U_str, V_str, W_str, n, m, p, tol=1e-9):
        result = {'U': U_str.tolist(), 'V': V_str.tolist(), 'W': W_str.tolist()}
        return result
    
    # If Strassen verification fails, fall back to schoolbook
    rank_fallback = n * m * p
    U_fb = np.zeros((n * m, rank_fallback))
    V_fb = np.zeros((m * p, rank_fallback))
    W_fb = np.zeros((p * n, rank_fallback))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    return {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}

# EVOLVE-BLOCK-END
