import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p, seed=42):
    """Verify the decomposition is correct."""
    rng = np.random.RandomState(seed)
    A = rng.rand(n, m)
    B = rng.rand(m, p)
    C_expected = A @ B
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    C_decomp = np.zeros((n, p))
    for r in range(U.shape[1]):
        u = U_r[:, :, r]
        v = V_r[:, :, r]
        w = W_r[:, :, r]
        temp = u @ v
        C_decomp += temp * w.T
    
    if not np.allclose(C_expected, C_decomp, atol=1e-10):
        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (Y00 + Y11)(Z00 + Z11)
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2 = Y10(Z00 + Z01)
        U[u_idx(y10[0], y10[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M3 = Y00(Z11 - Z00)
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4 = Y11(Z10 - Z00)
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] -= 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5 = (Y00 + Y01)Z11
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6 = (Y10 - Y00)(Z00 + Z01)
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y00[0], y00[1])][col] -= 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7 = (Y01 - Y11)(Z10 + Z11)
        U[u_idx(y01[0], y01[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] -= 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L (2x2) * B_11 (2x2) -> C_11 (2x2)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R (2x2) * B_21 (2x2) -> C_11 (2x2)
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L (2x2) * B_12 (2x2) -> C_12 (2x2)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (2x2) * B_22 (2x2) -> C_12 (2x2)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (2x2) * B_15 (2x1) -> C_15 (2x1), rank 4
    U[u_idx(0,0)][col] += 1; V[v_idx(0,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(0,1)][col] += 1; V[v_idx(1,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(1,0)][col] += 1; V[v_idx(0,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
    U[u_idx(1,1)][col] += 1; V[v_idx(1,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
    
    # Block 6: A_R (2x2) * B_25 (2x1) -> C_15 (2x1), rank 4
    U[u_idx(0,2)][col] += 1; V[v_idx(2,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(0,3)][col] += 1; V[v_idx(3,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
    U[u_idx(1,2)][col] += 1; V[v_idx(2,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
    U[u_idx(1,3)][col] += 1; V[v_idx(3,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
