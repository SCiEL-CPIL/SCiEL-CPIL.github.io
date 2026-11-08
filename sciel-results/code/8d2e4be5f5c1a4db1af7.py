"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We'll build a rank-36 decomposition using 4 Strassen blocks + 8 mat-vec products
    # Strategy:
    # - Split A into A_l (cols 0,1) and A_r (cols 2,3)
    # - Split B into B_top (rows 0,1) and B_bot (rows 2,3)
    # - For columns 0-3: use 2 Strassen blocks per pair of columns
    # - For column 4: use direct mat-vec (4 mults each for A_l*b_top and A_r*b_bot)
    
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    # Strassen 2x2x2 block
    # Y = [[y00, y01], [y10, y11]], Z = [[z00, z01], [z10, z11]]
    # Produces X = Y*Z
    def strassen_2x2x2(y_indices, z_indices, w_indices):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w01, w10, w11 = w_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1
        V[v_idx(*z11)][col] -= 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z00)][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z01)][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] += 1
        U[u_idx(*y11)][col] -= 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w00][col] += 1
        col += 1
    
    # Block 1: A_l * B_top[:,0:2] -> C[:,0:2]
    strassen_2x2x2(
        [(0,0), (0,1), (1,0), (1,1)],  # A_l
        [(0,0), (0,1), (1,0), (1,1)],  # B_top[:,0:2]
        [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)]
    )
    
    # Block 2: A_r * B_bot[:,0:2] -> C[:,0:2]
    strassen_2x2x2(
        [(0,2), (0,3), (1,2), (1,3)],  # A_r
        [(2,0), (2,1), (3,0), (3,1)],  # B_bot[:,0:2]
        [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)]
    )
    
    # Block 3: A_l * B_top[:,2:4] -> C[:,2:4]
    strassen_2x2x2(
        [(0,0), (0,1), (1,0), (1,1)],  # A_l
        [(0,2), (0,3), (1,2), (1,3)],  # B_top[:,2:4]
        [w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)]
    )
    
    # Block 4: A_r * B_bot[:,2:4] -> C[:,2:4]
    strassen_2x2x2(
        [(0,2), (0,3), (1,2), (1,3)],  # A_r
        [(2,2), (2,3), (3,2), (3,3)],  # B_bot[:,2:4]
        [w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)]
    )
    
    # Block 5: A_l * b_top (col 4) -> C[:,4]
    # C[0,4] += A[0,0]*B[0,4] + A[0,1]*B[1,4]
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # C[1,4] += A[1,0]*B[0,4] + A[1,1]*B[1,4]
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 6: A_r * b_bot (col 4) -> C[:,4]
    # C[0,4] += A[0,2]*B[2,4] + A[0,3]*B[3,4]
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # C[1,4] += A[1,2]*B[2,4] + A[1,3]*B[3,4]
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Trim to actual rank used
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    # Convert to lists of integers (entries are 0, ±1)
    U = U.astype(int).tolist()
    V = V.astype(int).tolist()
    W = W.astype(int).tolist()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Should not happen for this fixed construction, but guard anyway
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
