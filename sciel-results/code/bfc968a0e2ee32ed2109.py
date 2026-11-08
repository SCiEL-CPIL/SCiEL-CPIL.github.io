"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Rank 36 decomposition:
    # 4 Strassen 2x2x2 blocks (rank 7 each) = 28
    # 2 Matrix-Vector 2x2x1 blocks (rank 4 each) = 8
    # Total 36
    
    rank = 36
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    deadline = time.time() + 140.0

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def strassen_2x2x2(y00, y01, y10, y11, z00, z01, z10, z11, w00, w01, w10, w11):
        nonlocal col
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
        (0,0), (0,1), (1,0), (1,1),
        (0,0), (0,1), (1,0), (1,1),
        w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)
    )
    
    # Block 2: A_r * B_bot[:,0:2] -> C[:,0:2]
    strassen_2x2x2(
        (0,2), (0,3), (1,2), (1,3),
        (2,0), (2,1), (3,0), (3,1),
        w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)
    )
    
    # Block 3: A_l * B_top[:,2:4] -> C[:,2:4]
    strassen_2x2x2(
        (0,0), (0,1), (1,0), (1,1),
        (0,2), (0,3), (1,2), (1,3),
        w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)
    )
    
    # Block 4: A_r * B_bot[:,2:4] -> C[:,2:4]
    strassen_2x2x2(
        (0,2), (0,3), (1,2), (1,3),
        (2,2), (2,3), (3,2), (3,3),
        w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)
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
    
    # Convert to lists of integers
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
