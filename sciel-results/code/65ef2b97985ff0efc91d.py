"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s guard for 120s limit
    
    n, m, p = 2, 4, 5
    
    # We use a rank 36 decomposition based on block Strassen.
    # Structure:
    # - 4 blocks of 2x2x2 using Strassen (rank 7 each) = 28
    # - 2 blocks of 2x2x1 using direct (rank 4 each) = 8
    # Total = 36
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        """
        Apply Strassen's algorithm for a 2x2x2 block multiplication.
        
        Parameters:
        y_indices: list of 4 (i,j) tuples for the 2x2 A-block: [y00, y01, y10, y11]
        z_indices: list of 4 (j,k) tuples for the 2x2 B-block: [z00, z01, z10, z11]
        w_offsets: list of 4 (k,i) tuples for the 2x2 C-block: [w00, w01, w10, w11]
                    where w00 is C[0,0], w01 is C[0,1], w10 is C[1,0], w11 is C[1,1]
        
        Strassen formulas:
        M1 = (y00+y11)*(z00+z11)
        M2 = (y10+y11)*z00
        M3 = y00*(z01-z11)
        M4 = y11*(z10-z00)
        M5 = (y00+y01)*z11
        M6 = (y10-y00)*(z00+z01)
        M7 = (y01-y11)*(z10+z11)
        
        Results:
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        nonlocal col
        
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w01, w10, w11 = w_offsets
        
        # M1 = (y00+y11)*(z00+z11)
        # Contributes to: C00 (+1), C11 (+1)
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1
        
        # M2 = (y10+y11)*z00
        # Contributes to: C10 (+1), C11 (-1)
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] -= 1
        col += 1
        
        # M3 = y00*(z01-z11)
        # Contributes to: C01 (+1), C11 (+1)
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] -= 1
        W[w_idx(w01[0], w01[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1
        
        # M4 = y11*(z10-z00)
        # Contributes to: C00 (+1), C10 (+1)
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] -= 1
        W[w_idx(w00[0], w00[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        col += 1
        
        # M5 = (y00+y01)*z11
        # Contributes to: C00 (-1), C01 (+1)
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] -= 1
        W[w_idx(w01[0], w01[1])][col] += 1
        col += 1
        
        # M6 = (y10-y00)*(z00+z01)
        # Contributes to: C11 (+1)
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y00[0], y00[1])][col] -= 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1
        
        # M7 = (y01-y11)*(z10+z11)
        # Contributes to: C00 (+1)
        U[u_idx(y01[0], y01[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] -= 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] += 1
        col += 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A block: rows 0-1, cols 0-1
    # B block: rows 0-1, cols 0-1
    # C block: rows 0-1, cols 0-1 -> W indices: (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (0,1), (1,0), (1,1)]  # (k,i) pairs: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    strassen_block(y1, z1, w1)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A block: rows 0-1, cols 2-3
    # B block: rows 2-3, cols 0-1
    # C block: rows 0-1, cols 0-1 -> same W indices
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (0,1), (1,0), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A block: rows 0-1, cols 0-1
    # B block: rows 0-1, cols 2-3
    # C block: rows 0-1, cols 2-3 -> W indices: (k=2,i=0), (k=2,i=1), (k=3,i=0), (k=3,i=1)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (2,1), (3,0), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A block: rows 0-1, cols 2-3
    # B block: rows 2-3, cols 2-3
    # C block: rows 0-1, cols 2-3 -> same W indices
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (2,1), (3,0), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # This is 2x2 * 2x1 = 2x1, direct multiplication (rank 4)
    # A block: rows 0-1, cols 0-1
    # B block: rows 0-1, col 4
    # C block: rows 0-1, col 4 -> W indices: (k=4,i=0), (k=4,i=1)
    
    # Direct: C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # Term 1: A[0,0]*B[0,4]
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # Term 2: A[0,1]*B[1,4]
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # Direct: C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    # Term 3: A[1,0]*B[0,4]
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Term 4: A[1,1]*B[1,4]
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    # This is 2x2 * 2x1 = 2x1, direct multiplication (rank 4)
    # A block: rows 0-1, cols 2-3
    # B block: rows 2-3, col 4
    # C block: rows 0-1, col 4 -> W indices: (k=4,i=0), (k=4,i=1)
    
    # Direct: C[0,4] += A[0,2]*B[2,4] + A[0,3]*B[3,4]
    # Term 5: A[0,2]*B[2,4]
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # Term 6: A[0,3]*B[3,4]
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # Direct: C[1,4] += A[1,2]*B[2,4] + A[1,3]*B[3,4]
    # Term 7: A[1,2]*B[2,4]
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Term 8: A[1,3]*B[3,4]
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    assert col == rank, f"Expected {rank} but got {col}"
    
    if time.time() > deadline:
        # If we hit the deadline (unlikely for static construction), return best effort
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
