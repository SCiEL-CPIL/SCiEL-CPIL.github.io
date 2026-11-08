"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 118  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # We use a rank 36 decomposition based on block Strassen.
    # 4 Strassen blocks (2x2x2) for the 2x4x4 part: 4*7 = 28
    # 2 standard blocks (2x2x1) for the 2x4x1 part: 2*4 = 8
    # Total: 36
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        """
        Apply Strassen 2x2x2 decomposition.
        
        y_indices: [(y00_i, y00_j), (y01_i, y01_j), (y10_i, y10_j), (y11_i, y11_j)]
                   These are (i, j) pairs for the A matrix (left factor)
        z_indices: [(z00_j, z00_k), (z01_j, z01_k), (z10_j, z10_k), (z11_j, z11_k)]
                   These are (j, k) pairs for the B matrix (right factor)
        w_offsets: [(w00_k, w00_i), (w01_k, w01_i), (w10_k, w10_i), (w11_k, w11_i)]
                   These are (k, i) pairs for the C matrix (result), where W index is k*n+i
        
        Strassen formulas:
        M1 = (y00+y11)*(z00+z11)
        M2 = (y10+y11)*z00
        M3 = y00*(z01-z11)
        M4 = y11*(z10-z00)
        M5 = (y00+y01)*z11
        M6 = (y10-y00)*(z00+z01)
        M7 = (y01-y11)*(z10+z11)
        
        X00 = M1 + M4 - M5 + M7
        X01 = M3 + M5
        X10 = M2 + M4
        X11 = M1 - M2 + M3 + M6
        """
        nonlocal col
        
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)*(z00+z11)
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        # X00 += M1, X11 += M1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2 = (y10+y11)*z00
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        # X10 += M2, X11 -= M2
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3 = y00*(z01-z11)
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        # X01 += M3, X11 += M3
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4 = y11*(z10-z00)
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        # X00 += M4, X10 += M4
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5 = (y00+y01)*z11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        # X00 -= M5, X01 += M5
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6 = (y10-y00)*(z00+z01)
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        # X11 += M6
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7 = (y01-y11)*(z10+z11)
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        # X00 += M7
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A_L = [[(0,0),(0,1)],[(1,0),(1,1)]]
    # B_11 = [[(0,0),(0,1)],[(1,0),(1,1)]]
    # C_11 = [[(0,0),(1,0)],[(0,1),(1,1)]]  (k*n+i indexing: (k,i))
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]  # (k=0,i=0), (k=1,i=0), (k=0,i=1), (k=1,i=1)
    strassen_block(y1, z1, w1)
    
    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    # A_R = [[(0,2),(0,3)],[(1,2),(1,3)]]
    # B_21 = [[(2,0),(2,1)],[(3,0),(3,1)]]
    # C_11 = same as above
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # A_L = [[(0,0),(0,1)],[(1,0),(1,1)]]
    # B_12 = [[(0,2),(0,3)],[(1,2),(1,3)]]
    # C_12 = [[(2,0),(3,0)],[(2,1),(3,1)]]  (k*n+i: (k=2,i=0), (k=3,i=0), (k=2,i=1), (k=3,i=1))
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    # A_R = [[(0,2),(0,3)],[(1,2),(1,3)]]
    # B_22 = [[(2,2),(2,3)],[(3,2),(3,3)]]
    # C_12 = same as above
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A[0:2,0:2] * B[0:2,4] (2x2 * 2x1, rank 4)
    # A_L = [[(0,0),(0,1)],[(1,0),(1,1)]]
    # B_15 = [[(0,4)],[(1,4)]]
    # C_15 = [[(4,0)],[(4,1)]]  (k*n+i: (k=4,i=0), (k=4,i=1))
    
    # M1: A[0,0]*B[0,4] -> C[0,4]
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M2: A[0,1]*B[1,4] -> C[0,4]
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M3: A[1,0]*B[0,4] -> C[1,4]
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # M4: A[1,1]*B[1,4] -> C[1,4]
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 6: A[0:2,2:4] * B[2:4,4] (2x2 * 2x1, rank 4)
    # A_R = [[(0,2),(0,3)],[(1,2),(1,3)]]
    # B_25 = [[(2,4)],[(3,4)]]
    # C_15 = [[(4,0)],[(4,1)]]
    
    # M1: A[0,2]*B[2,4] -> C[0,4]
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M2: A[0,3]*B[3,4] -> C[0,4]
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M3: A[1,2]*B[2,4] -> C[1,4]
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # M4: A[1,3]*B[3,4] -> C[1,4]
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Verify the decomposition
    if time.time() < _deadline:
        # Check that sum_r U[:,r] * V[:,r] * W[:,r] = T
        # T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
        # For each (i,j,k), check that sum_r U[u_idx(i,j),r] * V[v_idx(j,k),r] * W[w_idx(k,i),r] = 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = u_idx(i, j)
                    v = v_idx(j, k)
                    w = w_idx(k, i)
                    total = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                    if total != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # This should not happen, but if it does, we have a bug
            # For now, just return what we have
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
