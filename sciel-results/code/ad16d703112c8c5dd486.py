"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: block-partitioned decomposition
    # Split p=8 into two p=4 blocks.
    # For each p=4 block, split m=3 into m=2 and m=1.
    # For m=2, p=4 part, split p=4 into two p=2 blocks.
    # Each p=2 block is 2x2x2, rank 7 (Strassen).
    # So m=2, p=4 part rank is 7 + 7 = 14.
    # m=1, p=4 part is 2x1x4, rank 8 (Schoolbook).
    # Total rank per p=4 block: 14 + 8 = 22.
    # Total rank for p=8: 22 + 22 = 44.
    
    total_rank = 44
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def _add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Strassen 2x2x2 decomposition
        # Local U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global U index: (i_off + li) * m + (j_off + lj)
        # Local V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global V index: (j_off + lj) * p + (k_off + lk)
        # Local W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
        # Global W index: (k_off + lk) * n + (i_off + li)

        # m1: (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m2: (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1

        # m3: a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1

        # m5: (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1

        # m6: (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m7: (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def _add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3
    _add_strassen_block(0, 0, 0)
    _add_strassen_block(0, 0, 2)
    _add_schoolbook_2x1x4(0, 2, 0)

    # Block 2: k in 4..7
    _add_strassen_block(0, 0, 4)
    _add_strassen_block(0, 0, 6)
    _add_schoolbook_2x1x4(0, 2, 4)

    # Verify the decomposition: check that U @ diag @ V @ W^T reconstructs the tensor
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We verify this is exactly the identity for matrix multiplication
    
    # Convert to numpy for fast verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor and check against the canonical matrix multiplication tensor
    # T[i,j,k] should equal 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor for matrix multiplication is T[i,j,k] = 1 (it's the structure tensor)
    # Actually, the tensor M[i,j,k] = 1 for all i,j,k represents the multiplication
    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    
    # Build the structure tensor of matrix multiplication
    # For (n x m) * (m x p) -> (n x p)
    # M[i,j,k] = 1 for all valid i,j,k
    
    # Check a few entries
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(total_rank):
                    val += U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-10:
        # Fallback: pure schoolbook decomposition (rank n*m*p = 48)
        total_rank = n * m * p
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
