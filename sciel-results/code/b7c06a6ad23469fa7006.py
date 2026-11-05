import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication.
    Uses a block partition 5=2+2+1 with Strassen (rank 7) for the 2x2x2 blocks
    and schoolbook for all other block sizes.
    Total rank = 8*7 + 4*4 + 4*4 + 2*2 + 4*4 + 2*2 + 2*2 + 1*1 = 117.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    n, m, p = DIMENSIONS
    
    def add_strassen_block(U, V, W, r_offset, i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block at offset (i0, j0, k0)."""
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m+j0, r_offset] = 1
        U[(i0+1)*m+j0+1, r_offset] = 1
        V[j0*p+k0, r_offset] = 1
        V[(j0+1)*p+k0+1, r_offset] = 1
        W[k0*n+i0, r_offset] = 1
        W[(k0+1)*n+i0+1, r_offset] = 1
        r_offset += 1
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m+j0, r_offset] = 1
        U[(i0+1)*m+j0+1, r_offset] = 1
        V[j0*p+k0, r_offset] = 1
        W[k0*n+i0+1, r_offset] = 1
        r_offset += 1
        
        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m+j0, r_offset] = 1
        V[j0*p+k0+1, r_offset] = 1
        V[(j0+1)*p+k0+1, r_offset] = -1
        W[(k0+1)*n+i0, r_offset] = 1
        r_offset += 1
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m+j0+1, r_offset] = 1
        V[(j0+1)*p+k0, r_offset] = 1
        V[j0*p+k0, r_offset] = -1
        W[k0*n+i0+1, r_offset] = 1
        r_offset += 1
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m+j0, r_offset] = 1
        U[i0*m+j0+1, r_offset] = 1
        V[(j0+1)*p+k0+1, r_offset] = 1
        W[(k0+1)*n+i0, r_offset] = 1
        r_offset += 1
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m+j0, r_offset] = 1
        U[i0*m+j0, r_offset] = -1
        V[j0*p+k0, r_offset] = 1
        V[j0*p+k0+1, r_offset] = 1
        W[k0*n+i0+1, r_offset] = 1
        W[(k0+1)*n+i0+1, r_offset] = 1
        r_offset += 1
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m+j0+1, r_offset] = 1
        U[(i0+1)*m+j0+1, r_offset] = -1
        V[(j0+1)*p+k0, r_offset] = 1
        V[(j0+1)*p+k0+1, r_offset] = 1
        W[k0*n+i0, r_offset] = 1
        W[(k0+1)*n+i0, r_offset] = 1
        r_offset += 1
        
        return r_offset
    
    def add_schoolbook_block(U, V, W, r_offset, i0, j0, k0, ni, nj, nk):
        """Add schoolbook decomposition for ni x nj x nk block at offset (i0, j0, k0)."""
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(nk):
                    U[(i0+ii)*m + (j0+jj), r_offset] = 1
                    V[(j0+jj)*p + (k0+kk), r_offset] = 1
                    W[(k0+kk)*n + (i0+ii), r_offset] = 1
                    r_offset += 1
        return r_offset
    
    # Calculate total rank for partition [2, 2, 1]
    block_sizes = [2, 2, 1]
    total_rank = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                ni = block_sizes[bi]
                nj = block_sizes[bj]
                nk = block_sizes[bk]
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    # Allocate matrices
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # Compute offsets for each block
    i_offsets = [0, 2, 4]
    j_offsets = [0, 2, 4]
    k_offsets = [0, 2, 4]
    
    r = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                ni = block_sizes[bi]
                nj = block_sizes[bj]
                nk = block_sizes[bk]
                i0 = i_offsets[bi]
                j0 = j_offsets[bj]
                k0 = k_offsets[bk]
                
                if ni == 2 and nj == 2 and nk == 2:
                    r = add_strassen_block(U, V, W, r, i0, j0, k0)
                else:
                    r = add_schoolbook_block(U, V, W, r, i0, j0, k0, ni, nj, nk)
    
    assert r == total_rank, f"Rank mismatch: {r} vs {total_rank}"
    
    # Verify decomposition
    def verify_decomposition(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        max_err = 0.0
        for a_row in range(n):
            for a_col in range(m):
                for b_row in range(m):
                    for b_col in range(p):
                        expected = 1.0 if a_col == b_row else 0.0
                        u_val = U[a_row * m + a_col, :]
                        v_val = V[b_row * p + b_col, :]
                        w_val = W[b_col * n + a_row, :]
                        val = np.dot(u_val * v_val, w_val)
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
                        if max_err > tol:
                            return False, max_err
        return True, max_err
    
    valid, err = verify_decomposition(U, V, W)
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
