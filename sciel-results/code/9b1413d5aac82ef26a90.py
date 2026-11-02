"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130.0
    SAFETY_MARGIN = 20.0
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact.
        
        The tensor equation for matrix multiplication C = A @ B where
        A is n x m and B is m x p:
        C[i, k] = sum_j A[i, j] * B[j, k]
        
        The decomposition requires:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
        
        More precisely, for each (i, j, k), the sum over r of
        U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if (i,j,k) is a valid
        matmul index, but actually the tensor T[i,j,k] = 1 for all valid indices, and
        the decomposition must satisfy:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in range(n), j in range(m), k in range(p)
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Build the full tensor contraction
        # For each (i, j, k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 for all valid (i, j, k)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_with_numpy(U, V, W, rank):
        """Faster verification using numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Build matrices for efficient verification
        # For each r, we have vectors u_r (n*m), v_r (m*p), w_r (p*n)
        # The condition is: for all (i,j,k), sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
        
        # Reshape to 3D: U -> (rank, n, m), V -> (rank, m, p), W -> (rank, p, n)
        U3 = U.reshape(rank, n, m)
        V3 = V.reshape(rank, m, p)
        W3 = W.reshape(rank, p, n)
        
        # For each (i, j, k), compute sum over r
        # T[i, j, k] = sum_r U3[r, i, j] * V3[r, j, k] * W3[r, k, i]
        
        # Use einsum or manual loop
        T = np.zeros((n, m, p))
        for r in range(rank):
            # U3[r] is (n, m), V3[r] is (m, p), W3[r] is (p, n)
            # We need outer product structure
            # For fixed r: contribution to T[i,j,k] = U3[r,i,j] * V3[r,j,k] * W3[r,k,i]
            # This is like: for each i, j, k: U[i,j] * V[j,k] * W[k,i]
            # We can compute this as:
            # temp[i,j,k] = U[i,j] * V[j,k] * W[k,i]
            # Using broadcasting:
            # U3[r][:, :, None] * V3[r][None, :, :] gives (n, m, p) but V3[r] is (m, p)
            # Actually: U[i,j] * V[j,k] * W[k,i]
            # = (U3[r][:, :, None] * V3[r][None, :, :]) then multiply by W3[r] appropriately
            # W3[r] is (p, n), we need W[k, i] for each (i, k)
            # So W3[r].T is (n, p), and we need W3[r].T[i, k] = W[k, i]
            
            # Let's do it step by step:
            # For each i: U[i, :, None] * (V * W3.T[i, :]) where V is (m, p) and W3.T[i, :] is (p,)
            # Actually simpler: use einsum
            T += np.einsum('ij,jk,ki->ijk', U3[r], V3[r], W3[r])
        
        return np.allclose(T, 1.0, atol=1e-9)

    def build_strassen_decomposition():
        """Build the known rank-55 Strassen decomposition.
        
        Split m=3 into 2+1. Use Strassen on the 2x2 block (rows 0,1 of A and cols 0,1 of B)
        for each block of 2 columns of B (5 blocks), and naive for the remaining column (col 2 of B).
        
        Wait, that's not quite right. Let me think again.
        
        Actually, the standard approach:
        - A is 2x3, B is 3x10
        - Split B's rows into 2+1: B_top (2x10) and B_bottom (1x10)
        - Split A's cols into 2+1: A_left (2x2) and A_right (2x1)
        - C = A_left @ B_top + A_right @ B_bottom
        
        For A_left (2x2) @ B_top (2x10): use Strassen per 2-column block of B_top
          Each 2x2 @ 2x2 block uses 7 multiplications (Strassen)
          There are 5 such blocks (10 columns / 2)
          Total: 35 multiplications
        
        For A_right (2x1) @ B_bottom (1x10): naive
          Each element: 2*10 = 20 multiplications (but actually it's just outer product)
          A_right[i,0] * B_bottom[0,k] for each i,k: 2*10 = 20 terms, but each is one "multiplication"
          Actually, in tensor decomposition, each term A[i,j]*B[j,k] contributes to C[i,k]
          For j=2 (the third row), we have 2*10 = 20 terms
          Each term needs one rank-1 decomposition
          Total: 20 multiplications
        
        Grand total: 35 + 20 = 55
        
        Let's build this:
        """
        rank = 55
        U = [[0] * rank for _ in range(n * m)]  # (2*3, 55) = (6, 55)
        V = [[0] * rank for _ in range(m * p)]  # (3*10, 55) = (30, 55)
        W = [[0] * rank for _ in range(p * n)]  # (10*2, 55) = (20, 55)
        
        col = 0
        
        # Part 1: Strassen for A[:, 0:2] @ B[0:2, :]
        # A_top is 2x2 (rows 0,1 of A, cols 0,1), B_top is 2x10 (rows 0,1 of B, all cols)
        # Split B_top into 5 blocks of 2 columns each
        for block in range(5):
            k0 = block * 2      # first column of this block
            k1 = block * 2 + 1  # second column of this block
            
            # A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
            # B00 = B[0,k0], B01 = B[0,k1], B10 = B[1,k0], B11 = B[1,k1]
            
            # M1 = (A00 + A11) * (B00 + B11)
            U[0*3 + 0][col] += 1  # A[0,0]
            U[1*3 + 1][col] += 1  # A[1,1]
            V[0*10 + k0][col] += 1   # B[0,k0]
            V[1*10 + k1][col] += 1   # B[1,k1]
            W[k0*2 + 0][col] += 1    # C[0,k0] += M1
            W[k1*2 + 1][col] += 1    # C[1,k1] += M1
            col += 1
            
            # M2 = (A10 + A11) * B00
            U[1*3 + 0][col] += 1  # A[1,0]
            U[1*3 + 1][col] += 1  # A[1,1]
            V[0*10 + k0][col] += 1   # B[0,k0]
            W[k0*2 + 1][col] += 1    # C[1,k0] += M2
            W[k1*2 + 1][col] += -1   # C[1,k1] -= M2
            col += 1
            
            # M3 = A00 * (B01 - B11)
            U[0*3 + 0][col] += 1  # A[0,0]
            V[0*10 + k1][col] += 1   # B[0,k1]
            V[1*10 + k1][col] += -1  # -B[1,k1]
            W[k1*2 + 0][col] += 1    # C[0,k1] += M3
            W[k1*2 + 1][col] += 1    # C[1,k1] += M3
            col += 1
            
            # M4 = A11 * (B10 - B00)
            U[1*3 + 1][col] += 1  # A[1,1]
            V[1*10 + k0][col] += 1   # B[1,k0]
            V[0*10 + k0][col] += -1  # -B[0,k0]
            W[k0*2 + 0][col] += 1    # C[0,k0] += M4
            W[k0*2 + 1][col] += 1    # C[1,k0] += M4
            col += 1
            
            # M5 = (A00 + A01) * B11
            U[0*3 + 0][col] += 1  # A[0,0]
            U[0*3 + 1][col] += 1  # A[0,1]
            V[1*10 + k1][col] += 1   # B[1,k1]
            W[k0*2 + 0][col] += -1   # C[0,k0] -= M5
            W[k1*2 + 0][col] += 1    # C[0,k1] += M5
            col += 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            U[1*3 + 0][col] += 1   # A[1,0]
            U[0*3 + 0][col] += -1  # -A[0,0]
            V[0*10 + k0][col] += 1   # B[0,k0]
            V[0*10 + k1][col] += 1   # B[0,k1]
            W[k1*2 + 1][col] += 1    # C[1,k1] += M6
            col += 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            U[0*3 + 1][col] += 1   # A[0,1]
            U[1*3 + 1][col] += -1  # -A[1,1]
            V[1*10 + k0][col] += 1   # B[1,k0]
            V[1*10 + k1][col] += 1   # B[1,k1]
            W[k0*2 + 0][col] += 1    # C[0,k0] += M7
            col += 1
        
        # Part 2: Naive for A[:, 2] @ B[2, :]
        # A_right is 2x1 (col 2 of A), B_bottom is 1x10 (row 2 of B)
        # C[i,k] += A[i,2] * B[2,k]
        for i in range(n):
            for k in range(p):
                U[i*3 + 2][col] += 1   # A[i,2]
                V[2*10 + k][col] += 1  # B[2,k]
                W[k*2 + i][col] += 1   # C[i,k] += term
                col += 1
        
        assert col == rank, f"Column count {col} != rank {rank}"
        
        # Verify
        if not verify_with_numpy(U, V, W, rank):
            raise ValueError("Strassen decomposition verification failed")
        
        return U, V, W, rank

    # Build the rank-55 decomposition
    U, V, W, best_rank = build_strassen_decomposition()
    
    # Try to improve by finding a lower-rank decomposition
    # Strategy: local search to merge or eliminate columns
    
    def try_local_search(U, V, W, rank, max_iters=1000):
        """Try to reduce rank by local search.
        
        Attempt to find a rank-(rank-1) decomposition by:
        1. For each pair of columns (r1, r2), try to express one as a linear combination
           of the other plus a new column that maintains the tensor equation.
        
        This is complex, so instead we try:
        - For each column r, check if it can be removed (i.e., if the remaining
          rank-1 decomposition still satisfies the tensor equation).
        
        More practically, we try small perturbations and see if we can find
        a valid decomposition with fewer effective terms.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
        start_ls = time.time()
        
        for it in range(max_iters):
            if time.time() - start_ls > DEADLINE - start_time - SAFETY_MARGIN:
                break
            
            # Try removing each column
            for r in range(best_rank):
                # Skip column r and see if the remaining columns still work
                # This is only valid if column r is "redundant" in some sense
                # For now, just check if any column has all zeros (shouldn't happen)
                if np.allclose(U[r], 0) and np.allclose(V[r], 0) and np.allclose(W[r], 0):
                    # Remove this column
                    U = np.delete(U, r, axis=0)
                    V = np.delete(V, r, axis=0)
                    W = np.delete(W, r, axis=0)
                    best_rank -= 1
                    break
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
    
    # Try local search
    if time.time() < DEADLINE - SAFETY_MARGIN:
        U_ls, V_ls, W_ls, rank_ls = try_local_search(U, V, W, best_rank)
        if rank_ls < best_rank:
            U, V, W, best_rank = U_ls, V_ls, W_ls, rank_ls
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
