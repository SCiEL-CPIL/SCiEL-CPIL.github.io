"""Matrix multiplication 3x5x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 90."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_fast(U, V, W):
        """Vectorized verification: check tensor decomposition is exact."""
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        # For each (i,j,k): sum over c of U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
        # Use einsum: T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)

    def reduce_rank(U, V, W):
        """Remove linearly dependent columns from the decomposition.
        
        A column c is redundant if there exist coefficients alpha_d (d != c)
        such that for all (i,j,k): U[i,j,c]*V[j,k,c]*W[k,i,c] = sum_d alpha_d * U[i,j,d]*V[j,k,d]*W[k,i,d].
        
        We use a simpler approach: try to find columns that are exact linear
        combinations by checking if the tensor contribution can be replicated.
        """
        rank = len(U[0]) if U else 0
        if rank <= 1:
            return U, V, W
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        kept = list(range(rank))
        # Try to remove each column if it's a linear combination of others
        # We check: can column c be expressed as sum of alpha_d * col_d?
        # This requires: for all (i,j,k), U[i,j,c]*V[j,k,c]*W[k,i,c] = sum_d alpha_d * U[i,j,d]*V[j,k,d]*W[k,i,d]
        # This is a system of n*m*p equations in (rank-1) unknowns.
        
        for c in range(rank):
            if time.time() > deadline:
                break
            if c not in kept:
                continue
            others = [d for d in kept if d != c]
            if not others:
                continue
            
            # Build system: sum_d alpha_d * (U[i,j,d]*V[j,k,d]*W[k,i,d]) = U[i,j,c]*V[j,k,c]*W[k,i,c]
            # Number of equations: n*m*p = 90
            # Number of unknowns: len(others)
            A_mat = np.zeros((n * m * p, len(others)))
            b_vec = np.zeros(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        rhs = U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                        b_vec[idx] = rhs
                        for oi, d in enumerate(others):
                            A_mat[idx, oi] = U_arr[i * m + j, d] * V_arr[j * p + k, d] * W_arr[k * n + i, d]
                        idx += 1
            
            try:
                # Solve least squares
                alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)
                # Check if residual is close to zero
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    kept.remove(c)
            except np.linalg.LinAlgError:
                pass
        
        if len(kept) == rank:
            return U, V, W
        
        # Rebuild with only kept columns
        new_rank = len(kept)
        U_new = [[0] * new_rank for _ in range(n * m)]
        V_new = [[0] * new_rank for _ in range(m * p)]
        W_new = [[0] * new_rank for _ in range(p * n)]
        for ni, old_idx in enumerate(kept):
            for i in range(n * m):
                U_new[i][ni] = U[i][old_idx]
            for j in range(m * p):
                V_new[j][ni] = V[j][old_idx]
            for k in range(p * n):
                W_new[k][ni] = W[k][old_idx]
        
        return U_new, V_new, W_new

    def strassen_2x2x2():
        """Correct Strassen 2x2x2 decomposition (rank 7).
        
        For C = A*B where A, B are 2x2:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1 + M4 - M5 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        
        U encodes A factors, V encodes B factors, W encodes output coefficients.
        U row index: i*2+j for A[i,j]
        V row index: j*2+k for B[j,k]  
        W row index: k*2+i for C[k,i] (trace convention: W[k*n+i])
        """
        # U: (4, 7) - each column c gives the A-factor vector for term c
        # For term c, U[i,j,c] is the coefficient of A[i,j] in the c-th product
        U = [
            # M1: (A00+A11)*(B00+B11)
            [1, 0, 1, 0, 1, 0, 0],  # A00
            [0, 1, 0, 0, 0, 1, -1], # A01
            [0, 1, 0, 0, 0, -1, 1], # A10
            [1, 0, 0, 1, 0, 0, 0],  # A11
        ]
        # V: (4, 7) - each column c gives the B-factor vector for term c
        V = [
            [1, 1, 0, -1, 0, 1, 0],  # B00
            [1, 0, 1, 0, 1, 1, 0],   # B01
            [1, 0, -1, 0, 0, 0, 1],  # B10
            [1, 0, 0, 1, 1, 0, 1],   # B11
        ]
        # W: (4, 7) - W[k*2+i, c] is coefficient for C[k,i]
        # C00 = M1 + M4 - M5 - M7 => W[0*2+0] = W[0]
        # C01 = M3 + M5 => W[0*2+1] = W[1]
        # C10 = M2 + M4 => W[1*2+0] = W[2]
        # C11 = M1 - M2 + M3 + M7 => W[1*2+1] = W[3]
        W = [
            [1, 0, 0, 1],   # C[0,0]: M1 + M4 - M5 - M7
            [0, 1, 0, 0],   # C[0,1]: M3 + M5
            [0, 0, 1, 1],   # C[1,0]: M2 + M4
            [1, 0, 0, 1],   # C[1,1]: M1 - M2 + M3 + M7
        ]
        # Wait, let me recheck. W[k*n+i, c] where n=2 for the block.
        # W index: k*2 + i
        # C[0,0] = M1 + M4 - M5 - M7: coefficients for M1..M7 = [1, 0, 0, 1, -1, 0, -1]
        # C[0,1] = M3 + M5: coefficients = [0, 0, 1, 0, 1, 0, 0]
        # C[1,0] = M2 + M4: coefficients = [0, 1, 0, 1, 0, 0, 0]
        # C[1,1] = M1 - M2 + M3 + M7: coefficients = [1, -1, 1, 0, 0, 0, 1]
        
        W = [
            [1, 0, 0, 1, -1, 0, -1],  # C[0,0]
            [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
            [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
            [1, -1, 1, 0, 0, 0, 1],   # C[1,1]
        ]
        
        return U, V, W, 7

    def schoolbook_block(nb, mb, pb):
        """Schoolbook for nb x mb * mb x pb."""
        rank = nb * mb * pb
        U = [[0] * rank for _ in range(nb * mb)]
        V = [[0] * rank for _ in range(mb * pb)]
        W = [[0] * rank for _ in range(pb * nb)]
        col = 0
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    U[i * mb + j][col] = 1
                    V[j * pb + k][col] = 1
                    W[k * nb + i][col] = 1
                    col += 1
        return U, V, W, rank

    def get_block(nb, mb, pb):
        """Get the best known decomposition for nb x mb * mb x pb."""
        if nb == 2 and mb == 2 and pb == 2:
            return strassen_2x2x2()
        else:
            return schoolbook_block(nb, mb, pb)

    def build_block_decomposition():
        """Build the block decomposition using Strassen for 2x2x2 blocks."""
        if time.time() > deadline:
            return None
        
        # Split A (3x5) into rows [0,1] and [2]
        # Split B (5x6) into cols [0,1], [2,3], [4,5]
        
        blocks = [
            (2, 2, 2, 0, 0, 0),    # A[0:2,:], B[:,0:2]
            (2, 2, 2, 0, 2, 2),    # A[0:2,:], B[:,2:4]
            (2, 1, 2, 0, 4, 4),    # A[0:2,:], B[:,4:6]
            (1, 2, 2, 2, 0, 0),    # A[2:3,:], B[:,0:2]
            (1, 2, 2, 2, 2, 2),    # A[2:3,:], B[:,2:4]
            (1, 1, 2, 2, 4, 4),    # A[2:3,:], B[:,4:6]
        ]
        
        U_rows = n * m  # 15
        V_rows = m * p  # 30
        W_rows = p * n  # 18
        
        # Determine total rank
        total_rank = 0
        block_info = []
        for nb, mb, pb, i_off, j_off, k_off in blocks:
            _, _, _, r = get_block(nb, mb, pb)
            total_rank += r
            block_info.append((nb, mb, pb, i_off, j_off, k_off, r))
        
        U = [[0] * total_rank for _ in range(U_rows)]
        V = [[0] * total_rank for _ in range(V_rows)]
        W = [[0] * total_rank for _ in range(W_rows)]
        
        rank_offset = 0
        for nb, mb, pb, i_off, j_off, k_off, r in block_info:
            U_block, V_block, W_block, _ = get_block(nb, mb, pb)
            
            for i in range(nb):
                for j in range(mb):
                    for c in range(r):
                        U[i_off * m + j_off + j][rank_offset + c] = U_block[i * mb + j][c]
            
            for j in range(mb):
                for k in range(pb):
                    for c in range(r):
                        V[j_off * p + k_off + k][rank_offset + c] = V_block[j * pb + k][c]
            
            for k in range(pb):
                for i in range(nb):
                    for c in range(r):
                        W[k_off * n + i_off + i][rank_offset + c] = W_block[k * nb + i][c]
            
            rank_offset += r
        
        return {"U": U, "V": V, "W": W}

    # Build block decomposition
    result = build_block_decomposition()
    
    if result is not None:
        U, V, W = result["U"], result["V"], result["W"]
        if verify_fast(U, V, W):
            # Try to reduce rank
            U2, V2, W2 = reduce_rank(U, V, W)
            if verify_fast(U2, V2, W2):
                return {"U": U2, "V": V2, "W": W2}
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
