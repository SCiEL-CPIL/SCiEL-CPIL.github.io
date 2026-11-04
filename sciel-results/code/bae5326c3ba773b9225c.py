"""Matrix multiplication 2x4x8 via Strassen block decomposition with dependency reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    # Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # Convert to numpy for SVD
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Stack: each column c has U[:,c], V[:,c], W[:,c]
    # We want to find if sum_c alpha_c * (U_c, V_c, W_c) = 0
    # This is equivalent to finding null space of the stacked matrix
    stacked = np.vstack([U_np, V_np, W_np])  # shape (56, 56)
    
    # Compute SVD
    try:
        U_svd, s, Vt = np.linalg.svd(stacked, full_matrices=False)
        # Rank is number of singular values > tolerance
        tol = 1e-10
        r = int(np.sum(s > tol))
        
        if r < rank:
            # Find null space: last (rank - r) rows of Vt
            null_space = Vt[r:]  # shape (rank-r, rank)
            
            # Use null space to express dependent columns in terms of independent ones
            # We need to find which columns to keep. Use a greedy approach:
            # Try to find a set of r columns that are linearly independent
            # and express the rest in terms of them using the null space.
            
            # First, let's identify which columns are "pivot" columns
            # by looking at the SVD left singular vectors
            U_svd_rank = U_svd[:, :r]  # shape (56, r)
            
            # Find pivot columns: for each null vector, find the column with largest |component|
            # that we can use to eliminate
            keep_cols = []
            eliminate_cols = []
            
            # Greedy: process null vectors, for each find the column to eliminate
            # and express it in terms of kept columns
            # Start by assuming all columns are candidates to keep
            # Process null space vectors one by one
            
            # Better approach: use QR with column pivoting on the stacked matrix
            # to find a good set of independent columns
            Q, R = np.linalg.qr(stacked, mode='complete')
            # Actually, let's use a different strategy:
            # Find r independent columns via rank-revealing QR
            # Then express the remaining (rank-r) columns using the null space
            
            # Use scipy's rank-revealing QR if available, otherwise use SVD-based approach
            # For now, let's try a simpler approach:
            # Identify the r columns that have the largest norms in the "reduced" space
            
            # Compute the projection onto the row space
            # The row space is spanned by the first r rows of Vt
            row_space_basis = Vt[:r]  # shape (r, rank)
            
            # For each column, compute how well it's explained by the row space
            # Actually, we want to find which columns to eliminate.
            # A column j can be eliminated if there's a null vector with a large component at j.
            
            # Let's try: for each null vector, pick the column with max |component| to eliminate
            # and express it in terms of others
            # But we need to be careful about the order
            
            # Alternative: use the fact that if null_space has rows, we can use them
            # to eliminate columns. Let's try to find a triangular system.
            
            # Simple heuristic: sort columns by their norm in the null space
            # and eliminate the ones with largest null space components
            null_norms = np.linalg.norm(null_space, axis=1)  # norm of each null vector
            # For each column, compute max |component| across null vectors
            col_max_comp = np.max(np.abs(null_space), axis=0)  # shape (rank,)
            
            # Columns with large null space components are good candidates to eliminate
            # But we need to ensure the remaining columns are independent
            
            # Let's try a different approach: use the SVD to find a good basis
            # The first r columns of Vt span the row space.
            # We can use these to express dependent columns.
            
            # Find pivot columns using the row space basis
            # A column j is a pivot if its projection onto the row space has a large component
            # in a direction not yet covered
            
            # Simpler: just try to keep the first r columns and verify
            # If that fails, try other combinations
            
            # For now, let's try keeping columns based on which have smallest null space components
            # (i.e., they are most "independent")
            keep_indices = np.argsort(col_max_comp)[:r]
            keep_indices = np.sort(keep_indices)
            
            U_reduced = U_np[:, keep_indices].tolist()
            V_reduced = V_np[:, keep_indices].tolist()
            W_reduced = W_np[:, keep_indices].tolist()
            
            # Verify correctness with a random test
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((p, n))
            for c in range(r):
                C_computed += U_reduced[:, c] * V_reduced[:, c] * W_reduced[:, c].reshape(n, p)
            
            # Check if close
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
            
            # If verification failed, try keeping the first r columns
            U_reduced = U_np[:, :r].tolist()
            V_reduced = V_np[:, :r].tolist()
            W_reduced = W_np[:, :r].tolist()
            
            C_computed = np.zeros((p, n))
            for c in range(r):
                C_computed += U_reduced[:, c] * V_reduced[:, c] * W_reduced[:, c].reshape(n, p)
            
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
            
            # If verification failed, return original
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
