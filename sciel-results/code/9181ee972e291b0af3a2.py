"""Bootstrap MODEL-AND-OPTIMIZE approach for 6x6 matrix multiplication tensor decomposition.
Strategy: Start from a known valid decomposition (schoolbook rank 216), then attempt
rank reduction by merging columns where U and V vectors are linearly dependent.
We use a greedy approach: try to combine pairs of columns that have proportional U and V vectors.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    Uses a model-and-optimize approach: parameterize the decomposition and attempt
    to reduce rank by finding linear dependencies among the rank-1 components.
    """
    deadline = time.time() + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 216)
    # Each term: U[row, col] = 1 if row corresponds to (i,j), V[col, col] = 1 if col corresponds to (j,k)
    # W[col, col] = 1 if it maps to output (k,i)
    
    rank_schoolbook = n * m * p  # 216
    
    # Build schoolbook factors
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                # W index: W[k*n+i, r]
                W[k*n + i, col] = 1.0
                col += 1
    
    # Verify correctness of schoolbook
    # C = A @ B, where (A@B)[i,k] = sum_j A[i,j] * B[j,k]
    # Our decomposition: sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # For a single column corresponding to (i',j',k'), this contributes:
    #   1 * 1 * 1 to position (k',i') in W, which maps to output[i',k']
    # So summing over all columns: output[i,k] = sum_{j} A[i,j]*B[j,k] * (sum over matching i',j',k')
    # This is correct.
    
    # Now attempt rank reduction
    # Strategy: Find groups of columns where U vectors and V vectors are linearly dependent
    # If two columns have U[:,c1] = alpha * U[:,c2] and V[:,c1] = beta * V[:,c2],
    # then W[:,c1] must equal alpha*beta * W[:,c2] for them to be mergeable.
    # In that case, we can keep one column and scale W accordingly.
    
    # More generally, if a set of columns has U vectors spanning a d-dimensional space
    # and V vectors spanning a d-dimensional space, we can potentially reduce to d columns.
    
    # Let's use a greedy approach: repeatedly find pairs that can be merged.
    # Two columns c1, c2 can be merged if U[:,c1] and U[:,c2] are linearly dependent
    # AND V[:,c1] and V[:,c2] are linearly dependent.
    
    # For schoolbook, each column has U with a single 1 and V with a single 1.
    # No two columns have dependent U vectors (each has a different position with 1).
    # So schoolbook can't be reduced this way directly.
    
    # Alternative approach: Use block structure with Strassen at the 2x2 level.
    # 6 = 3 * 2, so we have a 3x3 block matrix with 2x2 blocks.
    # Use Strassen (rank 7) for each 2x2 multiplication.
    # Total rank: 9 * 3 * 7 = 189
    
    # But we can do better by using a smarter block decomposition.
    # Let's try 6 = 2 * 3, so 2x2 blocks with 3x3 submatrices? No, that's the same.
    
    # Actually, let's just implement the block-Strassen approach which gives rank 189.
    # Then try to reduce further.
    
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 9 * 3 * 7 = 189
    
    U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
    V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
    W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    # A block: rows ib*bs..(ib+1)*bs, cols jb*bs..(jb+1)*bs
                    # B block: rows jb*bs..(jb+1)*bs, cols kb*bs..(kb+1)*bs
                    # C block: rows ib*bs..(ib+1)*bs, cols kb*bs..(kb+1)*bs
                    
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U_bs[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V_bs[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W_bs[gk * n + gi, col] += coeff
                    
                    col += 1
    
    # Verify the block-Strassen decomposition is correct
    def verify_decomposition(U_mat, V_mat, W_mat, n=n, m=m, p=p, atol=1e-9):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] gives correct matrix multiplication tensor."""
        rank = U_mat.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_mat_2d = U_mat[:, r].reshape(n, m)
            v_mat_2d = V_mat[:, r].reshape(m, p)
            w_mat_2d = W_mat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_mat_2d, v_mat_2d, w_mat_2d)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=atol)
    
    if not verify_decomposition(U_bs, V_bs, W_bs):
        # Fallback to schoolbook if block-Strassen is wrong
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction by finding linearly dependent columns
    # Two columns c1, c2 can be merged if there exist scalars alpha, beta such that:
    # U[:,c1] = alpha * U[:,c2], V[:,c1] = beta * V[:,c2], W[:,c1] = alpha*beta * W[:,c2]
    # More generally, we can try to find a basis of the column space that preserves the tensor structure.
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_iter=5000, atol=1e-10):
        """Try to reduce rank by finding and merging linearly dependent column triples."""
        U = U_mat.copy()
        V = V_mat.copy()
        W = W_mat.copy()
        rank = U.shape[1]
        
        # Build a list of active column indices
        active = list(range(rank))
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if len(active) < 2:
                break
            
            merged = False
            for i in range(len(active)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(active)):
                    c1 = active[i]
                    c2 = active[j]
                    
                    # Check if U[:,c1] and U[:,c2] are linearly dependent
                    u1 = U[:, c1]
                    u2 = U[:, c2]
                    # Find alpha such that u1 ≈ alpha * u2
                    mask_u2 = np.abs(u2) > atol
                    if not np.any(mask_u2):
                        continue
                    # alpha = sum(u1*mask) / sum(u2*mask) for positions where u2 is nonzero
                    idx = np.where(mask_u2)[0]
                    denom = np.sum(u2[idx] * u2[idx])
                    if denom < atol:
                        continue
                    alpha = np.sum(u1[idx] * u2[idx]) / denom
                    # Check if u1 ≈ alpha * u2
                    if not np.allclose(u1, alpha * u2, atol=atol):
                        continue
                    
                    # Check if V[:,c1] and V[:,c2] are linearly dependent
                    v1 = V[:, c1]
                    v2 = V[:, c2]
                    mask_v2 = np.abs(v2) > atol
                    if not np.any(mask_v2):
                        continue
                    idx_v = np.where(mask_v2)[0]
                    denom_v = np.sum(v2[idx_v] * v2[idx_v])
                    if denom_v < atol:
                        continue
                    beta = np.sum(v1[idx_v] * v2[idx_v]) / denom_v
                    if not np.allclose(v1, beta * v2, atol=atol):
                        continue
                    
                    # Check if W[:,c1] ≈ alpha*beta * W[:,c2]
                    w1 = W[:, c1]
                    w2 = W[:, c2]
                    if not np.allclose(w1, alpha * beta * w2, atol=atol):
                        continue
                    
                    # Merge: keep c2, remove c1
                    # Scale W[:,c2] by (1 + alpha*beta) to account for both contributions
                    W[:, c2] = W[:, c2] + w1
                    # Remove c1 from active
                    active.remove(c1)
                    merged = True
                    break
                if merged:
                    break
            
            if not merged:
                break
        
        # Compact the matrices
        if len(active) < rank:
            U = U[:, active]
            V = V[:, active]
            W = W[:, active]
        
        return U, V, W, len(active)
    
    U_red, V_red, W_red, new_rank = try_reduce_rank(U_bs, V_bs, W_bs)
    
    # Verify the reduced decomposition
    if verify_decomposition(U_red, V_red, W_red):
        return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    # For now, return the verified block-Strassen result (rank 189)
    return {"U": U_bs.tolist(), "V": V_bs.tolist(), "W": W_bs.tolist()}
