import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 110
    
    # Build the rank-50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init
    
    # Attempt rank reduction
    # Stack U, V, W columns into a single matrix M of shape (n*m + m*p + p*n, rank)
    M = np.vstack([U.T, V.T, W.T]) # Shape: (rank, total_dim)
    
    # Find null space or linear dependencies
    # If rank(M) < rank_init, we can reduce.
    # Compute SVD
    try:
        U_svd, s, Vh = np.linalg.svd(M.T, full_matrices=False)
        # s contains singular values.
        # If some s are close to 0, there are dependencies.
        tol = 1e-9
        rank_m = np.sum(s > tol)
        
        if rank_m < rank_init:
            # The null space of M.T is spanned by the last (rank_init - rank_m) rows of Vh
            # These vectors represent linear combinations of columns that sum to 0.
            # We can eliminate columns corresponding to these dependencies.
            # For simplicity, if rank_m == rank_init - 1, we can eliminate 1 column.
            # The dependency vector v in null space means sum(v_i * col_i) = 0.
            # We can remove column j where |v_j| is max, and express it in terms of others.
            
            # Get null space basis
            null_space = Vh[rank_m:].T # Shape: (rank_init, rank_init - rank_m)
            
            # We want to find a set of columns to remove.
            # Greedy: find a column that is in the span of the others.
            # This is equivalent to finding a vector in the null space with a large component.
            
            # Let's just try to remove columns one by one if possible.
            # For each column c, check if M[:, c] is in span of M[:, other].
            # This is O(R^2 * D) which is fine for R=50.
            
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            curr_rank = rank_init
            
            for _ in range(10): # Try to reduce up to 10 times
                if time.time() > deadline - 10:
                    break
                reduced = False
                for c in range(curr_rank):
                    # Check if column c is in span of others
                    other_cols = [i for i in range(curr_rank) if i != c]
                    if not other_cols:
                        continue
                    M_sub = M.T[:, other_cols] # (total_dim, curr_rank-1)
                    target = M.T[:, c]
                    
                    # Solve M_sub @ x = target
                    try:
                        coeffs, residuals, rank_s, sv = np.linalg.lstsq(M_sub, target, rcond=None)
                        recon = M_sub @ coeffs
                        if np.allclose(recon, target, atol=1e-9):
                            # Eliminate column c
                            # Shift columns
                            U_curr = np.delete(U_curr, c, axis=1)
                            V_curr = np.delete(V_curr, c, axis=1)
                            W_curr = np.delete(W_curr, c, axis=1)
                            M = np.vstack([U_curr.T, V_curr.T, W_curr.T])
                            curr_rank -= 1
                            reduced = True
                            break
                    except:
                        continue
                if not reduced:
                    break
                    
            if curr_rank < rank_init:
                return {"U": U_curr.tolist(), "V": V_curr.tolist(), "W": W_curr.tolist()}
    except:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
