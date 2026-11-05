"""Rewritten program using alternating projection with integer/half-integer constraints.
Attempts to find a lower rank decomposition than the schoolbook rank (72).
"""
import numpy as np
import random
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Construct the target tensor T
    # T is a 3D array of shape (nm, mp, pn)
    # T[i*m + j, j'*p + k, k'*n + i'] = 1 if i==i', j==j', k==k' else 0
    # Actually, the non-zero entries are at (i*m+j, j*p+k, k*n+i)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank

    # Helper to check if U, V, W form an exact decomposition
    def check_decomposition(U, V, W):
        # Compute sum_r U_r * V_r * W_r
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Result: (nm, mp, pn)
        # We can do this efficiently by looping over R or using einsum
        # np.einsum('ir, jr, kr -> ijk', U, V, W)
        try:
            reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
            # Check if it matches T
            if np.allclose(reconstructed, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False

    # Helper to project to nearest half-integer
    def project_half_int(M):
        return np.round(M * 2) / 2

    # Helper to project to nearest integer
    def project_int(M):
        return np.round(M)

    # Try to find a decomposition of rank R
    def try_rank(R, num_restarts=10, max_iter=100):
        for restart in range(num_restarts):
            # Initialize randomly
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Normalize to avoid scaling issues
            # U = U / np.linalg.norm(U, axis=0, keepdims=True)
            
            for it in range(max_iter):
                # Update U
                # Fix V, W. Solve for U.
                # We want sum_r U_r * V_r * W_r = T
                # This is a linear system for U.
                # Let A be the matrix where column r is vec(V_r * W_r) (outer product flattened)
                # Actually, it's a Kronecker product structure.
                # vec(U_r * V_r * W_r) = (W_r * V_r^T) * vec(U_r) ? No.
                # Let's use the property: vec(A B C^T) = (C * A^T) vec(B)
                # Here we have U_r (nm x 1), V_r (mp x 1), W_r (pn x 1).
                # The term is U_r V_r^T W_r^T ? No, it's an outer product of 3 vectors.
                # Flatten the 3D tensor T into a vector t of length nm*mp*pn.
                # t = sum_r vec(U_r * V_r * W_r)
                # vec(U_r * V_r * W_r) = (W_r * V_r) * U_r ? 
                # Let's check indices.
                # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Flatten T: T_flat[i*mp*pn + j*pn + k]
                # This is a linear system:
                # sum_r U[i, r] * (V[j, r] * W[k, r]) = T[i, j, k]
                # For fixed i, j, k, this is linear in U[i, r].
                # We can solve for each row of U independently? No, U[i, r] appears in multiple equations.
                # Actually, for a fixed i, the equations involving U[i, :] are:
                # sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k] for all j, k.
                # This is a linear system for U[i, :].
                # Let M_i be a matrix of size (mp*pn) x R.
                # M_i[j*pn + k, r] = V[j, r] * W[k, r]
                # Then M_i * U[i, :] = T[i, :, :] flattened.
                
                U_new = np.zeros((nm, R))
                for i in range(nm):
                    # Construct M_i
                    # V is (mp, R), W is (pn, R)
                    # We need outer product of each column of V and W, then stack?
                    # M_i is (mp*pn, R)
                    # M_i[row, r] = V[row // pn, r] * W[row % pn, r]
                    V_rep = np.repeat(V, pn, axis=0) # (mp*pn, R)
                    W_rep = np.tile(W, (mp, 1))     # (mp*pn, R)
                    M_i = V_rep * W_rep
                    T_i = T[i, :, :].flatten()
                    
                    # Solve M_i * U_i = T_i
                    # Least squares
                    try:
                        U_i, residuals, rank, sv = np.linalg.lstsq(M_i, T_i, rcond=None)
                    except Exception:
                        U_i = U[i, :]
                    
                    # Project to half-integers
                    U_i = project_half_int(U_i)
                    U_new[i, :] = U_i
                
                U = U_new

                # Update V
                V_new = np.zeros((mp, R))
                for j in range(mp):
                    # Equations: sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k] for all i, k
                    # M_j[i*pn + k, r] = U[i, r] * W[k, r]
                    U_rep = np.repeat(U, pn, axis=0) # (nm*pn, R)
                    W_rep = np.tile(W, (nm, 1))     # (nm*pn, R)
                    M_j = U_rep * W_rep
                    T_j = T[:, j, :].flatten() # Note: T is (nm, mp, pn), so T[:, j, :] is (nm, pn)
                    
                    try:
                        V_j, residuals, rank, sv = np.linalg.lstsq(M_j, T_j, rcond=None)
                    except Exception:
                        V_j = V[j, :]
                    
                    V_j = project_half_int(V_j)
                    V_new[j, :] = V_j
                V = V_new

                # Update W
                W_new = np.zeros((pn, R))
                for k in range(pn):
                    # Equations: sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k] for all i, j
                    # M_k[i*mp + j, r] = U[i, r] * V[j, r]
                    U_rep = np.repeat(U, mp, axis=0) # (nm*mp, R)
                    V_rep = np.tile(V, (nm, 1))     # (nm*mp, R)
                    M_k = U_rep * V_rep
                    T_k = T[:, :, k].flatten() # T[:, :, k] is (nm, mp)
                    
                    try:
                        W_k, residuals, rank, sv = np.linalg.lstsq(M_k, T_k, rcond=None)
                    except Exception:
                        W_k = W[k, :]
                    
                    W_k = project_half_int(W_k)
                    W_new[k, :] = W_k
                W = W_new

                # Check convergence
                if check_decomposition(U, V, W):
                    return U, V, W

            # End iteration
            # Check if current best is better
            if check_decomposition(U, V, W):
                return U, V, W

        # End restarts
        return None, None, None

    # Try ranks from schoolbook-1 down to a reasonable limit
    # Given time constraints, we try a few ranks.
    # We start with 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60
    # If we find one, we return it.
    
    # To save time, we limit the number of restarts and iterations for lower ranks.
    ranks_to_try = [71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
    
    for R in ranks_to_try:
        # Adjust restarts based on time? 
        # For now, fixed small number.
        U, V, W = try_rank(R, num_restarts=5, max_iter=20)
        if U is not None:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # If no lower rank found, return schoolbook
    rank = schoolbook_rank
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
