import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 140
    
    # Helper to check if a matrix is close to half-integers
    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        if np.all(np.abs(scaled - rounded) < tol):
            return True
        return False
    
    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    # Strategy: Use structured decompositions based on known results for matrix multiplication.
    # For 2x3 * 3x9, we can exploit the structure.
    # The tensor is T[i*m+j, j*p+k, k*n+i] = 1.
    
    # Let's try to find a decomposition of rank < 54.
    # We can use the fact that this is a bilinear map.
    
    # Approach 1: Try small ranks with random integer V, W and solve for U.
    # But we need to be smarter. Let's use the structure.
    
    # The matrix multiplication tensor for (n,m,p) has a known rank.
    # For (2,3,9), the rank is at most 54.
    # Let's search for ranks from 30 to 53.
    
    # We'll use a more efficient search.
    # Fix V and W to be sparse integer matrices.
    # Solve for U using least squares.
    # Check if U is half-integer.
    
    # To make this faster, we can precompute the matrix M for each (V, W) pair.
    # But V and W are random, so we can't precompute.
    
    # Instead, let's try a different approach:
    # Use the fact that the tensor is symmetric in some sense.
    
    # Let's try ranks from 40 to 53 first, as these are more likely to work.
    # Then if time permits, go lower.
    
    for r in range(40, rank_schoolbook):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
            
        found = False
        # Try multiple random initializations
        # Use small integer entries for V and W
        for trial in range(500):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(12345 + r * 10000 + trial)
            # Random integer matrices with entries in {-1, 0, 1}
            V = np.random.randint(-1, 2, size=(mp, r))
            W = np.random.randint(-1, 2, size=(pn, r))
            
            # Build M where M[b*pn+c, r] = V[b, r] * W[c, r]
            M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            
            # Check if M has full column rank
            if np.linalg.matrix_rank(M) < r:
                continue
                
            # Solve for U using least squares
            # U[a, :] M^T = T[a, :, :].flatten()
            # M @ U[a, :]^T = T[a, :, :].flatten()^T
            try:
                # Solve M @ X = T_flat^T for all a simultaneously
                # T_flat is (nm, mp*pn)
                T_flat = T.reshape(nm, mp * pn)
                # We want U such that U @ M^T = T_flat
                # This is equivalent to M @ U^T = T_flat^T
                # So U^T = M^{-1} T_flat^T (in least squares sense)
                # U = (M^{-1} T_flat^T)^T = T_flat (M^{-1})^T
                # Using least squares: U^T = lstsq(M, T_flat^T)
                U_transpose, residuals, rank, sv = np.linalg.lstsq(M, T_flat.T, rcond=None)
                U = U_transpose.T
            except:
                continue
                
            # Check if U is half-integer
            if is_half_integer(U, tol=1e-5):
                # Snap to nearest half-integer
                U_snapped = np.round(2.0 * U) / 2.0
                # Verify exactness with tighter tolerance
                if verify_solution(U_snapped, V, W, T, tol=1e-8):
                    best_rank = r
                    best_U, best_V, best_W = U_snapped, V, W
                    found = True
                    break
                    
        if found:
            # If we found a solution at rank r, try to go lower
            # But break out of the current r loop and continue with lower r
            # Actually, since we're iterating r from 40 upwards, if we find at r=40,
            # we should try r=39, etc. But our loop goes up. Let's restructure.
            # For now, break and return, or continue searching lower.
            # Since we want the minimum, let's continue searching lower ranks.
            # But the loop is going up. Let's fix this by trying lower ranks after.
            pass
    
    # Now try lower ranks from 30 to best_rank-1
    for r in range(30, best_rank):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        found = False
        for trial in range(300):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(99999 + r * 7777 + trial)
            V = np.random.randint(-1, 2, size=(mp, r))
            W = np.random.randint(-1, 2, size=(pn, r))
            
            M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            
            if np.linalg.matrix_rank(M) < r:
                continue
                
            try:
                T_flat = T.reshape(nm, mp * pn)
                U_transpose, residuals, rank, sv = np.linalg.lstsq(M, T_flat.T, rcond=None)
                U = U_transpose.T
            except:
                continue
                
            if is_half_integer(U, tol=1e-5):
                U_snapped = np.round(2.0 * U) / 2.0
                if verify_solution(U_snapped, V, W, T, tol=1e-8):
                    best_rank = r
                    best_U, best_V, best_W = U_snapped, V, W
                    found = True
                    break
    
    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
