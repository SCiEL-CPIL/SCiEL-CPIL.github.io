"""Program to find a rank-minimal exact tensor decomposition for 3x3 @ 3x4 matrix multiplication."""

import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a decomposition of rank r
    # We'll start with the known bound and try to go lower
    # Known: rank(3x3x3) <= 23, so rank(3x3x4) <= 23 + 3*3 = 32 (using 3x3x3 for first 3 cols + 3x3x1 for last)
    # Actually, we can do better. Let's try a search.
    
    # The tensor T has shape (n*m, m*p, p*n) = (9, 12, 12)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    
    def make_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T
    
    T = make_tensor()
    
    # Function to verify a decomposition
    def verify(U, V, W, tol=1e-7):
        """Check if sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reconstructed tensor: sum_r U[:,r] V[:,r] W[:,r]
        # Use einsum: 'ir,jr,kr->ijk'
        try:
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.max(np.abs(T_recon - T)) < tol
        except:
            return False
    
    # Function to try to find a decomposition of rank r
    def search_rank(r, max_iters=50, seed=0):
        """
        Try to find a rank-r decomposition using iterative refinement.
        This is a simplified version of the hypernet/alternating minimization.
        """
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Initialize randomly
            U = rng.randn(n*m, r)
            V = rng.randn(m*p, r)
            W = rng.randn(p*n, r)
            
            # Normalize columns
            for c in range(r):
                norm_u = np.linalg.norm(U[:, c])
                norm_v = np.linalg.norm(V[:, c])
                norm_w = np.linalg.norm(W[:, c])
                if norm_u < 1e-10 or norm_v < 1e-10 or norm_w < 1e-10:
                    U[:, c] = rng.randn(n*m)
                    V[:, c] = rng.randn(m*p)
                    W[:, c] = rng.randn(p*n)
                    continue
                scale = (norm_u * norm_v * norm_w) ** (1/3)
                U[:, c] /= scale
                V[:, c] /= scale
                W[:, c] /= scale
            
            # Alternating minimization
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each column i of U, minimize ||T - sum_r U[:,r] V[:,r] W[:,r]||
                # This is a linear problem: T = U @ (V ⊗ W)^T approximately
                # Actually, fix V and W, then T reshaped gives a linear system in U.
                
                # Flatten T to (n*m, m*p * p*n)
                T_flat = T.reshape(n*m, -1)
                
                # Build matrix B where B[:, r] = V[:,r] ⊗ W[:,r]
                # Actually, we want: T_flat @ B = 0 approximately, and U is the solution
                # Let's do: for each r, the contribution is U[:,r] * (V[:,r] ⊗ W[:,r])^T
                # So T_flat ≈ U @ M where M[r, j] = (V[:,r] ⊗ W[:,r])[j]
                
                M = np.zeros((r, m*p * p*n))
                for r_idx in range(r):
                    M[r_idx, :] = (V[:, r_idx] * W[:, r_idx]).ravel()
                
                # Solve U @ M = T_flat
                # This is an underdetermined system if r < m*p*p*n, which it is
                # Use least squares
                try:
                    U, _, _, _ = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                    U = U.T
                except:
                    pass
                
                # Update V
                T_flat_v = T.reshape(m*p, -1)
                M_v = np.zeros((r, n*m * p*n))
                for r_idx in range(r):
                    M_v[r_idx, :] = (U[:, r_idx] * W[:, r_idx]).ravel()
                try:
                    V, _, _, _ = np.linalg.lstsq(M_v.T, T_flat_v.T, rcond=None)
                    V = V.T
                except:
                    pass
                
                # Update W
                T_flat_w = T.reshape(p*n, -1)
                M_w = np.zeros((r, n*m * m*p))
                for r_idx in range(r):
                    M_w[r_idx, :] = (U[:, r_idx] * V[:, r_idx]).ravel()
                try:
                    W, _, _, _ = np.linalg.lstsq(M_w.T, T_flat_w.T, rcond=None)
                    W = W.T
                except:
                    pass
                
                # Check convergence
                if verify(U, V, W, tol=1e-6):
                    # Try to rationalize
                    U_rat = _try_rationalize(U)
                    V_rat = _try_rationalize(V)
                    W_rat = _try_rationalize(W)
                    if U_rat is not None and V_rat is not None and W_rat is not None:
                        if verify(U_rat, V_rat, W_rat, tol=1e-9):
                            return U_rat, V_rat, W_rat
            
            # If not verified with high precision, try to snap to half-integers
            U_snap = _snap_to_half_int(U)
            V_snap = _snap_to_half_int(V)
            W_snap = _snap_to_half_int(W)
            if verify(U_snap, V_snap, W_snap, tol=1e-9):
                return U_snap, V_snap, W_snap
    
    def _try_rationalize(A, max_denom=4):
        """Try to express entries of A as fractions with small denominator."""
        result = []
        for row in A:
            new_row = []
            for x in row:
                found = False
                for denom in range(1, max_denom + 1):
                    num = round(x * denom)
                    if abs(x - num / denom) < 1e-10:
                        new_row.append(Fraction(num, denom))
                        found = True
                        break
                if not found:
                    return None
            result.append(new_row)
        return result
    
    def _snap_to_half_int(A, tol=1e-6):
        """Snap entries to nearest half-integer if close."""
        result = []
        for row in A:
            new_row = []
            for x in row:
                # Check if close to k/2
                k = round(x * 2)
                if abs(x - k / 2.0) < tol:
                    new_row.append(Fraction(k, 2))
                else:
                    # Keep as is (or return None?)
                    new_row.append(Fraction(str(x)))
            result.append(new_row)
        return result
    
    # First, let's try to construct a known good decomposition
    # Rank 36: schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = None
    best_rank = n * m * p  # 36
    
    # Try ranks from lower to upper
    # We know rank 36 works
    # Let's try to find lower ranks
    
    # First, try the construction: 23-rank for 3x3x3 + 3x3x1 for the last column
    # This gives rank 23 + 9 = 32
    # But we need the 23-rank decomposition for 3x3x3
    
    # Let's try random search for ranks 32, 31, 30, 29, 28
    for target_rank in [32, 31, 30, 29, 28]:
        if time.time() > deadline - 10:
            break
        
        result = search_rank(target_rank, max_iters=20, seed=target_rank)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = target_rank
            break
    
    if best_result is None:
        # Fall back to schoolbook
        best_result = schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
