"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # The matrix multiplication tensor T of size (n*m) x (m*p) x (p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for the structural tensor
    # We want to find minimum rank r such that T = sum_c U[:,c] * V[:,c] * W[:,c]
    
    # Start with schoolbook rank 72 and try to reduce
    # Use numerical tensor decomposition with rational reconstruction
    
    def build_tensor():
        """Build the matrix multiplication tensor as a 3D array."""
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_tensor()
    
    # Try various ranks using numerical SVD-based tensor decomposition
    # For tensor T with dimensions I x J x K, we can use:
    # - Flatten to matrix and find rank
    # - Use iterative methods
    
    # First, let's establish the schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Use the approach: for a given rank r, set up the system
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    # This is a system of equations. We can use:
    # 1. Random initialization + iterative refinement
    # 2. Or use the fact that for small ranks, we can try exact methods
    
    # Let's try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) / something... actually the tensor rank 
    # is at least max(nm, mp, pn) in general? No, that's not right.
    # The rank is at least the rank of any matricization.
    
    # Matricization 1: (nm) x (mp*p*n) -> rank >= ?
    # Actually, for the multiplication tensor, known lower bounds exist.
    
    # Let's try a practical approach: use numerical tensor train or 
    # iterative rank reduction
    
    # For this specific case (3,3,8), let's try ranks from 50 to 71
    # and see if we can find valid decompositions
    
    def try_rank_decomposition(r, max_iter=5000, tol=1e-10):
        """Try to find rank-r decomposition using iterative method."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize to avoid scaling issues
        for c in range(r):
            norm = np.sqrt(np.sum(U[:,c]**2) * np.sum(V[:,c]**2) * np.sum(W[:,c]**2))
            if norm > 0:
                U[:,c] /= norm**(1/3)
                V[:,c] /= norm**(1/3)
                W[:,c] /= norm**(1/3)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute current tensor estimate
            # T_est[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
            # Use efficient computation via einsum
            T_est = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.linalg.norm(T_est - T)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
            
            # Update each factor
            # For U: T_est[i,j,k] = U[i,c]*V[j,c]*W[k,c]
            # Fix V and W, update U
            # T[i,j,k] ≈ sum_c U[i,c] * (V[j,c]*W[k,c])
            # This is like a matrix completion problem
            
            # Update U
            for c in range(r):
                # T[:, :, :] = sum_c U[:,c] * (V[:,c] * W[:,c])
                # For fixed c, the contribution is U[:,c] * outer(V[:,c], W[:,c])
                # Residual after removing c-th term
                T_res = T - np.einsum('ic,jc,kc->ijk', U, V, W) + np.einsum('ic,jc,kc->ijk', U, V, W)
                # Actually, let's do proper alternating least squares
                
            # Better: use ALS
            # Update U: for each i, T[i,:,:] = sum_c U[i,c] * V[:,c] * W[:,c]
            # => T[i,:,:] = sum_c U[i,c] * (V[:,c] * W[:,c])
            # This is a linear system in U[i,:]
            
            # Build the system for U
            # T[i,:,:] reshaped as (mp*p*n,) = U[i,:] @ (V*W) reshaped
            # V*W has shape (mp, pn) for each c, so V[:,c]*W[:,c] is a vector of size mp*pn
            # Actually V[:,c] is size mp, W[:,c] is size pn, their outer product is mp*pn
            # T[i,:,:] reshaped is size mp*pn
            # So T_reshaped[i, c] = U[i,c] * sum... no
            
            # Let me redo: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed i: T[i,j,k] over j,k = sum_c U[i,c] * V[j,c] * W[k,c]
            # Let M_c[j,k] = V[j,c] * W[k,c], a mp x pn matrix
            # Then T[i,:,:] = sum_c U[i,c] * M_c
            # Vectorizing: vec(T[i,:,:]) = sum_c U[i,c] * vec(M_c)
            # So we have a linear system: A @ U[i,:] = vec(T[i,:,:])
            # where A[c, d] = vec(M_d)[c]... 
            # Actually: vec(T[i,:,:])[idx] = sum_c U[i,c] * vec(M_c)[idx]
            # So [vec(M_0), vec(M_1), ..., vec(M_{r-1})] @ U[i,:] = vec(T[i,:,:])
            
            # Build A: shape (mp*pn, r)
            A_U = np.zeros((m*p * p*n, r))
            for c in range(r):
                M_c = np.outer(V[:,c], W[:,c])  # shape (mp, pn)
                A_U[:,c] = M_c.flatten()
            
            # Solve for each i
            for i in range(n*m):
                b = T[i,:,:].flatten()  # shape (mp*pn,)
                # Least squares: A_U @ x = b
                try:
                    x, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                    U[i,:] = x
                except:
                    pass
            
            # Update V similarly
            A_V = np.zeros((n*m * p*n, r))
            for c in range(r):
                M_c = np.outer(U[:,c], W[:,c])  # shape (nm, pn)
                A_V[:,c] = M_c.flatten()
            
            for j in range(m*p):
                b = T[:,j,:].flatten()  # shape (nm*pn,)
                try:
                    x, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                    V[j,:] = x
                except:
                    pass
            
            # Update W similarly
            A_W = np.zeros((n*m * m*p, r))
            for c in range(r):
                M_c = np.outer(U[:,c], V[:,c])  # shape (nm, mp)
                A_W[:,c] = M_c.flatten()
            
            for k in range(p*n):
                b = T[:,:,k].flatten()  # shape (nm*mp,)
                try:
                    x, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                    W[k,:] = x
                except:
                    pass
        
        # Check final error
        T_est = np.einsum('ic,jc,kc->ijk', best_U, best_V, best_W)
        err = np.linalg.norm(T_est - T)
        
        if err < 1e-8:
            # Try to convert to rational/half-integer
            U_rat = _to_rational(best_U)
            V_rat = _to_rational(best_V)
            W_rat = _to_rational(best_W)
            if U_rat is not None and V_rat is not None and W_rat is not None:
                # Verify exactness
                if _verify_exact(U_rat, V_rat, W_rat, n, m, p):
                    return U_rat, V_rat, W_rat
        return None
    
    def _to_rational(matrix, max_denom=1000):
        """Convert float matrix to half-integers if possible."""
        result = []
        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix[0])):
                val = matrix[i,j]
                # Try to find a fraction with small denominator
                found = False
                for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    frac = round(val * denom) / denom
                    if abs(val - frac) < 1e-6:
                        row.append(frac)
                        found = True
                        break
                if not found:
                    # Try larger denominators
                    for denom in range(3, max_denom+1, 2):
                        frac = round(val * denom) / denom
                        if abs(val - frac) < 1e-6:
                            row.append(frac)
                            found = True
                            break
                if not found:
                    return None
            result.append(row)
        return result
    
    def _verify_exact(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        # Check T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal terms are zero
        # For terms where the indices don't form a valid multiplication
        # T[a,b,c] should be 0 unless a=i*m+j, b=j'*p+k, c=k'*n+i' with j=j' and k=k'
        # Actually, the tensor is defined as T[i*m+j, j*p+k, k*n+i] = 1
        # All other entries are 0
        # Let's check a few off-diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This entry should be 1
                    pass
        # Check that non-structural entries are 0
        # Structural entries: T[i*m+j, j*p+k, k*n+i] = 1
        # Non-structural: all others should be 0
        # Let's verify by checking that the computed tensor matches
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i][c] * V[j][c] * W[k][c]
                    # Determine if this is a structural entry
                    ii = i // m
                    jj = i % m
                    jj2 = j // p
                    kk = j % p
                    kk2 = k // n
                    ii2 = k % n
                    if jj == jj2 and kk == kk2 and ii == ii2:
                        if abs(s - 1.0) > 1e-9:
                            return False
                    else:
                        if abs(s) > 1e-9:
                            return False
        return True
    
    # Try ranks from 60 down to see if we can find better than 72
    for r in range(60, 72):
        if time.time() > deadline - 20:
            break
        result = try_rank_decomposition(r, max_iter=2000)
        if result is not None:
            U_rat, V_rat, W_rat = result
            best_result = {"U": U_rat, "V": V_rat, "W": W_rat}
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
