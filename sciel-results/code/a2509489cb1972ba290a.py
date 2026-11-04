"""Program using alternating projection with integer/half-integer constraints
to find a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
"""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build the target tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # And for all other index combinations, the sum should be 0.
    
    # Actually, the tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form is: sum_{i,j,k} A[i,j] B[j,k] C[i,k] = sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) * (sum_{j,k} V[j*p+k,r] B[j,k]) * (sum_{i,k} W[k*n+i,r] C[i,k])
    # Wait, that's not quite right. Let me think more carefully.
    
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # No, that's not right either.
    
    # The correct condition: For the matrix multiplication tensor, we have:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as a trilinear form: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k] = <AB, C>
    # where T[i,j,k] = 1 for all valid (i,j,k).
    
    # But the decomposition is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k? No, that would make the tensor rank 1.
    
    # Let me reconsider. The tensor T is in R^{n*m} * R^{m*p} * R^{p*n}.
    # The components are T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (k,i).
    # T[a, b, c] = 1 if a corresponds to (i,j), b corresponds to (j,k), c corresponds to (k,i)
    # for the same i,j,k. Otherwise T[a, b, c] = 0.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), and 0 otherwise.
    
    # The decomposition: sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    # for all a, b, c.
    
    # For the schoolbook: U[i*m+j, r] = 1 if r corresponds to (i,j,k) where a=(i,j),
    # V[j*p+k, r] = 1 if r corresponds to (i,j,k) where b=(j,k),
    # W[k*n+i, r] = 1 if r corresponds to (i,j,k) where c=(k,i).
    
    # So for a given (a, b, c), the sum is 1 if there exists (i,j,k) such that
    # a = i*m+j, b = j*p+k, c = k*n+i, and 0 otherwise.
    
    # But wait, for a fixed a=(i,j), b=(j',k), c=(k',i'), the sum is 1 if i=i', j=j', k=k'.
    # So T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for the same i,j,k, else 0.
    
    # This is correct. The tensor is sparse: only n*m*p entries are 1, rest are 0.
    
    # Now, can we find a lower rank decomposition?
    
    # Strategy: Use alternating projections.
    # 1. Initialize U, V, W with random values.
    # 2. Iterate:
    #    a. For each column r, project U[:,r] to minimize ||sum_r U[:,r] V[:,r]^T W[:,r]^T - T||_F
    #       This is a linear least squares problem for U[:,r] given V[:,r] and W[:,r].
    #    b. Similarly for V and W.
    #    c. Project to integer/half-integer values.
    
    # Actually, the problem is non-convex. Let's use a different approach.
    
    # For small dimensions, we can try to solve the system of equations directly.
    # The conditions are:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all (i,j,k)
    # sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r] = 0 for all other (a,b,c)
    
    # This is a system of polynomial equations. For rank r, we have:
    # Unknowns: U (n*m*r), V (m*p*r), W (p*n*r) = r*(n*m + m*p + p*n) = r*(12 + 24 + 18) = r*54
    # Equations: (n*m)*(m*p)*(p*n) = 72*24*18 = 31104 (but many are redundant)
    
    # Actually, the number of independent equations is n*m*p = 72 (the non-zero entries)
    # plus the zero entries. But the zero entries are (n*m)*(m*p)*(p*n) - n*m*p = 31104 - 72 = 31032.
    
    # This is a very underdetermined system for small r, but the non-linearity makes it hard.
    
    # Let's try a heuristic: start with the schoolbook and try to merge columns.
    
    # Actually, let's try a random search for small ranks.
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from schoolbook down to a lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But for matrix multiplication, the lower bound is higher.
    
    # Let's try a few random initializations and use gradient descent.
    
    np.random.seed(42)
    
    # Try rank 36 first (half of schoolbook)
    for rank in [36, 48, 60, 72]:
        if rank >= best_rank:
            continue
            
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Alternating least squares
        for it in range(100):
            # Update U
            for r in range(rank):
                # Minimize ||U[:,r] * (V[:,r] * W[:,r]^T) - T_r||_F
                # where T_r is the contribution of the r-th term
                # Actually, we need to minimize ||sum_{r'} U[:,r'] V[:,r']^T W[:,r']^T - T||_F
                # For a fixed r, this is linear in U[:,r]:
                # U[:,r] * (V[:,r] * W[:,r]^T) = T - sum_{r' != r} U[:,r'] V[:,r']^T W[:,r']^T
                # Let M_r = V[:,r] * W[:,r]^T (outer product)
                # Then U[:,r] * M_r = T - sum_{r' != r} U[:,r'] M_{r'}
                # This is a linear system: U[:,r] = (T - sum_{r' != r} U[:,r'] M_{r'}) * M_r^+
                
                M_r = np.outer(V[:, r], W[:, r])
                residual = np.zeros((n * m, m * p, p * n))
                for r2 in range(rank):
                    if r2 != r:
                        residual += np.einsum('ar,br,cr->abc', U[:, r2], V[:, r2], W[:, r2])
                
                target = _build_target_tensor(n, m, p) - residual
                # Now solve U[:,r] * M_r = target
                # This is: for each (a,b,c), sum_d U[a,d] M_r[d,b] W_r[c] = target[a,b,c]
                # Wait, M_r is (m*p, p*n), so M_r[d,b] where d indexes m*p and b indexes p*n.
                # Actually, V[:,r] is (m*p,), W[:,r] is (p*n,), so M_r = V[:,r] * W[:,r]^T is (m*p, p*n).
                # U[:,r] is (n*m,), so U[:,r] * M_r doesn't make sense dimensionally.
                
                # Let me reconsider. The decomposition is:
                # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                # For fixed r, U[:,r] V[:,r]^T W[:,r]^T is a tensor of shape (n*m, m*p, p*n).
                # So the full sum is sum_r U[:,r] V[:,r]^T W[:,r]^T.
                
                # For updating U[:,r], we fix V and W:
                # U[:,r] V[:,r]^T W[:,r]^T = target
                # where target = T - sum_{r' != r} U[:,r'] V[:,r']^T W[:,r']^T
                # This is a linear equation in U[:,r]:
                # For each (b,c): sum_a U[a,r] V[b,r] W[c,r] = target[a,b,c]
                # Wait, that's not right. Let me write it out:
                # (U[:,r] V[:,r]^T W[:,r]^T)[a,b,c] = U[a,r] V[b,r] W[c,r]
                # So the equation is: U[a,r] V[b,r] W[c,r] = target[a,b,c] for all a,b,c.
                # This is: U[a,r] = target[a,b,c] / (V[b,r] W[c,r]) for all b,c where V[b,r] W[c,r] != 0.
                # But this must be consistent for all b,c.
                
                # This is a rank-1 tensor fitting problem. We can use ALS:
                # Minimize ||U[:,r] V[:,r]^T W[:,r]^T - target||_F
                # This is equivalent to minimizing ||reshape(target) - U[:,r] * (V[:,r] * W[:,r]^T)||_F
                # where reshape(target) is a matrix of shape (n*m, m*p * p*n).
                
                target_mat = target.reshape(n * m, m * p * p * n)
                VW = (V[:, r] * W[:, r]).reshape(m * p * p * n)
                # Now minimize ||target_mat - U[:,r] * VW^T||_F
                # This is a linear least squares: U[:,r] = target_mat * VW^T / (VW * VW^T)
                vw_norm_sq = np.dot(VW, VW)
                if vw_norm_sq > 1e-10:
                    U[:, r] = np.dot(target_mat, VW) / vw_norm_sq
            
            # Update V similarly
            for r in range(rank):
                M_r = np.outer(U[:, r], W[:, r])
                residual = np.zeros((n * m, m * p, p * n))
                for r2 in range(rank):
                    if r2 != r:
                        residual += np.einsum('ar,br,cr->abc', U[:, r2], V[:, r2], W[:, r2])
                
                target = _build_target_tensor(n, m, p) - residual
                target_mat = target.reshape(m * p, n * m * p * n)
                UW = (U[:, r] * W[:, r]).reshape(n * m * p * n)
                uw_norm_sq = np.dot(UW, UW)
                if uw_norm_sq > 1e-10:
                    V[:, r] = np.dot(target_mat, UW) / uw_norm_sq
            
            # Update W similarly
            for r in range(rank):
                M_r = np.outer(U[:, r], V[:, r])
                residual = np.zeros((n * m, m * p, p * n))
                for r2 in range(rank):
                    if r2 != r:
                        residual += np.einsum('ar,br,cr->abc', U[:, r2], V[:, r2], W[:, r2])
                
                target = _build_target_tensor(n, m, p) - residual
                target_mat = target.reshape(p * n, n * m * m * p)
                UV = (U[:, r] * V[:, r]).reshape(n * m * m * p)
                uv_norm_sq = np.dot(UV, UV)
                if uv_norm_sq > 1e-10:
                    W[:, r] = np.dot(target_mat, UV) / uv_norm_sq
            
            # Project to integer/half-integer values
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Check convergence
            if it % 10 == 0:
                error = _compute_error(U, V, W, n, m, p)
                if error < 1e-6:
                    break
        
        # Check if valid
        error = _compute_error(U, V, W, n, m, p)
        if error < 1e-6:
            best_rank = rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            break
    
    # If no better decomposition found, return schoolbook
    if best_U is None:
        rank = schoolbook_rank
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _build_target_tensor(n, m, p):
    """Build the target tensor T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for same i,j,k, else 0."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _compute_error(U, V, W, n, m, p):
    """Compute the error between the decomposition and the target tensor."""
    T = _build_target_tensor(n, m, p)
    # Compute sum_r U[a,r] V[b,r] W[c,r]
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(result - T)
