"""Matrix multiplication 3x3x6 - rank-minimal exact tensor decomposition."""

import numpy as np
import time
import sys

START_TIME = time.time()
DEADLINE = 145.0  # 5s safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _verify_decomp(U, V, W, n=3, m=3, p=6):
    """Verify decomposition. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _build_schoolbook(n=3, m=3, p=6):
    """Schoolbook rank-54 decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank23_als(n=3, rank=23, max_trials=30, max_iters=300):
    """Try to find a rank-23 decomposition for 3x3x3 matrix multiplication
    using Alternating Least Squares.
    
    The condition: for all i,j,k in {0,1,2}:
        sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    Returns (U, V, W) as numpy arrays of shape (9, rank) if found, else None.
    """
    if _time_left() < 10:
        return None
    
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    for trial in range(max_trials):
        if _time_left() < 15:
            break
        
        np.random.seed(trial * 1000 + 7)
        
        # Random initialization with various scales
        scale = [0.5, 1.0, 2.0, 0.1, 5.0][trial % 5]
        U = np.random.randn(n*n, rank) * scale
        V = np.random.randn(n*n, rank) * scale
        W = np.random.randn(n*n, rank) * scale
        
        for it in range(max_iters):
            if _time_left() < 15:
                break
            
            # Solve for U: for each (i,j), find U[i,j,:] such that
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
            for i in range(n):
                for j in range(n):
                    # A[k, r] = V[j,k,r] * W[k,i,r]
                    A_mat = np.zeros((n, rank))
                    for k in range(n):
                        A_mat[k, :] = V[j*n+k, :] * W[k*n+i, :]
                    # Need: A_mat @ U[i,j,:]^T = ones(n)
                    # i.e., A_mat.T @ U[i,j,:] = ones(n)
                    try:
                        sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat.T, np.ones(n), rcond=None)
                        U[i*n+j, :] = sol
                    except:
                        pass
            
            # Solve for V: for each (j,k), find V[j,k,:] such that
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i
            for j in range(n):
                for k in range(n):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*n+j, :] * W[k*n+i, :]
                    try:
                        sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat.T, np.ones(n), rcond=None)
                        V[j*n+k, :] = sol
                    except:
                        pass
            
            # Solve for W: for each (k,i), find W[k,i,:] such that
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all j
            for k in range(n):
                for i in range(n):
                    A_mat = np.zeros((n, rank))
                    for j in range(n):
                        A_mat[j, :] = U[i*n+j, :] * V[j*n+k, :]
                    try:
                        sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat.T, np.ones(n), rcond=None)
                        W[k*n+i, :] = sol
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*n+j, r] * V[j*n+k, r] * W[k*n+i, r]
                        err += (s - 1.0)**2
            err /= (n*n*n)
            
            if err < best_error:
                best_error = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
            
            # Early stop if no progress
            if it > 50 and err > best_error * 1.01:
                break
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def _try_rank23_gradient(n=3, rank=23, max_iters=5000):
    """Try to find a rank-23 decomposition using gradient descent on the loss
    L = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    """
    if _time_left() < 10:
        return None
    
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    for trial in range(10):
        if _time_left() < 15:
            break
        
        np.random.seed(trial * 777 + 3)
        U = np.random.randn(n*n, rank) * 0.5
        V = np.random.randn(n*n, rank) * 0.5
        W = np.random.randn(n*n, rank) * 0.5
        
        lr = 0.01
        for it in range(max_iters):
            if _time_left() < 15:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*n+j, r] * V[j*n+k, r] * W[k*n+i, r]
                        diff = s - 1.0
                        for r in range(rank):
                            dU[i*n+j, r] += diff * V[j*n+k, r] * W[k*n+i, r]
                            dV[j*n+k, r] += diff * U[i*n+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += diff * U[i*n+j, r] * V[j*n+k, r]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if it % 100 == 99:
                err = 0.0
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*n+j, r] * V[j*n+k, r] * W[k*n+i, r]
                            err += (s - 1.0)**2
                err /= (n*n*n)
                
                if err < best_error:
                    best_error = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < 1e-10:
                    return best_U, best_V, best_W
                
                # Decay learning rate
                lr *= 0.95
        
        # Final check
        err = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*n+j, r] * V[j*n+k, r] * W[k*n+i, r]
                    err += (s - 1.0)**2
        err /= (n*n*n)
        
        if err < best_error:
            best_error = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def _try_rank23_direct(n=3, rank=23):
    """Try a direct construction approach for rank-23 3x3x3 decomposition.
    
    Strategy: Start with the 27 schoolbook terms and try to find linear
    dependencies to reduce the rank.
    """
    if _time_left() < 10:
        return None
    
    # The 27 schoolbook terms for 3x3x3
    # Term (i,j,k): U=E_{i,j}, V=E_{j,k}, W=E_{k,i}
    
    # We want to find 23 vectors in the span of the 27 that still work.
    # This is equivalent to finding a 23-dimensional subspace that contains
    # a valid decomposition.
    
    # Alternative: use the fact that the tensor has symmetries.
    # The 3x3x3 multiplication tensor is invariant under cyclic permutations
    # of indices (i->j->k->i) combined with appropriate matrix transpositions.
    
    # Let me try a different approach: use a known parametrization.
    # The rank-23 decomposition can be found by solving a system of polynomial
    # equations. This is hard in general, but let me try a structured approach.
    
    # Actually, let me try to use the "Strassen-like" construction for 3x3.
    # The idea is to use 23 "products" (rank-1 terms) instead of 27.
    
    # For 2x2x2, Strassen uses 7 instead of 8 products.
    # For 3x3x3, the reduction from 27 to 23 requires a more complex scheme.
    
    # Let me try a numerical approach: set up the problem as finding
    # 23 rank-1 tensors that sum to the target tensor.
    
    # Target tensor T in R^9 x R^9 x R^9:
    # T[i*3+j, j*3+k, k*3+i] = 1 for all i,j,k in {0,1,2}
    # T[other entries] = 0
    
    # We want: sum_{r=0}^{22} U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    
    # This is equivalent to: for all (a,b,c) in {0..8}^3:
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Where T[a,b,c] = 1 if a=i*3+j, b=j*3+k, c=k*3+i for some i,j,k, else 0.
    
    # Let me verify: for a=i*3+j, b=j*3+k, c=k*3+i:
    # We need i from a: i = a//3, j = a%3
    # We need j from b: j' = b//3, k = b%3
    # We need k from c: k' = c//3, i' = c%3
    # T[a,b,c] = 1 if j == j' and k == k' and i == i'
    
    # So T[a,b,c] = 1 iff (a%3 == b//3) and (b%3 == c//3) and (c%3 == a//3)
    
    # Let me build this tensor and try to decompose it.
    
    T = np.zeros((n*n, n*n, n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = i*n + j
                b = j*n + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Now try to find rank-23 decomposition.
    # Use a randomized iterative approach.
    
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    for trial in range(20):
        if _time_left() < 15:
            break
        
        np.random.seed(trial * 31337 + 1)
        U = np.random.randn(n*n, rank)
        V = np.random.randn(n*n, rank)
        W = np.random.randn(n*n, rank)
        
        for it in range(200):
            if _time_left() < 15:
                break
            
            # ALS: solve for each factor
            for i in range(n):
                for j in range(n):
                    a = i*n + j
                    # Need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all b,c
                    # This is a system: for each (b,c), sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
                    # Let X[b,c] = V[b,:] * W[c,:] (element-wise, shape n*n x rank)
                    # Then: U[a,:] @ X[b,c]^T = T[a,b,c]
                    # This is: sum over (b,c) of (U[a,:] @ X[b,c]^T - T[a,b,c])^2 = 0
                    # Linear in U[a,:]: minimize sum_{b,c} (U[a,:] @ X[b,c]^T - T[a,b,c])^2
                    # = ||X_flat^T @ U[a,:]^T - T_flat||^2
                    # where X_flat is (n*n*n, rank) and T_flat is (n*n*n,)
                    
                    # Build the linear system
                    # For each (b,c): coefficient vector is V[b,:]*W[c,:] (element-wise)
                    # We need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                    
                    # This is underdetermined (rank=23 unknowns, n*n*n=27 equations)
                    # Use least squares
                    A_mat = np.zeros((n*n*n, rank))
                    t_vec = np.zeros(n*n*n)
                    idx = 0
                    for b in range(n*n):
                        for c in range(n*n):
                            A_mat[idx, :] = V[b, :] * W[c, :]
                            t_vec[idx] = T[a, b, c]
                            idx += 1
                    
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, t_vec, rcond=None)
                        U[a, :] = sol
                    except:
                        pass
            
            # Similarly for V and W
            for j in range(n):
                for k in range(n):
                    b = j*n + k
                    A_mat = np.zeros((n*n*n, rank))
                    t_vec = np.zeros(n*n*n)
                    idx = 0
                    for a in range(n*n):
                        for c in range(n*n):
                            A_mat[idx, :] = U[a, :] * W[c, :]
                            t_vec[idx] = T[a, b, c]
                            idx += 1
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, t_vec, rcond=None)
                        V[b, :] = sol
                    except:
                        pass
            
            for k in range(n):
                for i in range(n):
                    c = k*n + i
                    A_mat = np.zeros((n*n*n, rank))
                    t_vec = np.zeros(n*n*n)
                    idx = 0
                    for a in range(n*n):
                        for b in range(n*n):
                            A_mat[idx, :] = U[a, :] * V[b, :]
                            t_vec[idx] = T[a, b, c]
                            idx += 1
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, t_vec, rcond=None)
                        W[c, :] = sol
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for a in range(n*n):
                for b in range(n*n):
                    for c in range(n*n):
                        s = 0.0
                        for r in range(rank):
                            s += U[a, r] * V[b, r] * W[c, r]
                        err += (s - T[a, b, c])**2
            
            if err < best_error:
                best_error = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def _build_rank46_from_rank23(U23, V23, W23, n=3, p=6):
    """Construct a rank-46 decomposition for 3x3x6 from a rank-23 3x3x3 decomposition.
    
    Block 1: k=0,1,2 -> C[:,0:3] = A * B[:,0:3]
    Block 2: k=3,4,5 -> C[:,3:6] = A * B[:,3:6]
    """
    rank46 = 46
    U46 = np.zeros((n*n, rank46))
    V46 = np.zeros((n*p, rank46))
    W46 = np.zeros((p*n, rank46))
    
    # Block 1: k=0,1,2
    for r in range(23):
        U46[:, r] = U23[:, r]
        for j in range(n):
            for k in range(n):
                V46[j*p+k, r] = V23[j*n+k, r]
        for k in range(n):
            for i in range(n):
                W46[k*n+i, r] = W23[k*n+i, r]
    
    # Block 2: k=3,4,5
    for r in range(23):
        U46[:, r+23] = U23[:, r]
        for j in range(n):
            for k in range(n):
                V46[j*p+(k+3), r+23] = V23[j*n+k, r]
        for k in range(n):
            for i in range(n):
                W46[(k+3)*n+i, r+23] = W23[k*n+i, r]
    
    return U46, V46, W46

def construct():
    n, m, p = 3, 3, 6
    
    # Try to find rank-46 decomposition
    if _time_left() > 20:
        # Try ALS approach
        r23 = _try_rank23_als(n, 23, max_trials=20, max_iters=200)
        
        if r23 is None and _time_left() > 15:
            # Try direct approach
            r23 = _try_rank23_direct(n, 23)
        
        if r23 is None and _time_left() > 10:
            # Try gradient descent
            r23 = _try_rank23_gradient(n, 23, max_iters=3000)
        
        if r23 is not None and _time_left() > 5:
            U23, V23, W23 = r23
            U46, V46, W46 = _build_rank46_from_rank23(U23, V23, W23, n, p)
            
            # Verify
            if _verify_decomp(U46, V46, W46, n, m, p):
                return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
    
    # Fallback: schoolbook rank 54
    return _build_schoolbook(n, m, p)

DIMENSIONS = [3, 3, 6]
