"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1.
    """
    rank = len(U[0])
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)

    # Reshape to expose indices
    U_r = U_arr.reshape(n, m, rank)  # U[i,j,c]
    V_r = V_arr.reshape(m, p, rank)  # V[j,k,c]
    W_r = W_arr.reshape(p, n, rank)  # W[k,i,c]

    # We want T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # Vectorized: for each (i,j), T[i,j,:] = sum_c U_r[i,j,c] * (V_r[j,:,c] * W_r[:,i,c])
    # Precompute M[i,j] of shape (p, rank): M[i,j][k,c] = V_r[j,k,c] * W_r[k,i,c]
    # Then T[i,j,k] = sum_c U_r[i,j,c] * M[i,j][k,c]
    # We can compute T via einsum: 'ijc,jkc,kic->ijk' but that's expensive.
    # Instead, loop over i,j and use matrix multiply.
    
    # Build T efficiently:
    # For fixed i,j: T[i,j,:] = U_r[i,j,:] @ (V_r[j,:,:] * W_r[:,i,:]).T
    # V_r[j,:,:] is (p, rank), W_r[:,i,:] is (p, rank)
    # Element-wise product: (p, rank)
    # Transpose: (rank, p)
    # U_r[i,j,:] is (rank,)
    # Result: (p,)
    
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            # V_r[j, :, :] is (p, rank)
            # W_r[:, i, :] is (p, rank)
            M = (V_r[j, :, :] * W_r[:, i, :]).T  # (rank, p)
            T[i, j, :] = U_r[i, j, :] @ M
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _als_polish(U, V, W, n, m, p, max_iters=20, tol=1e-10, deadline=None):
    """ALS-style polish to reduce residual. Returns (U, V, W, residual)."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), U_r[i,j,:] = argmin sum_k (1 - sum_c U_r[i,j,c]*V_r[j,k,c]*W_r[k,i,c])^2
        # This is linear in U_r[i,j,:]. Let b[i,j,k] = V_r[j,k,c]*W_r[k,i,c] summed over c... no.
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # For fixed i,j: T[i,j,:] = U_r[i,j,:] @ M[i,j] where M[i,j][c,k] = V_r[j,k,c]*W_r[k,i,c]
        # We want T[i,j,:] = 1. So U_r[i,j,:] @ M[i,j] = 1
        # This is a linear system: M[i,j].T @ U_r[i,j,:].T = 1
        # M[i,j] is (rank, p), so M[i,j].T is (p, rank)
        # U_r[i,j,:].T is (rank, 1)
        # (p, rank) @ (rank, 1) = (p, 1) = ones(p,1)
        # Solve: U_r[i,j,:].T = (M[i,j].T @ M[i,j])^{-1} @ M[i,j].T @ ones
        # Or use least squares: U_r[i,j,:].T = np.linalg.lstsq(M[i,j].T, np.ones(p), rcond=None)[0]
        
        for i in range(n):
            for j in range(m):
                # M[i,j][c,k] = V_r[j,k,c] * W_r[k,i,c]
                # V_r[j,:,:] is (p, rank), W_r[:,i,:] is (p, rank)
                # M[i,j] = (V_r[j,:,:] * W_r[:,i,:]).T  # (rank, p)
                M = (V_r[j, :, :] * W_r[:, i, :]).T  # (rank, p)
                # Solve M.T @ x = ones(p) for x = U_r[i,j,:].T
                # M.T is (p, rank)
                try:
                    U_r[i, j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), V_r[j,k,:] = argmin sum_i (1 - sum_c U_r[i,j,c]*V_r[j,k,c]*W_r[k,i,c])^2
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # For fixed j,k: T[:,j,k] = V_r[j,k,:] @ N[j,k] where N[j,k][c,i] = U_r[i,j,c]*W_r[k,i,c]
        # N[j,k] is (rank, n)
        # V_r[j,k,:].T solves: N[j,k].T @ V_r[j,k,:].T = ones(n)
        for j in range(m):
            for k in range(p):
                # N[j,k][c,i] = U_r[i,j,c] * W_r[k,i,c]
                # U_r[:,j,:] is (n, rank), W_r[k,:,:] is (n, rank)
                # N[j,k] = (U_r[:,j,:] * W_r[k,:,:]).T  # (rank, n)
                N = (U_r[:, j, :] * W_r[k, :, :]).T  # (rank, n)
                try:
                    V_r[j, k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), W_r[k,i,:] = argmin sum_j (1 - sum_c U_r[i,j,c]*V_r[j,k,c]*W_r[k,i,c])^2
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # For fixed k,i: T[i,:,k] = W_r[k,i,:] @ P[k,i] where P[k,i][c,j] = U_r[i,j,c]*V_r[j,k,c]
        # P[k,i] is (rank, m)
        # W_r[k,i,:].T solves: P[k,i].T @ W_r[k,i,:].T = ones(m)
        for k in range(p):
            for i in range(n):
                # P[k,i][c,j] = U_r[i,j,c] * V_r[j,k,c]
                # U_r[i,:,:] is (m, rank), V_r[:,k,:] is (m, rank)
                # P[k,i] = (U_r[i,:,:] * V_r[:,k,:]).T  # (rank, m)
                P = (U_r[i, :, :] * V_r[:, k, :]).T  # (rank, m)
                try:
                    W_r[k, i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                except:
                    pass
    
    # Compute residual
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            M = (V_r[j, :, :] * W_r[:, i, :]).T
            T[i, j, :] = U_r[i, j, :] @ M
    residual = np.max(np.abs(T - 1.0))
    
    return U_r.reshape(n*m, rank).tolist(), V_r.reshape(m*p, rank).tolist(), W_r.reshape(p*n, rank).tolist(), residual

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p

    # Randomized structured search
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    # But we'll try a range of ranks
    
    min_rank = max(n*m, m*p, n*p)
    
    for target_rank in range(min_rank, best_rank + 1):
        if time.time() > deadline:
            break
        
        # For each rank, try several random initializations
        max_attempts = 50 if target_rank <= 50 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random sparse initialization with integer/half-integer values
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            W = [[0.0] * target_rank for _ in range(p * n)]
            
            # Use a structured sparsity pattern: each row has a few nonzeros
            # Values from {-1, -0.5, 0.5, 1}
            vals = [-1, -0.5, 0.5, 1]
            
            for i in range(n * m):
                nnz = random.randint(1, min(3, target_rank))
                cols = random.sample(range(target_rank), nnz)
                for c in cols:
                    U[i][c] = random.choice(vals)
            
            for j in range(m * p):
                nnz = random.randint(1, min(3, target_rank))
                cols = random.sample(range(target_rank), nnz)
                for c in cols:
                    V[j][c] = random.choice(vals)
            
            for k in range(p * n):
                nnz = random.randint(1, min(3, target_rank))
                cols = random.sample(range(target_rank), nnz)
                for c in cols:
                    W[k][c] = random.choice(vals)
            
            # Verify
            if _verify(U, V, W, n, m, p):
                if target_rank < best_rank:
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    if best_rank <= min_rank:
                        return best
            
            # If not verified, try ALS polish
            if time.time() < deadline - 10:
                U_p, V_p, W_p, resid = _als_polish(U, V, W, n, m, p, max_iters=10, deadline=deadline - 5)
                if resid < 1e-9:
                    if target_rank < best_rank:
                        best = {"U": U_p, "V": V_p, "W": W_p}
                        best_rank = target_rank
                        if best_rank <= min_rank:
                            return best
    
    return best
# EVOLVE-BLOCK-END
