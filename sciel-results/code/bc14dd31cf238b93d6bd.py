"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 3]

def _build_tensor():
    """Return the 3x3x3 matrix multiplication tensor T[i,j,k] = 1 if i==k else 0,
    indexed as T[i*m+j, k*n+i] -> but we use flat (i,j,k) with i in 0..2, j in 0..2, k in 0..2.
    Actually the tensor is T[a,b,c] where a=i*m+j (row of A * col of A), b=k*m+i? No.
    
    Standard: C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
    Tensor T[a,b,c] where a=(i,j), b=(j,k), c=(k,i) such that T[a,b,c] = 1 if a's j == b's j.
    a = i*m+j = 3i+j, b = j*p+k = 3j+k, c = k*n+i = 3k+i.
    T[a,b,c] = 1 iff (a mod 3) == (b // 3) i.e., j_from_a == j_from_b.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _check_decomposition(U, V, W):
    """Check if U,V,W exactly decompose the tensor using integer arithmetic.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    We check: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
    Multiply by 8 to clear half-integers: 8*U*V*W summed should equal 8*T.
    """
    T = _build_tensor()
    n, m, p = DIMENSIONS
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    # Convert to integers by multiplying by 2
    U2 = (U * 2).round().astype(np.int64)
    V2 = (V * 2).round().astype(np.int64)
    W2 = (W * 2).round().astype(np.int64)
    
    # Compute decomposition
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                lhs = int(np.sum(U2[a, :] * V2[b, :] * W2[c, :]))
                rhs = int(8 * T[a, b, c])
                if lhs != rhs:
                    return False
    return True

def _snap_to_half_int(x, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    x2 = x * 2
    rounded = np.round(x2)
    if np.all(np.abs(x2 - rounded) < tol):
        return rounded / 2.0
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Start with schoolbook rank 27
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    U = np.zeros((n*m, best_rank))
    V = np.zeros((m*p, best_rank))
    W = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    if _check_decomposition(U, V, W):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by removing terms
    # Strategy: For each possible rank r from 26 down to 20, try to find a rank-r decomposition
    # using ALS (alternating least squares) with random initializations and rank deflation
    
    np.random.seed(42)
    random.seed(42)
    
    # Flatten tensor for ALS: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Fix W, solve for U and V via linear least squares
    # Fix U and V, solve for W via linear least squares
    
    def als_step(U, V, W, n_iters=50):
        """Perform ALS iterations to minimize tensor reconstruction error."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            # Fix V and W, solve for U
            # For each a: U[a,:] should satisfy sum_r U[a,r] V[b,r] W[c,r] ≈ T[a,b,c]
            # Let M[a,b,c] = T[a,b,c]. We need U[a,:] such that for all b,c:
            # sum_r U[a,r] (V[b,r] W[c,r]) = T[a,b,c]
            # Let Z[c, r] = V[b,r] W[c,r] ... this is not separable.
            # Better: vec approach. For fixed a, we have:
            # sum_r U[a,r] * (outer product of V[:,r] and W[:,r]) flattened = T[a,:,:] flattened
            # Let P_r = V[:,r] ⊗ W[:,r] (flattened, size (m*p)*(p*n))
            # Then U[a,:] @ P = T[a,:,:].flatten()
            # This is a linear system in U[a,:].
            
            # Build system: for each a, A_sys[a,b,c] = V[b,r] * W[c,r] summed over r
            # Actually, for fixed a, we solve: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
            # Let X[a,b,c] = V[b,r] W[c,r] for each r, but we sum over r with coefficients U[a,r]
            # So: for each (b,c), sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
            # Let Q[b,c,r] = V[b,r] * W[c,r]
            # Then U[a,:] @ Q[b,c,:] = T[a,b,c]
            # Stack over (b,c): U[a,:] @ Q_flat = T[a,:,:].flatten()
            # where Q_flat is ((m*p)*(p*n), rank)
            
            # Compute Q_flat
            # Q[b,c,r] = V[b,r] * W[c,r]
            # Q_flat[idx, r] where idx = b*(p*n) + c
            Q = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, rank)
            Q_flat = Q.reshape(m*p * p*n, rank)
            
            # Solve for each a
            for a in range(n*m):
                rhs = T[a, :, :].reshape(m*p * p*n)
                # Least squares: U[a,:] @ Q_flat = rhs
                try:
                    sol, _, _, _ = np.linalg.lstsq(Q_flat.T, rhs, rcond=None)
                    U[a, :] = sol
                except:
                    pass
            
            # Fix U and W, solve for V
            # For each b: sum_r V[b,r] U[a,r] W[c,r] = T[a,b,c]
            # Q2[a,c,r] = U[a,r] * W[c,r]
            Q2 = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, rank)
            Q2_flat = Q2.reshape(n*m * p*n, rank)
            
            for b in range(m*p):
                rhs = T[:, b, :].reshape(n*m * p*n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(Q2_flat.T, rhs, rcond=None)
                    V[b, :] = sol
                except:
                    pass
            
            # Fix U and V, solve for W
            # For each c: sum_r W[c,r] U[a,r] V[b,r] = T[a,b,c]
            Q3 = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
            Q3_flat = Q3.reshape(n*m * m*p, rank)
            
            for c in range(p*n):
                rhs = T[:, :, c].reshape(n*m * m*p)
                try:
                    sol, _, _, _ = np.linalg.lstsq(Q3_flat.T, rhs, rcond=None)
                    W[c, :] = sol
                except:
                    pass
        
        return U, V, W
    
    def try_rank_reduction(target_rank, n_trials=20):
        """Try to find a decomposition of target_rank using ALS with random starts."""
        n, m, p = DIMENSIONS
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_rand = np.random.randn(n*m, target_rank) * 0.5
            V_rand = np.random.randn(m*p, target_rank) * 0.5
            W_rand = np.random.randn(p*n, target_rank) * 0.5
            
            # ALS
            U_als, V_als, W_als = als_step(U_rand, V_rand, W_rand, n_iters=30)
            
            # Try snapping to half-integers
            U_snapped = _snap_to_half_int(U_als)
            V_snapped = _snap_to_half_int(V_als)
            W_snapped = _snap_to_half_int(W_als)
            
            if U_snapped is not None and V_snapped is not None and W_snapped is not None:
                if _check_decomposition(U_snapped, V_snapped, W_snapped):
                    return U_snapped, V_snapped, W_snapped
        
        return None, None, None
    
    # Try ranks from 26 down
    for r in range(best_rank - 1, max(best_rank - 8, 15), -1):
        if time.time() > deadline:
            break
        
        U_r, V_r, W_r = try_rank_reduction(r, n_trials=15)
        if U_r is not None:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = r
            break  # Found a lower rank, stop
    
    # If we still have the schoolbook rank, return it
    if best_U is None:
        best_U = U
        best_V = V
        best_W = W
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
