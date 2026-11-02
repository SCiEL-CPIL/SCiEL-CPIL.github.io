"""Matrix multiplication 3x3 * 3x4 with reduced rank decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _target_tensor(n, m, p):
    """Build the target tensor T of shape (n*m, m*p, p*n) where
    T[i*m+j, j*p+k, k*n+i] = 1 and 0 elsewhere."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _from_columns(U, V, W):
    """Reconstruct tensor from factor matrices."""
    n, m, p = 3, 3, 4
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for r in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return T

def _verify(U, V, W, tol=1e-10):
    """Check if the decomposition is exact."""
    n, m, p = 3, 3, 4
    T = _from_columns(U, V, W)
    T_target = _target_tensor(n, m, p)
    return np.max(np.abs(T - T_target)) < tol

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank15():
    """
    Attempt a rank-15 decomposition using a structured approach.
    
    For 3x3 * 3x4, we use the following strategy:
    - The middle dimension is m=3, so we have 3 "rows" of B / 3 "cols" of A.
    - We can use a Laderman-like construction adapted for rectangular case.
    
    Construction:
    Let's use 15 terms. The key idea is to exploit the structure:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    We'll use the following 15 terms:
    - 12 "base" terms: for each (j,k) pair, a term that captures A[i,j]*B[j,k]
      but we need to handle the i-dependence.
    
    Actually, let me use a different approach. For the 3x3*3x3 case, 
    Strassen gives rank 7. For 3x3*3x4, we can extend this.
    
    Let me try a direct construction with rank 15:
    Use 3 "Strassen-like" blocks for the 3x3 part (rank 7 each would be too much),
    so instead use a simpler approach.
    
    Alternative: Use the fact that we can write the multiplication as:
    C = A * B where A is 3x3 and B is 3x4.
    
    Let me try rank = 12 + 3 = 15:
    - 12 terms for the "diagonal" structure
    - 3 correction terms
    
    Actually, let me just try to solve the system numerically for rank 15.
    """
    n, m, p = 3, 3, 4
    rank = 15
    T_target = _target_tensor(n, m, p)
    
    # Use iterative approach: initialize randomly and refine
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for attempt in range(50):
        np.random.seed(42 + attempt)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Gradient-free refinement: alternating least squares
        for iteration in range(200):
            # Update U: for each row a, minimize ||sum_r U[a,r]*V[:,r]*W[:,r] - T_target[a,:,:]||
            for a in range(n * m):
                # We need: sum_r U[a,r] * (V[:,r] * W[:,r]) = T_target[a,:,:].ravel()
                # This is: M @ U[a,:] = T_target[a,:,:].ravel()
                # where M[b,c] = sum_r V[b,r]*W[c,r] ... no, that's not right.
                # Actually: sum_r U[a,r] * V[b,r] * W[c,r] = T_target[a,b,c]
                # For fixed a: sum_r U[a,r] * (V[b,r] * W[c,r]) = T_target[a,b,c]
                # Let X_r = V[:,r] * W[:,r] (element-wise product, shape (m*p, p*n))
                # Then: sum_r U[a,r] * X_r = T_target[a,:,:]
                # This is a linear system: [X_0, X_1, ..., X_{r-1}] @ U[a,:] = T_target[a,:,:].ravel()
                X = np.einsum('br,cr->bcr', V, W)  # shape (m*p, p*n, rank)
                X_flat = X.transpose(2, 0, 1).reshape(rank, -1)  # shape (rank, m*p*p*n)
                rhs = T_target[a, :, :].ravel()
                # Solve: X_flat @ u_a = rhs
                try:
                    u_a = np.linalg.lstsq(X_flat, rhs, rcond=None)[0]
                    U[a, :] = u_a
                except:
                    pass
            
            # Update V similarly
            for b in range(m * p):
                X = np.einsum('ar,cr->acr', U, W)  # shape (n*m, p*n, rank)
                X_flat = X.transpose(2, 0, 1).reshape(rank, -1)
                rhs = T_target[:, b, :].ravel()
                try:
                    v_b = np.linalg.lstsq(X_flat, rhs, rcond=None)[0]
                    V[b, :] = v_b
                except:
                    pass
            
            # Update W similarly
            for c in range(p * n):
                X = np.einsum('ar,br->abr', U, V)  # shape (n*m, m*p, rank)
                X_flat = X.transpose(2, 0, 1).reshape(rank, -1)
                rhs = T_target[:, :, c].ravel()
                try:
                    w_c = np.linalg.lstsq(X_flat, rhs, rcond=None)[0]
                    W[c, :] = w_c
                except:
                    pass
            
            # Check convergence
            T = _from_columns(U, V, W)
            err = np.max(np.abs(T - T_target))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10:
        return best_U, best_V, best_W
    return None

def _try_rank14():
    """Attempt rank-14 decomposition."""
    n, m, p = 3, 3, 4
    rank = 14
    T_target = _target_tensor(n, m, p)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for attempt in range(30):
        np.random.seed(100 + attempt)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for iteration in range(300):
            for a in range(n * m):
                X = np.einsum('br,cr->bcr', V, W)
                X_flat = X.transpose(2, 0, 1).reshape(rank, -1)
                rhs = T_target[a, :, :].ravel()
                try:
                    U[a, :] = np.linalg.lstsq(X_flat, rhs, rcond=None)[0]
                except:
                    pass
            
            for b in range(m * p):
                X = np.einsum('ar,cr->acr', U, W)
                X_flat = X.transpose(2, 0, 1).reshape(rank, -1)
                rhs = T_target[:, b, :].ravel()
                try:
                    V[b, :] = np.linalg.lstsq(X_flat, rhs, rcond=None)[0]
                except:
                    pass
            
            for c in range(p * n):
                X = np.einsum('ar,br->abr', U, V)
                X_flat = X.transpose(2, 0, 1).reshape(rank, -1)
                rhs = T_target[:, :, c].ravel()
                try:
                    W[c, :] = np.linalg.lstsq(X_flat, rhs, rcond=None)[0]
                except:
                    pass
            
            T = _from_columns(U, V, W)
            err = np.max(np.abs(T - T_target))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10:
        return best_U, best_V, best_W
    return None

def _to_list(U, V, W):
    """Convert numpy arrays to lists with exact integer/half-integer rounding."""
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                # Round to nearest 0.5
                v = round(val * 2) / 2
                # If very close to integer, use integer
                if abs(v - round(v)) < 1e-9:
                    v = int(round(v))
                r.append(v)
            result.append(r)
        return result
    return to_list(U), to_list(V), to_list(W)

def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 140  # Safety margin
    
    # Try rank 14 first
    if time.time() < deadline - 60:
        result = _try_rank14()
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                Ul, Vl, Wl = _to_list(U, V, W)
                # Verify with list version
                U_np = np.array(Ul, dtype=float)
                V_np = np.array(Vl, dtype=float)
                W_np = np.array(Wl, dtype=float)
                if _verify(U_np, V_np, W_np):
                    return {"U": Ul, "V": Vl, "W": Wl}
    
    # Try rank 15
    if time.time() < deadline - 30:
        result = _try_rank15()
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                Ul, Vl, Wl = _to_list(U, V, W)
                U_np = np.array(Ul, dtype=float)
                V_np = np.array(Vl, dtype=float)
                W_np = np.array(Wl, dtype=float)
                if _verify(U_np, V_np, W_np):
                    return {"U": Ul, "V": Vl, "W": Wl}
    
    # Fallback: schoolbook rank 36
    U, V, W = _schoolbook(n, m, p)
    Ul, Vl, Wl = _to_list(U, V, W)
    return {"U": Ul, "V": Vl, "W": Wl}
# EVOLVE-BLOCK-END
