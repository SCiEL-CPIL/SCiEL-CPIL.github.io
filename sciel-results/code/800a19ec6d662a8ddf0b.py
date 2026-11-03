"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    
    The tensor T has size (n*m) x (m*p) x (p*n) where:
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if j == j' and i == i', else 0
    
    Schoolbook uses rank n*m*p. We search for lower rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def build_tensor():
        """Build the target tensor as a 3D array."""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for kp in range(p):
                        T[i * m + j, j * p + k, kp * n + i] = 1.0
        return T
    
    T = build_tensor()
    
    def verify(U, V, W, tol=1e-9):
        """Verify that sum_c U[a,c]*V[b,c]*W[c_idx,c] == T[a,b,c_idx] for all a,b,c_idx."""
        rank = len(U[0])
        # Use vectorized verification
        for a in range(n * m):
            for b in range(m * p):
                # Compute sum_c U[a,c] * V[b,c] * W[:,c] and compare to T[a,b,:]
                prod = U[a] * V[b]  # shape (rank,)
                result = prod @ W  # shape (p*n,)
                if np.any(np.abs(result - T[a, b]) > tol):
                    return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix operations."""
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # For each a, b: sum_c U[a,c]*V[b,c]*W[c_idx,c]
        # = (U * V^T) contracted with W
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # For fixed a: sum_c U[a,c] * (V[b,c] * W[c_idx,c])
        # = U[a,:] @ (V[b,:] * W.T)  -- no, that's not right
        
        # Actually: result[a, b, c_idx] = sum_c U[a,c] * V[b,c] * W[c_idx,c]
        # = sum_c (U[a,c] * V[b,c]) * W[c_idx,c]
        # Let M[a,b] = U[a,:] * V[b,:]  (element-wise, shape rank)
        # Then result[a,b,c_idx] = M[a,b] @ W[c_idx,:]
        
        # Vectorized: 
        # U_expanded: (n*m, 1, rank)
        # V_expanded: (1, m*p, rank)
        # M = U_expanded * V_expanded: (n*m, m*p, rank)
        # result = M @ W.T: (n*m, m*p, p*n)
        
        U_exp = U[:, np.newaxis, :]  # (n*m, 1, rank)
        V_exp = V[np.newaxis, :, :]  # (1, m*p, rank)
        M = U_exp * V_exp  # (n*m, m*p, rank)
        result = M @ W.T  # (n*m, m*p, p*n)
        
        if np.any(np.abs(result - T) > tol):
            return False
        return True
    
    def try_decomposition(rank, max_iters=200, seed=None):
        """Try to find a decomposition of given rank."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        n_m = n * m
        m_p = m * p
        p_n = p * n
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            U = np.random.uniform(-1, 1, (n_m, rank))
            V = np.random.uniform(-1, 1, (m_p, rank))
            W = np.random.uniform(-1, 1, (p_n, rank))
            
            # Quick check
            if verify_fast(U, V, W, tol=1e-6):
                return U.tolist(), V.tolist(), W.tolist()
            
            # Try local optimization: gradient descent on the residual
            # Residual R[a,b,c] = sum_c' U[a,c']*V[b,c']*W[c,c'] - T[a,b,c]
            # We want to minimize ||R||_F^2
            
            for inner in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Compute residual
                U_exp = U[:, np.newaxis, :]
                V_exp = V[np.newaxis, :, :]
                M = U_exp * V_exp
                result = M @ W.T
                R = result - T  # (n_m, m_p, p_n)
                
                if np.linalg.norm(R, 'fro') < 1e-8:
                    break
                
                # Gradient with respect to U:
                # dR/da = V[b,c] * W[c_idx,c]
                # grad_U[a,c] = sum_{b,c_idx} R[a,b,c_idx] * V[b,c] * W[c_idx,c]
                # = (R * (V_exp * W.T[np.newaxis,:,:])).sum(axis=(1,2))
                
                # grad_U[a,c] = sum_b sum_c_idx R[a,b,c_idx] * V[b,c] * W[c_idx,c]
                # Let's compute: for each a, grad_U[a,:] = sum_b (R[a,b,:] * (V[b,:] * W.T)).sum... 
                # Actually: grad_U[a,c] = sum_b sum_c_idx R[a,b,c_idx] * V[b,c] * W[c_idx,c]
                # = sum_b [R[a,b,:] * (V[b,c] * W.T[c_idx,c] for all c_idx)]
                # Hmm, let me think differently.
                
                # grad_U[a,c] = sum_{b,c_idx} R[a,b,c_idx] * V[b,c] * W[c_idx,c]
                # = sum_b [R[a,b,:] @ (V[b,c] * W[:,c])]  -- no
                # = sum_b V[b,c] * (R[a,b,:] @ W[:,c])
                
                # Let's compute R @ W: (n_m, m_p, rank) where (R@W)[a,b,c] = sum_c_idx R[a,b,c_idx]*W[c_idx,c]
                RW = np.einsum('abc,dc->abd', R, W)  # (n_m, m_p, rank)
                # grad_U[a,c] = sum_b RW[a,b,c] * V[b,c]
                grad_U = np.einsum('abc,bc->ac', RW, V)
                
                # grad_V[b,c] = sum_{a,c_idx} R[a,b,c_idx] * U[a,c] * W[c_idx,c]
                RU = np.einsum('abc,dc->abd', R, W)  # same as RW
                # Wait: grad_V[b,c] = sum_a sum_c_idx R[a,b,c_idx] * U[a,c] * W[c_idx,c]
                # = sum_a U[a,c] * (R[:,b,:] @ W[:,c])
                # Let RW2[a,b,c] = sum_c_idx R[a,b,c_idx]*W[c_idx,c] (same as RW above)
                grad_V = np.einsum('abc,ac->bc', RW, U)
                
                # grad_W[c_idx,c] = sum_{a,b} R[a,b,c_idx] * U[a,c] * V[b,c]
                # = sum_a sum_b R[a,b,c_idx] * U[a,c] * V[b,c]
                UV = np.einsum('ac,bc->ab', U, V)  # (n_m, m_p)
                # grad_W[c_idx,c] = sum_{a,b} R[a,b,c_idx] * UV[a,b]
                grad_W = np.einsum('abc,ab->dc', R, UV)
                
                # Gradient descent step
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if verify_fast(U, V, W, tol=1e-7):
                    return U.tolist(), V.tolist(), W.tolist()
            
            # If we got close, try to refine
            U_exp = U[:, np.newaxis, :]
            V_exp = V[np.newaxis, :, :]
            M = U_exp * V_exp
            result = M @ W.T
            if np.linalg.norm(result - T, 'fro') < 1e-5:
                # Try to snap to exact values
                if verify_fast(U, V, W, tol=1e-9):
                    return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try various ranks from low to high
    best_result = None
    
    # Theoretical lower bound: at least max(n*m, m*p, p*n) / something
    # For 4x4*4x9: n*m=16, m*p=36, p*n=36
    # Schoolbook: 144
    
    # Try ranks that might work
    ranks_to_try = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 144]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        result = try_decomposition(rank, max_iters=100, seed=42 + rank)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W, tol=1e-9):
                best_result = (U, V, W, rank)
                break
    
    # If we found a lower rank, use it
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Otherwise, fall back to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
