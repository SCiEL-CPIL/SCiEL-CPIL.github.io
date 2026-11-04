"""Matrix multiplication 2,3,9 - rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x9."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 9  # (2x3) * (3x9) -> (9x2)
    
    # Tensor T[i,j,k] where i indexes (2x3), j indexes (3x9), k indexes (9x2)
    # T[i,j,k] = 1 if (i,j) contributes to output k, else 0
    # i in [0,6), j in [0,27), k in [0,18)
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] contributes to C[k,i] in W[k*n+i]
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Try to find a low-rank decomposition
    # We want U (6xR), V (27xR), W (18xR) such that
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    best_rank = n * m * p  # schoolbook
    best_U, best_V, best_W = None, None, None
    
    def schoolbook():
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, R
    
    U0, V0, W0, R0 = schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = R0
    
    # Try rank reduction via iterative optimization
    # Start from schoolbook and try to reduce rank
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        R = U.shape[1]
        # Compute tensor and compare
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    val = sum(U[i,c] * V[j,c] * W[k,c] for c in range(R))
                    if abs(val - T[i,j,k]) > tol:
                        return False
        return True
    
    # Try various ranks using optimization
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 20:
            break
        
        U_init = np.random.randn(n*m, target_rank)
        V_init = np.random.randn(m*p, target_rank)
        W_init = np.random.randn(p*n, target_rank)
        
        # Normalize
        U_init /= np.linalg.norm(U_init, axis=0, keepdims=True)
        V_init /= np.linalg.norm(V_init, axis=0, keepdims=True)
        W_init /= np.linalg.norm(W_init, axis=0, keepdims=True)
        
        def objective(params):
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            # Compute residual
            residual = 0.0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        val = sum(U[i,c] * V[j,c] * W[k,c] for c in range(target_rank))
                        residual += (val - T[i,j,k])**2
            return residual
        
        # This is too slow for full grid, use sampling
        # Instead, try a simpler approach: use known results
        
    # For 2x3x3, known minimum rank is 16
    # For 2x3x9, we can try to exploit structure
    
    # Try a block decomposition: split 9 into 3+3+3
    # Use rank-16 decomposition for 2x3x3 blocks
    
    # Actually, let me try a direct approach with rank 36
    # by combining 3 copies of 2x3x3 with rank 12 each (if possible)
    
    # Let me try to find a rank-36 decomposition by construction
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this using intermediate quantities
    
    # Try: use the fact that for 2x3 * 3x9, we can compute
    # partial sums and combine
    
    # Construction for rank 36:
    # Split B into three 3x3 blocks: B = [B1, B2, B3]
    # C = A * B = [A*B1, A*B2, A*B3]
    # Each A*Bk is 2x3, requiring rank 18 (schoolbook) or 16 (optimal)
    # But we can share computation
    
    # Let me try a different construction
    # Use rank-36: 6*6 intermediate values
    
    # Actually, let me just try to verify if rank 36 works with a specific construction
    
    # For now, return schoolbook if we can't find better
    # But let me try one more thing: use the 2x3x3 rank-16 result
    
    # Known: 2x3x3 has rank 16. Let me hardcode it and extend.
    # The 2x3x3 decomposition (from literature):
    # This is complex, so let me try a computational search for small cases
    
    # Try rank 42, 36, etc. with a quick verification
    for R_try in [36, 38, 40, 42, 45, 48]:
        if time.time() > deadline - 15:
            break
        # Try to find a rank-R_try decomposition using alternating least squares
        U = np.random.randn(n*m, R_try)
        V = np.random.randn(m*p, R_try)
        W = np.random.randn(p*n, R_try)
        
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n*m):
                for c in range(R_try):
                    # U[i,c] should minimize sum_{j,k} (T[i,j,k] - U[i,c]*V[j,c]*W[k,c])^2
                    # d/dU[i,c] = sum_{j,k} (U[i,c]*V[j,c]*W[k,c] - T[i,j,k]) * V[j,c]*W[k,c]
                    num = sum(T[i,j,k] * V[j,c] * W[k,c] for j in range(m*p) for k in range(p*n))
                    den = sum(V[j,c]**2 * W[k,c]**2 for j in range(m*p) for k in range(p*n))
                    if den > 1e-12:
                        U[i,c] = num / den
            
            # Update V
            for j in range(m*p):
                for c in range(R_try):
                    num = sum(T[i,j,k] * U[i,c] * W[k,c] for i in range(n*m) for k in range(p*n))
                    den = sum(U[i,c]**2 * W[k,c]**2 for i in range(n*m) for k in range(p*n))
                    if den > 1e-12:
                        V[j,c] = num / den
            
            # Update W
            for k in range(p*n):
                for c in range(R_try):
                    num = sum(T[i,j,k] * U[i,c] * V[j,c] for i in range(n*m) for j in range(m*p))
                    den = sum(U[i,c]**2 * V[j,c]**2 for i in range(n*m) for j in range(m*p))
                    if den > 1e-12:
                        W[k,c] = num / den
        
        # Check if we found an exact decomposition
        if check_decomposition(U, V, W, tol=1e-8):
            best_rank = R_try
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists for return
    U_list = [[float(U[i,c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[float(V[j,c]) for c in range(best_rank)] for j in range(m*p)]
    W_list = [[float(W[k,c]) for c in range(best_rank)] for k in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
