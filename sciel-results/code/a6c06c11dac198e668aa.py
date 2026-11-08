"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build target tensor: T[i, j, k] = 1 for all valid (i,j,k)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) is not a valid lower bound
    # A better lower bound: the rank is at least max(n, m, p) * min(n, m, p) / max(n, m, p)
    # Actually, let's just try ranks from 12 to 24
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    
    def verify(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err < tol
    
    def try_rank(r, num_restarts=50, max_iters=2000):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_factors = None
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random Gaussian
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # ALS iterations
            prev_err = float('inf')
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve linear system
                for i in range(n):
                    for j in range(m):
                        # U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:])_c = 1 for all k
                        # This is: U[i*m+j, :] @ (V[j*p:,:]*W[:,i][:,None]).T ... 
                        # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # For fixed i,j: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for each k
                        # Let A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                        # Then U[i*m+j,:] @ A.T = ones(p)
                        # So U[i*m+j,:] = solve(A.T @ A, A.T @ ones(p))
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve: A.T @ U_row = ones(p), i.e., U_row = (A.T @ A)^{-1} A.T @ ones
                        ATA = A.T @ A
                        ATb = A.T @ np.ones(p)
                        try:
                            U[i * m + j, :] = np.linalg.solve(ATA + 1e-10 * np.eye(r), ATb)
                        except np.linalg.LinAlgError:
                            U[i * m + j, :] = np.linalg.lstsq(ATA + 1e-10 * np.eye(r), ATb, rcond=None)[0]
                
                # Update V: for each (j,k), solve linear system
                for j in range(m):
                    for k in range(p):
                        # sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
                        # Let B[i,c] = U[i*m+j,c] * W[k*n+i,c]
                        # Then B.T @ V_row = ones(n)
                        B = np.zeros((n, r))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        BTA = B.T @ B
                        BTb = B.T @ np.ones(n)
                        try:
                            V[j * p + k, :] = np.linalg.solve(BTA + 1e-10 * np.eye(r), BTb)
                        except np.linalg.LinAlgError:
                            V[j * p + k, :] = np.linalg.lstsq(BTA + 1e-10 * np.eye(r), BTb, rcond=None)[0]
                
                # Update W: for each (k,i), solve linear system
                for k in range(p):
                    for i in range(n):
                        # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
                        # Let C[j,c] = U[i*m+j,c] * V[j*p+k,c]
                        # Then C.T @ W_row = ones(m)
                        C = np.zeros((m, r))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        CTA = C.T @ C
                        CTb = C.T @ np.ones(m)
                        try:
                            W[k * n + i, :] = np.linalg.solve(CTA + 1e-10 * np.eye(r), CTb)
                        except np.linalg.LinAlgError:
                            W[k * n + i, :] = np.linalg.lstsq(CTA + 1e-10 * np.eye(r), CTb, rcond=None)[0]
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            err += (s - 1.0) ** 2
                err = np.sqrt(err)
                
                if err < best_err:
                    best_err = err
                    best_factors = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-10:
                    break
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
        
        if best_factors is not None and best_err < 1e-6:
            return best_factors
        return None
    
    # Try ranks from lower to upper
    for r in range(12, best_rank):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, num_restarts=30, max_iters=500)
        if result is not None:
            U, V, W = result
            # Try to snap to rational/half-integer values
            best_rank = r
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    return best_result
# EVOLVE-BLOCK-END
