import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strassen-based decomposition for 2x3x5 matrix multiplication
    # Strategy: Split the 2x3 matrix into two 2x2 and 2x1 blocks (or similar)
    # Actually, let's use a direct construction based on known results.
    # The minimal rank for 2x3x5 is known to be 28 (from the cell findings).
    # Let me try to construct a rank-28 decomposition.
    
    # Approach: Use Strassen for the 2x2x2 sub-blocks and schoolbook for the rest.
    # Matrix A is 2x3, Matrix B is 3x5.
    # C = A * B, where C is 2x5.
    
    # Let's decompose B into columns: B = [B1, B2, B3] where B1 is 3x2, B2 is 3x2, B3 is 3x1
    # C = A * B1 + A * B2 + A * B3
    # A * B1: 2x3 * 3x2 = 2x2 (use Strassen, rank 7)
    # A * B2: 2x3 * 3x2 = 2x2 (use Strassen, rank 7)
    # A * B3: 2x3 * 3x1 = 2x1 (schoolbook, rank 3)
    # But wait, A * B1 is not simply 2x2 because A is 2x3.
    
    # Let me be more careful. A is 2x3, B is 3x5.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split B into two parts: B_left (3x2) and B_right (3x3)
    # C_left = A * B_left: 2x3 * 3x2 = 2x2
    # C_right = A * B_right: 2x3 * 3x3 = 2x3
    
    # For C_left (2x2): We can use Strassen if we split A into two 2x2 blocks... but A is 2x3.
    # Let A = [A1, A2] where A1 is 2x2, A2 is 2x1.
    # C_left = A1 * B_left[:,0:2] + A2 * B_left[:,2:3]... this is getting complicated.
    
    # Alternative: Direct rank-28 construction from literature.
    # The known minimal rank for 2x3x5 is 28.
    # Let me try to find it via a structured approach.
    
    # Actually, let's just use the numerical optimization approach but with a better initialization
    # and more iterations, targeting rank 28.
    
    def try_rank(target_rank):
        nU = n * m * target_rank
        nV = m * p * target_rank
        nW = p * n * target_rank
        total_vars = nU + nV + nW
        
        def objective_and_gradient(x):
            U = x[:nU].reshape(n*m, target_rank)
            V = x[nU:nU+nV].reshape(m*p, target_rank)
            W = x[nU+nV:].reshape(p*n, target_rank)
            
            S = np.einsum('ar,br,cr->abc', U, V, W)
            error = S - T
            val = np.sum(error**2)
            
            gU = 2.0 * np.einsum('abc,br,cr->ar', error, V, W)
            gV = 2.0 * np.einsum('abc,ar,cr->br', error, U, W)
            gW = 2.0 * np.einsum('abc,ar,br->cr', error, U, V)
            
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return val, grad
        
        import scipy.optimize as opt
        
        for r in range(30):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + r)
            x0 = np.random.randn(total_vars) * 0.1
            
            try:
                res = opt.minimize(
                    objective_and_gradient,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
            except Exception:
                continue
            
            if res.fun < 1e-10:
                U = res.x[:nU].reshape(n*m, target_rank)
                V = res.x[nU:nU+nV].reshape(m*p, target_rank)
                W = res.x[nU+nV:].reshape(p*n, target_rank)
                
                S = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(S, T, atol=1e-8):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try rank 28 first (known to be possible)
    result = try_rank(28)
    if result is not None:
        return result

    # Try rank 29
    result = try_rank(29)
    if result is not None:
        return result

    # Fallback to schoolbook
    return schoolbook()
