"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
    deadline = time.time() + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # Schoolbook fallback (rank 24)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def verify(U, V, W, n, m, p):
        """Check if the decomposition is valid."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W, n, m, p):
        """Check if the decomposition is valid using numpy."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    random.seed(42)
    np.random.seed(42)
    
    def try_rank_optimized(r, max_attempts=200):
        """Try to find a valid rank-r decomposition using optimization."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Gradient descent to minimize the error
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    return None
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # s = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            # ds/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                            # ds/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
                            # ds/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
                            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = s - 1.0
                            dU[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                            dV[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                            dW[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            if verify_np(U_rounded, V_rounded, W_rounded, n, m, p):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            # Try with different learning rates
            if attempt % 5 == 3:
                lr = 0.05
                for step in range(300):
                    if time.time() > deadline:
                        return None
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                                err = s - 1.0
                                dU[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                                dV[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                                dW[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify_np(U_rounded, V_rounded, W_rounded, n, m, p):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        return None
    
    def try_rank_structured(r, max_attempts=50):
        """Try structured initializations for rank-r decomposition."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with sparse structure
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Assign random non-zero entries
            for c in range(r):
                # Each column should have some non-zero entries
                for _ in range(np.random.randint(2, 5)):
                    U[np.random.randint(0, n*m), c] = np.random.choice([-1, -0.5, 0.5, 1])
                for _ in range(np.random.randint(2, 5)):
                    V[np.random.randint(0, m*p), c] = np.random.choice([-1, -0.5, 0.5, 1])
                for _ in range(np.random.randint(2, 5)):
                    W[np.random.randint(0, p*n), c] = np.random.choice([-1, -0.5, 0.5, 1])
            
            if verify_np(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Try with more structured initialization based on schoolbook
            if attempt % 10 == 0:
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                
                # Use a subset of schoolbook terms as starting point
                terms = list(range(n * m * p))
                random.shuffle(terms)
                for idx in range(min(r, len(terms))):
                    t = terms[idx]
                    i = t // (m * p)
                    rem = t % (m * p)
                    j = rem // p
                    k = rem % p
                    c = idx
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                
                # Perturb some entries
                for _ in range(10):
                    U[np.random.randint(0, n*m), np.random.randint(0, r)] = np.random.choice([-1, -0.5, 0, 0.5, 1])
                    V[np.random.randint(0, m*p), np.random.randint(0, r)] = np.random.choice([-1, -0.5, 0, 0.5, 1])
                    W[np.random.randint(0, p*n), np.random.randint(0, r)] = np.random.choice([-1, -0.5, 0, 0.5, 1])
                
                if verify_np(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try rank 11 first with optimization
    result = try_rank_optimized(11, max_attempts=30)
    if result is not None:
        best_result = result
        best_rank = 11
        return best_result
    
    # Try structured search for rank 11
    result = try_rank_structured(11, max_attempts=50)
    if result is not None:
        best_result = result
        best_rank = 11
        return best_result
    
    # Try rank 12
    result = try_rank_optimized(12, max_attempts=30)
    if result is not None:
        best_result = result
        best_rank = 12
        return best_result
    
    result = try_rank_structured(12, max_attempts=50)
    if result is not None:
        best_result = result
        best_rank = 12
        return best_result
    
    # If all else fails, return the best we have
    return best_result
# EVOLVE-BLOCK-END
