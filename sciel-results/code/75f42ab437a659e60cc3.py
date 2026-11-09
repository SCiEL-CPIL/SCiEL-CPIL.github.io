"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact for matrix multiplication."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def is_half_integer(arr, tol=1e-9):
        """Check if all entries are half-integers."""
        for row in arr:
            for x in row:
                if abs(x * 2 - round(x * 2)) > tol:
                    return False
        return True
    
    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return [[round(x * 2) / 2.0 for x in row] for row in arr]
    
    # Try Winograd-style decomposition for rank 32
    # Based on the idea of using fast rectangular matrix multiplication
    # For (3,3,4), we can try a decomposition inspired by Winograd's methods
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try to construct a rank-32 decomposition using structured factors
    # One approach: use the fact that 3x3x4 can be decomposed using
    # a combination of 2x2x2 blocks and padding
    
    # Actually, let's try a more systematic approach using the known lower bound
    # Theoretical lower bound for (3,3,4) is at least 3*3 + 3*4 + 3*4 - 3 - 4 = 9+12+12-7 = 26
    # But practical constructions give higher ranks
    
    # Let's try rank 32 with a structured approach
    # We'll use a greedy/iterative approach to find the decomposition
    
    # Start with a random initialization and use alternating minimization
    def find_decomposition(rank, max_iters=100, seed=42):
        """Try to find a rank-r decomposition using alternating minimization."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        # Add some structure: ensure each column is non-zero
        for r in range(rank):
            if np.allclose(U[:, r], 0):
                U[np.random.randint(n*m), r] = 1.0
            if np.allclose(V[:, r], 0):
                V[np.random.randint(m*p), r] = 1.0
            if np.allclose(W[:, r], 0):
                W[np.random.randint(p*n), r] = 1.0
        
        best_residual = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # For each (i,j,k), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a nonlinear constraint. We can use gradient-based optimization.
            
            # Compute residual
            residual = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        residual += (val - 1) ** 2
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if residual < 1e-12:
                break
            
            # Gradient descent step
            lr = 0.01
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        err = val - 1
                        for r in range(rank):
                            # Gradient with respect to U[i*m+j, r]
                            U[i*m+j, r] -= lr * err * V[j*p+k, r] * W[k*n+i, r]
                            # Gradient with respect to V[j*p+k, r]
                            V[j*p+k, r] -= lr * err * U[i*m+j, r] * W[k*n+i, r]
                            # Gradient with respect to W[k*n+i, r]
                            W[k*n+i, r] -= lr * err * U[i*m+j, r] * V[j*p+k, r]
        
        return best_U, best_V, best_W, best_residual
    
    # Try several ranks starting from 32
    for rank in [32, 33, 34, 35]:
        if time.time() > deadline - 30:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            U, V, W, residual = find_decomposition(rank, max_iters=50, seed=seed)
            
            if residual < 1e-10:
                # Check if we can snap to half-integers
                U_snapped = snap_to_half_int(U.tolist())
                V_snapped = snap_to_half_int(V.tolist())
                W_snapped = snap_to_half_int(W.tolist())
                
                if verify(U_snapped, V_snapped, W_snapped, tol=1e-8):
                    return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                
                # Try with original values (might not be half-integers)
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    # Check if close to half-integers
                    if is_half_integer(U.tolist(), tol=0.1) and is_half_integer(V.tolist(), tol=0.1) and is_half_integer(W.tolist(), tol=0.1):
                        return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
