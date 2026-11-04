"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Helper to create schoolbook decomposition
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Helper to verify decomposition exactly using rational arithmetic
    def verify_exact(U, V, W, n, m, p, tol=1e-10):
        """Verify that the decomposition is exact."""
        # Check all 1-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check a sample of 0-entries
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (j == j2 and i == i2 and k == k2):
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        
        return True
    
    # Helper to compute tensor error (Frobenius norm of difference)
    def tensor_error(U, V, W, n, m, p):
        """Compute the error in the tensor decomposition."""
        # Build the target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Compute the approximated tensor
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Efficient computation using einsum
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        return np.linalg.norm(T_approx - T)
    
    # Helper to convert to fraction list
    def to_fraction_list(matrix):
        result = []
        for row in matrix:
            frac_row = []
            for val in row:
                if abs(val) < 1e-10:
                    frac_row.append(0)
                else:
                    for denom in [1, 2, 4, 8, 16]:
                        numer = round(val * denom)
                        if abs(val - numer / denom) < 1e-6:
                            frac_row.append(numer / denom)
                            break
                    else:
                        frac_row.append(val)
            result.append(frac_row)
        return result
    
    # Initialize with schoolbook
    U, V, W, rank = make_schoolbook(n, m, p)
    best_result = (U.copy(), V.copy(), W.copy())
    best_rank = rank
    
    # Strategy: Use hierarchical decomposition exploiting commutative ring structure
    # For 2x5x6, we can decompose using polynomial multiplication structure
    
    # Try to find lower rank decompositions using random initialization + optimization
    for target_rank in range(min(30, n * m * p), 1, -1):
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Multiple random restarts for each rank
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(target_rank * 100 + restart)
            
            # Random initialization
            U_r = np.random.randn(n * m, target_rank) * 0.5
            V_r = np.random.randn(m * p, target_rank) * 0.5
            W_r = np.random.randn(p * n, target_rank) * 0.5
            
            # Gradient-based optimization
            # The objective is to minimize ||T_approx - T||_F^2
            # where T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradients
                # dL/dU[a,r] = sum_{b,c} (T_approx[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
                # We can compute this efficiently
                
                # Compute T_approx
                T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                
                # Compute error
                error = T_approx - tensor_error_target(n, m, p)
                
                # Compute gradients
                grad_U = np.einsum('abc,br,cr->ar', error, V_r, W_r)
                grad_V = np.einsum('abc,ar,cr->br', error, U_r, W_r)
                grad_W = np.einsum('abc,ar,br->cr', error, U_r, V_r)
                
                # Update
                U_r -= lr * grad_U
                V_r -= lr * grad_V
                W_r -= lr * grad_W
            
            # Check if we found a valid decomposition
            err = tensor_error(U_r, V_r, W_r, n, m, p)
            if err < 1e-8:
                # Verify exactly
                if verify_exact(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_result = (U_r.copy(), V_r.copy(), W_r.copy())
                    break
        
        if best_rank <= target_rank:
            break
    
    # Final polish: try to clean up the solution
    if best_rank < n * m * p:
        U_final, V_final, W_final = best_result
        # Try to snap to rational values
        for mat in [U_final, V_final, W_final]:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    for denom in [1, 2, 4, 8]:
                        numer = round(mat[i, j] * denom)
                        if abs(mat[i, j] - numer / denom) < 1e-4:
                            mat[i, j] = numer / denom
                            break
    
    U_final, V_final, W_final = best_result
    return {"U": to_fraction_list(U_final), "V": to_fraction_list(V_final), "W": to_fraction_list(W_final)}

def tensor_error_target(n, m, p):
    """Create the target multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T
# EVOLVE-BLOCK-END
