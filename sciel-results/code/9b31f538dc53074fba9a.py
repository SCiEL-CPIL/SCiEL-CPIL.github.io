"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # First, create the schoolbook solution as fallback (Rank 12)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    deadline = time.time() + 140  # Leave 10s margin
    
    # Try to find lower rank decompositions
    # Known lower bound is 6, known rank is 8.
    # We try 8 first as it is the known minimal rank.
    for r in [8, 7, 6]:
        if time.time() > deadline:
            break
        
        result = _find_decomposition(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break  # We found the minimum rank, stop
    
    return best_result


def _find_decomposition(n, m, p, r, deadline):
    """
    Try to find a rank-r decomposition.
    Uses random initialization and gradient descent / ALS.
    """
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Iteratively refine using Alternating Least Squares (ALS)
        # We solve for one matrix at a time while fixing the others.
        # The system is: sum_c U_a,c V_b,c W_c,c = 1
        # Note: indices are flattened.
        
        for it in range(500):
            if time.time() > deadline:
                return None
            
            # Update W using least squares
            # For each (i,k), we have equation: sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
            # This is not a simple linear system for W because of sum_j.
            # However, we can use gradient descent on the loss function.
            
            # Loss L = sum_{i,j,k} (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1)^2
            # Gradient w.r.t W[k*n+i,c]:
            # dL/dW[k*n+i,c] = 2 * sum_j (val_{i,j,k} - 1) * U[i*m+j,c] * V[j*p+k,c]
            
            # Calculate current values
            vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vals[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            
            # Check convergence
            loss = np.sum((vals - 1.0)**2)
            if loss < 1e-8:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                loss_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                            loss_rounded += (val - 1.0)**2
                            
                if loss_rounded < 1e-6:
                    U_list = [[float(U_rounded[i, c]) for c in range(r)] for i in range(n*m)]
                    V_list = [[float(V_rounded[i, c]) for c in range(r)] for i in range(m*p)]
                    W_list = [[float(W_rounded[i, c]) for c in range(r)] for i in range(p*n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = vals[i, j, k] - 1.0
                        # dL/dU[i*m+j,c] = 2 * err * V[j*p+k,c] * W[k*n+i,c]
                        grad_U[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                        # dL/dV[j*p+k,c] = 2 * err * U[i*m+j,c] * W[k*n+i,c]
                        grad_V[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                        # dL/dW[k*n+i,c] = 2 * err * U[i*m+j,c] * V[j*p+k,c]
                        grad_W[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
            
            # Simple gradient descent step with adaptive learning rate
            # We can use a fixed small step or normalize gradients
            step_size = 0.01
            U -= step_size * grad_U
            V -= step_size * grad_V
            W -= step_size * grad_W
            
            # Optional: normalize to prevent explosion
            # U /= np.linalg.norm(U) * np.sqrt(n*m*r)
            # V /= np.linalg.norm(V) * np.sqrt(m*p*r)
            # W /= np.linalg.norm(W) * np.sqrt(p*n*r)
            
    return None
# EVOLVE-BLOCK-END
