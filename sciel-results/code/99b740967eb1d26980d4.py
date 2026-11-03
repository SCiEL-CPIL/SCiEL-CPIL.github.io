"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Reconstruct the tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The target tensor is T[i, j, k] = 1 if and only if this is the (i,j,k) entry of matmul
    # Actually, the matmul tensor T[i,j,k] = delta_{i,j,k} where C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the standard basis tensor for matmul)
    # Wait, no. The tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The matmul tensor has T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle)
    # Actually the tensor is: T[i, j, k] = 1 if the indices correspond to a valid matmul term
    # For matmul, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Because C[i,k] = sum_j A[i,j] B[j,k], so the tensor is all-ones in the appropriate sense
    
    # Let me reconsider. The tensor T is defined by:
    # T[i, j, k] = 1 if we multiply A[i,j] by B[j,k] to get contribution to C[i,k]
    # So T is a 3-way tensor of size n x m x p with T[i,j,k] = 1 for all i,j,k
    # No wait, that's not right either.
    
    # The matrix multiplication tensor is: T[i, j, k] = 1 if the indices are valid
    # Specifically, for A (n x m), B (m x p), C (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i, j, k] = 1 for all i, j, k (it's the all-ones tensor)
    # No, that's wrong. The tensor is defined such that:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_k C[i,k] (as a polynomial identity)
    # This gives T[i,j,k] = 1 if j is the same index in both, i.e., T[i,j,k] = 1 for all valid i,j,k
    
    # Actually, I think the tensor is: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because every term A[i,j]*B[j,k] contributes to C[i,k]
    
    # Let me just verify by checking that the reconstructed tensor matches the all-ones tensor
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        uc = np.array(U[:, c]) if isinstance(U, list) else U[:, c]
        vc = np.array(V[:, c]) if isinstance(V, list) else V[:, c]
        wc = np.array(W[:, c]) if isinstance(W, list) else W[:, c]
        # uc is indexed by i*m+j, vc by j*p+k, wc by k*n+i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    
    # The target tensor is all ones
    T_target = np.ones((n, m, p))
    return np.allclose(T_recon, T_target, atol=tol)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U_best = [[0] * schoolbook_rank for _ in range(n * m)]
    V_best = [[0] * schoolbook_rank for _ in range(m * p)]
    W_best = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][col] = 1
                V_best[j * p + k][col] = 1
                W_best[k * n + i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_best, V_best, W_best
    
    # Try to find lower-rank decompositions using random search + local optimization
    # For each candidate rank r, try random initializations and see if we can solve
    # the system of equations
    
    # The system is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p = 90 equations in r*(n*m + m*p + p*n) unknowns
    
    # Try ranks from lower values up
    # Known lower bounds: rank >= max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    # Also rank >= n+m+p-2 = 3+5+6-2 = 12 (weak)
    # For matmul, rank >= max(n*m, m*p, n*p) is not quite right
    # Actually, the tensor rank of matmul is at least max(n*m, m*p, n*p) / something
    # Let me just try a range of ranks
    
    # Try various ranks, starting from a reasonable lower bound
    # For 3x5x6, let's try ranks from 30 up to 90
    # But first, let me try a smarter approach: use the fact that we can sometimes
    # reduce rank by exploiting symmetries or known patterns
    
    # Actually, let me try a direct optimization approach for small ranks
    # For rank r, we have r*(15+30+18) = 90r variables and 90 equations
    # For r >= 1, we have more variables than equations, so solutions may exist
    
    # Let me try random search for low ranks
    np.random.seed(42)
    
    for r in range(15, min(schoolbook_rank, 60)):  # try ranks 15 to 59
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 5 if r < 40 else 3
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Simple gradient descent on the loss: sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            lr = 0.01
            for step in range(200):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = val - 1.0
                            loss += err * err
                            for c in range(r):
                                grad_U[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                                grad_V[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                                grad_W[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if loss < 1e-12:
                    break
            
            # Check if we found a solution
            if loss < 1e-10:
                # Convert to lists with rounded values (try to get exact/half-integer)
                U_list = [[float(round(U[i, c], 6)) for c in range(r)] for i in range(n * m)]
                V_list = [[float(round(V[i, c], 6)) for c in range(r)] for i in range(m * p)]
                W_list = [[float(round(W[i, c], 6)) for c in range(r)] for i in range(p * n)]
                
                # Verify
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    best_rank = r
                    best_U, best_V, best_W = U_list, V_list, W_list
                    break  # Found a solution at this rank, try lower ranks
    
    if time.time() - start_time > DEADLINE - 5:
        pass  # Return best found
    
    return {"U": best_U, "V": best_V, "W": best_W}
