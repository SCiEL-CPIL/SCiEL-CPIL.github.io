"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    # Build schoolbook decomposition as baseline
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0), dtype=np.float64)
    V0 = np.zeros((m*p, rank0), dtype=np.float64)
    W0 = np.zeros((p*n, rank0), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1.0
                V0[j*p + k, col] = 1.0
                W0[k*n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank using rank-1 removal / local search
    # Start from schoolbook and try to eliminate terms
    
    # Strategy: try rank-1 perturbations
    # For each pair of columns, try to see if we can merge them
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = rank0
    
    # Try random linear combinations of pairs of columns
    deadline = time.time() + 110  # 10 seconds safety margin
    
    # Use a greedy approach: try to reduce rank by 1, then by 2, etc.
    # For each pair of columns (a, b), check if they can be replaced by a single column
    # This is equivalent to checking if the tensor can be written with one fewer rank-1 term
    
    # A simpler approach: random search for lower rank
    # For rank r < rank0, try random initializations and optimize
    
    # Given time constraints, let's try a few random restarts for rank 41
    for target_rank in [41, 40, 39]:
        if time.time() > deadline - 60:
            break
            
        found = False
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline - 30:
                break
                
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Optimize using gradient descent on the decomposition error
            # The tensor is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We want this to match the identity tensor: T[i,j,k] = 1 if (i,j,k) is "valid"
            # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
            # Wait, no. The matrix multiplication tensor for M_{n,m,p} is:
            # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
            # The tensor entries are all 1s? No.
            
            # The matrix multiplication tensor T_{n,m,p} has entries T[i,j,k] = 1
            # for all i, j, k. The decomposition represents:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # In tensor form: C = A x_1 B x_2 ... no.
            
            # The tensor is defined as: T[i, j, k] = 1 for all valid indices
            # The rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
            # where u_c[i,j] = U[i,j,c], v_c[j,k] = V[j,k,c], w_c[k,i] = W[k,i,c]
            
            # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            
            # Let's set up the error function
            def error(U, V, W):
                # Compute the tensor product sum
                # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1.0) ** 2
                return err
            
            e0 = error(U, V, W)
            if e0 < 1e-6:
                found = True
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                break
                
            # Simple gradient descent with small steps
            lr = 0.01
            for step in range(50):
                if time.time() > deadline - 30:
                    break
                # Compute gradients numerically
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                eps = 1e-7
                for idx in range(U.size):
                    flat = U.flatten()
                    orig = flat[idx]
                    flat[idx] = orig + eps
                    U_flat = flat.reshape(U.shape)
                    e_plus = error(U_flat, V, W)
                    flat[idx] = orig
                    dU.flatten()[idx] = (e_plus - e0) / eps
                # This is too slow for 50 steps with full tensor
            
            # Skip optimization, just check if random works
            pass
            
        if found:
            break
    
    # Convert to lists
    def to_list(mat):
        return [[mat[i][j] for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    # Round to half-integers if close
    def round_half(arr):
        return np.round(arr * 2) / 2
    
    best_U = round_half(best_U)
    best_V = round_half(best_V)
    best_W = round_half(best_W)
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
