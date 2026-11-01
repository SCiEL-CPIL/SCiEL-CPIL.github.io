"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import itertools
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] for output (i,k)."""
    # T[i,j,k] = 1 if output (i,k) depends on input (i,j) and (j,k)
    # Actually, the tensor for matmul: T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # The multiplication tensor M[i,j,k] = delta[i,j] * delta[j,k] in Einstein notation
    # But for our decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should give the product
    # Let's think of it as: C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j]*B[j,k] for each (i,k)
    # The tensor T has indices (i*m+j, j*p+k, k*n+i) and value 1 when the term contributes
    # Actually, the correct formulation:
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to equal 1 when (i,j,k) forms a valid matmul term
    # The matmul tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # For each (i,j,k), the term is 1
    
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the tensor: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Efficient computation using einsum
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_true = _build_tensor(n, m, p)
    
    diff = np.abs(T_recon - T_true)
    return np.max(diff) < tol, np.max(diff)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank(n, m, p, rank, time_budget=10.0):
    """Try to find a decomposition of the given rank."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    # Start with a random initialization based on schoolbook terms
    # We'll use a subset of schoolbook terms and try to combine them
    n_schoolbook = n * m * p
    if rank > n_schoolbook:
        return None  # Schoolbook is better
    
    # Strategy: Start with schoolbook, then try to reduce rank by finding linear dependencies
    # For now, try a local search approach
    
    # Initialize with first 'rank' schoolbook terms
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Use all schoolbook terms if rank >= n*m*p
    if rank >= n_schoolbook:
        for col in range(rank):
            i = col // (m * p)
            j = (col // p) % m
            k = col % p
            U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
    else:
        # For lower rank, we need a different approach
        # Try random linear combinations
        for col in range(rank):
            # Randomly select terms and combine
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.5:
                            U[i * m + j, col] += random.gauss(0, 1)
                            V[j * p + k, col] += random.gauss(0, 1)
                            W[k * n + i, col] += random.gauss(0, 1)
    
    # Local search: try to improve the decomposition
    best_diff = np.inf
    for iteration in range(100):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.max(np.abs(T_recon - T))
        
        if diff < best_diff:
            best_diff = diff
            if best_diff < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
        
        # Try a random perturbation
        col = random.randint(0, rank - 1)
        if random.random() < 0.3:
            # Perturb U
            idx = random.randint(0, n * m - 1)
            U[idx, col] += random.gauss(0, 0.5)
        elif random.random() < 0.3:
            # Perturb V
            idx = random.randint(0, m * p - 1)
            V[idx, col] += random.gauss(0, 0.5)
        else:
            # Perturb W
            idx = random.randint(0, p * n - 1)
            W[idx, col] += random.gauss(0, 0.5)
    
    # Check final result
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    diff = np.max(np.abs(T_recon - T))
    if diff < 1e-10:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication of sizes 2x3 and 3x7.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # First, get the schoolbook solution as a fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p  # 42
    best_result = (U_school, V_school, W_school)
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Try ranks from lower bound up to schoolbook rank
    for rank in range(max(n*m, m*p, p*n), best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, time_budget=20.0)
            if result is not None:
                U, V, W = result
                # Verify
                valid, diff = _verify(U, V, W, n, m, p)
                if valid:
                    best_rank = rank
                    best_result = (U, V, W)
                    break  # Found a valid decomposition at this rank, try lower
    
    # Return the best result found
    U, V, W = best_result
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
