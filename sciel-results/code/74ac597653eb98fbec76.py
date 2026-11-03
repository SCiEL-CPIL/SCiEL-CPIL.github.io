"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Build the multiplication tensor: T[i,j,k] = delta for (i,j,k)
    # The decomposition requires: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # And for all other triples, the sum should be 0.
    # Actually, the tensor is: T[i,j,k] = 1 if this is a valid (i,j,k) triple, else 0
    # But matrix mult tensor: T_{i,j,k}^{(a,b,c)} where a=i*m+j? No.
    # Let me reconsider. The tensor for C = A*B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has indices (i, j, k) and the decomposition is:
    # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_k V[j,k,c] * B[j,k]) ... no that's wrong.
    
    # Standard: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T_{i,k}^{j} = 1 for each (i,j,k)
    # Decomposition: C[i,k] = sum_c U[i,c] * V[k,c] * W[j,c] ... no.
    
    # Let me use the convention from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And this should be 0 for "invalid" combinations, but actually every (i,j,k) triple is valid
    # in the tensor. The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right either. The matrix multiplication tensor is:
    # T_{i,k}^{j} = 1, meaning the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So the tensor has three index groups: (i,j,k) and the value is 1 for each combination.
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This must equal 1 for ALL (i,j,k) in the valid range.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, max_time=100):
    """Try to find a rank-`target_rank` decomposition."""
    start_time = time.time()
    
    # Start from schoolbook and try to merge/remove terms
    # Use a greedy approach: try to express one term as a combination of others
    
    # Initialize with schoolbook
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # We want to find U, V, W with fewer columns such that this holds.
    
    # Approach: use random initialization and gradient-free optimization
    # Or: try to find a rank-r decomposition using the fact that the tensor
    # has structure.
    
    # Let's try a different approach: use known Strassen-like constructions
    # For 4x4 x 4x8, we can think of it as two 4x4 x 4x4 multiplications.
    # Strassen gives rank 7 for 2x2 x 2x2.
    # For 4x4, we can use Strassen recursively or other methods.
    
    # Actually, let's try a computational approach:
    # 1. Start with a random rank-r initialization
    # 2. Use alternating minimization to fit the tensor
    
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(20):
        if time.time() - start_time > max_time - 5:
            break
        
        # Random initialization
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Alternating minimization
        for iteration in range(50):
            if time.time() - start_time > max_time - 5:
                break
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            # For fixed i,j: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # Let M_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
            # Then sum_c U[i*m+j,c] * M_k[c] = 1 for all k
            # This is: U[i*m+j,:] @ M^T = 1 (vector of ones)
            # Where M has rows M_k for k=0..p-1
            
            for i in range(n):
                for j in range(m):
                    M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    # Solve: U[i*m+j, :] @ M.T = ones(p)
                    # This is underdetermined if target_rank < p, overdetermined if target_rank >= p
                    # Use least squares
                    try:
                        U[i*m+j, :], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    try:
                        V[j*p+k, :], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    try:
                        W[k*n+i, :], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    except:
                        pass
        
        # Check if we got a valid decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try various target ranks in increasing order
    # Theoretical lower bound: max(n*m, m*p, p*n) / something
    # For 4x4 x 4x8: lower bound is at least max(16, 32, 32) = 32
    # Known constructions might achieve around 70-90
    
    # Try from a reasonable estimate downward
    base_rank = n * m * p  # 128
    for target_rank in range(100, 30, -1):
        if time.time() > deadline - 20:
            break
        result = _try_rank_reduction(n, m, p, target_rank, max_time=min(30, deadline - time.time()))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
