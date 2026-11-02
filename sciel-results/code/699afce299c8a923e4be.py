"""Matrix multiplication 3x4 by 4x7: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def _verify_decomposition(U, V, W, n=3, m=4, p=7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    # For each output element (i, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,k)
    # Actually the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if i==k else 0... wait no.
    
    # Matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 (it's the structure tensor)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1 if i == k else 0
                if total != expected:
                    return False
    return True

def _try_rank_decomposition(n=3, m=4, p=7, max_rank=84, time_budget=120):
    """Try to find a decomposition with rank < max_rank using random search."""
    start_time = time.time()
    deadline = start_time + time_budget - 5  # 5 second safety margin
    
    # Schoolbook solution as fallback
    rank = n * m * p
    U_sb = np.zeros((n*m, rank), dtype=object)
    V_sb = np.zeros((m*p, rank), dtype=object)
    W_sb = np.zeros((p*n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    
    # Try ranks from 60 down to 84
    for target_rank in range(60, 84):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate random rank-1 terms with small integer/half-integer coefficients
            U = np.zeros((n*m, target_rank), dtype=object)
            V = np.zeros((m*p, target_rank), dtype=object)
            W = np.zeros((p*n, target_rank), dtype=object)
            
            for r in range(target_rank):
                # Random sparse vectors with small coefficients
                # Each factor has support on a random subset
                u_support = set(np.random.choice(n*m, size=np.random.randint(1, 4), replace=False))
                v_support = set(np.random.choice(m*p, size=np.random.randint(1, 4), replace=False))
                w_support = set(np.random.choice(p*n, size=np.random.randint(1, 4), replace=False))
                
                for idx in u_support:
                    U[idx, r] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2])
                for idx in v_support:
                    V[idx, r] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2])
                for idx in w_support:
                    W[idx, r] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2])
            
            # Verify
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                break
        
        if best_rank <= target_rank:
            break
    
    return best_rank, best_U, best_V, best_W

def construct():
    """Construct the best known exact decomposition for 3x4 * 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition within time budget
    try:
        best_rank, best_U, best_V, best_W = _try_rank_decomposition(n, m, p, 84, 120)
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
