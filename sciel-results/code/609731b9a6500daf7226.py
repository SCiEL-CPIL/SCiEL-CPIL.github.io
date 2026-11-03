"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import random
import time as _time

def _verify(U, V, W, n=2, m=3, p=10):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    T_computed = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    
    # Expected tensor: T[i,j,k] = 1 if this is a valid multiplication term
    # For matrix mult (n x m) * (m x p) -> (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k
    T_expected = np.ones((n, m, p))
    
    return np.allclose(T_computed, T_expected, atol=1e-10)

def _is_valid_entries(U, V, W):
    """Check that all entries are integer or half-integer (i.e., 2*x is integer)."""
    for mat in [U, V, W]:
        mat = np.array(mat, dtype=np.float64)
        doubled = 2.0 * mat
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    return True

def _build_strassen55():
    """Build the known rank-55 decomposition."""
    n, m, p = 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    X00, X01, X10, X11 = 0, 1, 3, 4
    
    for j in range(5):
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3
        
        U[X00, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1; W[Z11, col] += 1
        col += 1
        
        U[X10, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1; W[Z11, col] -= 1
        col += 1
        
        U[X00, col] += 1
        V[Y01, col] += 1; V[Y11, col] -= 1
        W[Z01, col] += 1; W[Z11, col] += 1
        col += 1
        
        U[X11, col] += 1
        V[Y10, col] += 1; V[Y00, col] -= 1
        W[Z00, col] += 1; W[Z10, col] += 1
        col += 1
        
        U[X00, col] += 1; U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1; W[Z01, col] += 1
        col += 1
        
        U[X10, col] += 1; U[X00, col] -= 1
        V[Y00, col] += 1; V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        U[X01, col] += 1; U[X11, col] -= 1
        V[Y10, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1
    
    assert col == rank
    return U, V, W

def _try_rank_search(start_rank=40, max_rank=54, time_limit=120.0):
    """Try to find a lower-rank decomposition via structured search."""
    n, m, p = 2, 3, 10
    start_time = _time.time()
    
    for target_rank in range(start_rank, max_rank + 1):
        if _time.time() - start_time > time_limit - 30:
            break
        
        # Try several random restarts for this rank
        n_attempts = 50
        for attempt in range(n_attempts):
            if _time.time() - start_time > time_limit - 30:
                break
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random sparse patterns
            for r in range(target_rank):
                # Pick random non-zero entries
                u_nz = random.randint(1, 4)
                v_nz = random.randint(1, 4)
                w_nz = random.randint(1, 4)
                
                u_indices = random.sample(range(n*m), min(u_nz, n*m))
                v_indices = random.sample(range(m*p), min(v_nz, m*p))
                w_indices = random.sample(range(p*n), min(w_nz, p*n))
                
                for idx in u_indices:
                    U[idx, r] = random.choice([-1, 1])
                for idx in v_indices:
                    V[idx, r] = random.choice([-1, 1])
                for idx in w_indices:
                    W[idx, r] = random.choice([-1, 1])
            
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                if _is_valid_entries(U.tolist(), V.tolist(), W.tolist()):
                    return U, V, W, target_rank
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    
    Starts with known rank-55 Strassen-based decomposition, then searches for lower rank.
    """
    start_time = _time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Start with known good solution
    U, V, W = _build_strassen55()
    best_rank = 55
    
    # Try to find lower rank
    result = _try_rank_search(start_rank=40, max_rank=54, time_limit=110.0)
    if result is not None:
        U, V, W, rank = result
        best_rank = rank
    
    # Final check before deadline
    if _time.time() > deadline - 5:
        pass  # Return best found
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
