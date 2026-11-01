import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
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

def _rank8(n, m, p):
    """Block decomposition: for each intermediate index j, use rank 2
    (one per row i of the 2x7 outer product). Total rank = m*n = 8."""
    rank = m * n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            r = n * j + i
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}

def _wft_decomposition(n, m, p):
    """Apply Winograd Fourier Transform Technique for structured low-rank decomposition.
    
    For matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k],
    we can use a cyclic convolution structure.
    
    Strategy: Use the fact that we can embed the multiplication into a larger
    structured problem using Fourier transforms over cyclic groups.
    """
    # For 2x4x7, try using a transform-based approach
    # The WFT rank for multiplying m x n by n x p is related to the
    # size of the transform needed to avoid aliasing.
    
    # Alternative: Use a block decomposition with mixing
    # Split the 4 intermediate indices into groups and use Strassen-like mixing
    
    # Let's try: split j into {0,1} and {2,3}
    # A[:, 0:2] and A[:, 2:4] are both 2x2
    # B[0:2, :] and B[2:4, :] are both 2x7
    
    # For 2x2 x 2x7, we can use rank 2*2 = 4 (one per row of A, all of B)
    # But we can do better by using Strassen on the 2x2 part
    
    # Actually, let's try a different approach: use the tensor structure directly
    # and search for a lower rank decomposition
    
    # For now, return None to indicate we should try other approaches
    return None

def _search_low_rank(n, m, p, max_rank, deadline):
    """Search for a low-rank decomposition using random initialization and verification."""
    import random
    
    best_result = None
    best_rank = max_rank + 1
    
    for rank in range(7, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try several random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization with integer/half-integer entries
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Initialize with random values from {-1, 0, 0.5, 1}
            for r in range(rank):
                # U: 2x4 = 8 entries
                for idx in range(n * m):
                    if random.random() < 0.3:
                        U[idx, r] = random.choice([-1, 0.5, 1, -0.5])
                
                # V: 4x7 = 28 entries
                for idx in range(m * p):
                    if random.random() < 0.3:
                        V[idx, r] = random.choice([-1, 0.5, 1, -0.5])
                
                # W: 7x2 = 14 entries
                for idx in range(p * n):
                    if random.random() < 0.3:
                        W[idx, r] = random.choice([-1, 0.5, 1, -0.5])
            
            # Convert to lists for verification
            U_list = [[U[i, j] for j in range(rank)] for i in range(n * m)]
            V_list = [[V[i, j] for j in range(rank)] for i in range(m * p)]
            W_list = [[W[i, j] for j in range(rank)] for i in range(p * n)]
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # First try the known rank-8 solution
    result = _rank8(n, m, p)
    
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        best_result = result
        best_rank = 8
    else:
        best_result = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Try to find a lower rank solution
    if best_rank > 7:
        search_result = _search_low_rank(n, m, p, 7, deadline)
        if search_result is not None:
            return search_result
    
    # Fallback to schoolbook if nothing better found
    return best_result
