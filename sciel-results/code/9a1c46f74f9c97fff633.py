import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130  # safety margin before 150s
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Build the matrix multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # Shape: (n*m, m*p, p*n) = (9, 18, 18)
    
    # For rank r decomposition:
    # C = A*B where A is n×m, B is m×p, C is n×p
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i*m+j, j*p+k, k*n+i]
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 when indices are consistent
    # Specifically, for the (i,j) entry of A, (j,k) entry of B, (i,k) entry of C
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # Otherwise T[a,b,c] = 0
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Try to find low rank decomposition
    # Start with known upper bound
    
    # For 3x3x3, optimal rank is 23. For 3x3x6, upper bound is 46.
    # Let's try to find a decomposition with rank <= 46
    
    best_result = None
    best_rank = n * m * p  # schoolbook = 54
    
    # First, get schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_result = (U_sb, V_sb, W_sb, rank_schoolbook)
    
    # Try rank 46 (two copies of 3x3x3 optimal)
    # For 3x3x3, I'll use a known rank-23 decomposition
    # Actually, let me try to find a good decomposition using optimization
    
    # Try various ranks from 40 up to 54
    for rank in range(40, 55):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition
        result = _try_rank_decomposition(target, n, m, p, rank, deadline - (time.time() - start_time))
        if result is not None:
            U, V, W = result
            best_result = (U, V, W, rank)
            break  # Found lower rank, stop
    
    if best_result is None:
        U, V, W, rank = best_result
    else:
        U, V, W, rank = best_result
    
    # Convert to lists
    U_list = [[float(U[i,j]) for j in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i,j]) for j in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i,j]) for j in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_rank_decomposition(target, n, m, p, rank, time_budget):
    """Try to find a rank-r decomposition of the matrix multiplication tensor."""
    start = time.time()
    
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Initialize with random or structured values
    np.random.seed(42)
    
    # Use a structured initialization based on schoolbook
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    # Initialize using a subset of schoolbook terms
    if rank >= nm * mp * pn:
        # Full schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return (U, V, W)
    
    # For lower rank, use random initialization with some structure
    for c in range(rank):
        # Randomly assign some non-zero entries
        U[:, c] = np.random.randn(nm) * 0.1
        V[:, c] = np.random.randn(mp) * 0.1
        W[:, c] = np.random.randn(pn) * 0.1
    
    # Use iterative optimization
    lr = 0.01
    for iteration in range(200):
        if time.time() - start > time_budget:
            break
        
        # Compute gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # T_decomposed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Gradient w.r.t U[a,r]: sum_{b,c} target[a,b,c]*V[b,r]*W[c,r] - sum_{b,c} T_decomp[a,b,c]*V[b,r]*W[c,r]
        # But we only care about the difference
        
        # Compute current decomposition
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        error = target - T_decomp
        
        # Gradient for U
        grad_U = np.einsum('abc,br,cr->ar', error, V, W)
        grad_V = np.einsum('abc,ar,cr->br', error, U, W)
        grad_W = np.einsum('abc,ar,br->cr', error, U, V)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        if iteration % 20 == 0:
            err_norm = np.linalg.norm(error)
            if err_norm < 1e-6:
                # Verify and return
                T_check = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_check, target, atol=1e-5):
                    return (U, V, W)
    
    # Final check
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    if np.allclose(T_check, target, atol=1e-4):
        return (U, V, W)
    
    return None

# EVOLVE-BLOCK-END
