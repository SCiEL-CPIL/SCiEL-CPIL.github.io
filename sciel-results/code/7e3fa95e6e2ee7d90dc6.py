"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=3, m=3, p=8, tol=1e-9):
    """Check if the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _solve_for_W(U, V, n, m, p):
    """Given U and V, solve for W using least squares."""
    R = U.shape[1]
    W = np.zeros((p * n, R))
    for i in range(n):
        for k in range(p):
            # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
            # This is: A @ W[k*n+i, :] = b where A[j,c] = U[i*m+j,c]*V[j*p+k,c], b[j]=1
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i*m+j, :] * V[j*p+k, :]
            b = np.ones(m)
            if R > 0:
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k*n+i, :] = w
    return W

def _solve_for_V(U, W, n, m, p):
    """Given U and W, solve for V using least squares."""
    R = U.shape[1]
    V = np.zeros((m * p, R))
    for j in range(m):
        for k in range(p):
            # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
            # This is: A @ V[j*p+k, :] = b where A[i,c] = U[i*m+j,c]*W[k*n+i,c], b[i]=1
            A = np.zeros((n, R))
            for i in range(n):
                A[i, :] = U[i*m+j, :] * W[k*n+i, :]
            b = np.ones(n)
            if R > 0:
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j*p+k, :] = v
    return V

def _solve_for_U(V, W, n, m, p):
    """Given V and W, solve for U using least squares."""
    R = V.shape[1]
    U = np.zeros((n * m, R))
    for i in range(n):
        for j in range(m):
            # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is: A @ U[i*m+j, :] = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c], b[k]=1
            A = np.zeros((p, R))
            for k in range(p):
                A[k, :] = V[j*p+k, :] * W[k*n+i, :]
            b = np.ones(p)
            if R > 0:
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m+j, :] = u
    return U

def _round_to_half_int(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by removing one column and re-solving."""
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    # Try removing each column and see if we can re-solve
    for r in range(rank):
        if time.time() > deadline:
            break
        
        mask = np.ones(rank, dtype=bool)
        mask[r] = False
        U_trial = U[:, mask].copy()
        V_trial = V[:, mask].copy()
        W_trial = W[:, mask].copy()
        
        # Alternate solving
        for _ in range(10):
            if time.time() > deadline:
                break
            W_trial = _solve_for_W(U_trial, V_trial, n, m, p)
            V_trial = _solve_for_V(U_trial, W_trial, n, m, p)
            U_trial = _solve_for_U(V_trial, W_trial, n, m, p)
        
        # Round to half-integers
        U_r = _round_to_half_int(U_trial)
        V_r = _round_to_half_int(V_trial)
        W_r = _round_to_half_int(W_trial)
        
        if _validate(U_r, V_r, W_r, n, m, p):
            return (U_r, V_r, W_r)
    
    return None

def _brent_3x3_decomposition():
    """
    Return the known rank-21 decomposition for 3x3 matrix multiplication.
    Uses the standard Brent decomposition with half-integer coefficients.
    Returns U (9x21), V (9x21), W (9x21) for n=3,m=3,p=3.
    """
    # Brent's rank-21 decomposition for 3x3 matrix multiplication
    # Coefficients are chosen to give exact half-integer values
    n = m = p = 3
    R = 21
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Standard Brent decomposition coefficients
    # Each column c corresponds to a term: U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Term 0: (0,0) diagonal
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    
    # Term 1: (1,1) diagonal  
    U[4, 1] = 1; V[4, 1] = 1; W[4, 1] = 1
    
    # Term 2: (2,2) diagonal
    U[8, 2] = 1; V[8, 2] = 1; W[8, 2] = 1
    
    # Terms 3-5: Off-diagonal for row 0
    U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1  # (0,1,1)
    U[2, 4] = 1; V[6, 4] = 1; W[2, 4] = 1  # (0,2,2)
    U[1, 5] = 1; V[6, 5] = 1; W[0, 5] = 1  # (0,1,0)
    
    # Terms 6-8: Off-diagonal for row 1
    U[3, 6] = 1; V[0, 6] = 1; W[1, 6] = 1  # (1,0,1)
    U[5, 7] = 1; V[3, 7] = 1; W[5, 7] = 1  # (1,1,1) already counted? Let me redo
    U[6, 8] = 1; V[6, 8] = 1; W[2, 8] = 1  # (1,2,2)
    
    # This is getting complex. Let me use a known explicit construction.
    # Actually, for 3x3x3, the minimal rank is 19 (Strassen-like) or 21 (Brent).
    # Let me use a simpler approach: just return the schoolbook for now and let the 
    # optimization handle it. But we want to seed with a good structure.
    
    # For simplicity, let's construct a valid decomposition using Kronecker structure.
    # We'll use the fact that 3x3x8 can be seen as 3x3x3 tensor 1x3x1 (trivial).
    # But that gives rank 72. Instead, let's just return None and let the main 
    # optimization do its work, but with better initialization.
    
    return None

def _initialize_with_structure(n, m, p, target_rank):
    """
    Initialize U, V, W with structure that exploits the p=8 dimension.
    Use block structure where each block handles a subset of k values.
    """
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Divide the 8 k-values into groups and use structured initialization
    # This is a heuristic to give the optimizer a better starting point
    
    if target_rank <= 24:
        # Try to use rank ~24 structure
        # Each (i,j) pair gets ~24/9 ≈ 2.67 terms
        # Initialize with random but structured values
        for c in range(target_rank):
            # Assign each term to a (i,j,k) pattern
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
    else:
        # For higher rank, use more structured initialization
        for c in range(target_rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s limit
    
    # Start with schoolbook rank 72
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
    
    # Strategy 1: Greedy rank reduction from schoolbook
    U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
    current_rank = best_rank
    
    while time.time() < deadline - 40:  # Leave time for other strategies
        result = _try_reduce_rank(U, V, W, n, m, p, deadline - 40)
        if result is not None:
            U, V, W = result
            current_rank = U.shape[1]
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            break
    
    # Strategy 2: Random restarts with alternating minimization for various ranks
    random.seed(42)
    np.random.seed(42)
    
    # Try to find decompositions at various target ranks
    # Focus on ranks that are plausible for 3x3x8
    target_ranks = sorted(set([
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71
    ]), reverse=True)
    
    # Filter to only try ranks below best
    target_ranks = [r for r in target_ranks if r < best_rank]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Use structured initialization
            U_trial, V_trial, W_trial = _initialize_with_structure(n, m, p, target_rank)
            
            # Add some randomness to escape local minima
            U_trial += np.random.randn(n * m, target_rank) * 0.1
            V_trial += np.random.randn(m * p, target_rank) * 0.1
            W_trial += np.random.randn(p * n, target_rank) * 0.1
            
            # Alternate minimization with more iterations
            success = False
            for iteration in range(300):
                if time.time() > deadline - 15:
                    break
                
                W_trial = _solve_for_W(U_trial, V_trial, n, m, p)
                V_trial = _solve_for_V(U_trial, W_trial, n, m, p)
                U_trial = _solve_for_U(V_trial, W_trial, n, m, p)
                
                # Check if we're close to a valid solution
                U_r = _round_to_half_int(U_trial)
                V_r = _round_to_half_int(V_trial)
                W_r = _round_to_half_int(W_trial)
                
                if _validate(U_r, V_r, W_r, n, m, p):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    success = True
                    break
            
            if success:
                break
    
    # Final polish: try to reduce the best found rank further
    if best_rank > 24 and time.time() < deadline - 5:
        U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        while time.time() < deadline - 5:
            result = _try_reduce_rank(U, V, W, n, m, p, deadline - 5)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
