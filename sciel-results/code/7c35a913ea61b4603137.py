import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2x2():
    """Return U, V, W for rank-7 Strassen decomposition of 2x2x2."""
    # Standard Strassen: 7 products
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U factors (for A): U[i*2+j][c]
    # V factors (for B): V[j*2+k][c]
    # W factors (for C): W[k*2+i][c]
    
    U = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    V = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    W = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1  # A11, A22
    V[0][0] = 1; V[3][0] = 1  # B11, B22
    W[0][0] = 1; W[3][0] = 1  # C11, C22 (M1 contributes to both)
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1  # A21, A22
    V[0][1] = 1  # B11
    W[2][1] = 1  # C21
    
    # M3 = A11(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1; V[3][2] = -1  # B12, -B22
    W[1][2] = 1  # C12
    
    # M4 = A22(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1; V[0][3] = -1  # B21, -B11
    W[0][3] = 1; W[2][3] = 1  # C11, C21
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1  # A11, A12
    V[3][4] = 1  # B22
    W[0][4] = -1; W[1][4] = 1  # -C11, C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1  # A21, -A11
    V[0][5] = 1; V[1][5] = 1  # B11, B12
    W[3][5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1  # A12, -A22
    V[2][6] = 1; V[3][6] = 1  # B21, B22
    W[0][6] = 1  # C11
    
    return U, V, W

def _try_block_decomposition(n, m, p, deadline):
    """
    Try to construct a decomposition by partitioning into blocks.
    Use Strassen for 2x2x2 sub-blocks and schoolbook for the rest.
    """
    if time.time() > deadline:
        return None
    
    # Strategy: use Strassen for all possible 2x2x2 blocks
    # For 3x5x6, we can identify overlapping or non-overlapping 2x2x2 blocks
    
    # Let's try a simple approach: use Strassen for the first 2x2x2 block
    # and schoolbook for everything else
    if n < 2 or m < 2 or p < 2:
        return None
    
    schoolbook_count = n * m * p - 8
    strassen_rank = 7
    total_rank = schoolbook_count + strassen_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Assign Strassen columns for the 2x2x2 block (i<2, j<2, k<2)
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = strassen_U[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j * p + k][c] = strassen_V[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k * n + i][c] = strassen_W[k * 2 + i][c]
    
    # Assign schoolbook columns for all other (i,j,k)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _try_multi_strassen(n, m, p, deadline):
    """
    Try to use multiple Strassen blocks if dimensions allow.
    """
    if time.time() > deadline:
        return None
    
    # For 3x5x6, we could potentially use:
    # - One 2x2x2 block (rank 7 instead of 8)
    # - Check if we can fit another 2x2x2 block
    
    # The 3x5x6 tensor has indices i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    # We can place non-overlapping 2x2x2 blocks at different positions
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} - already done
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Block 3: i in {0,1}, j in {4,?}, ... - can't fit since j only goes to 4
    # Block 4: i in {2,?}, ... - can't fit since i only goes to 2
    
    # Let's try two non-overlapping 2x2x2 blocks
    # Block 1: i<2, j<2, k<2
    # Block 2: i<2, j in {2,3}, k in {2,3}
    
    if n < 2 or m < 4 or p < 4:
        return None
    
    # Count schoolbook elements
    # Total: 3*5*6 = 90
    # Block 1 covers 8 elements, Block 2 covers 8 elements
    # Schoolbook: 90 - 16 = 74
    # Strassen: 7 + 7 = 14
    # Total rank: 74 + 14 = 88
    
    schoolbook_count = n * m * p - 16
    total_rank = schoolbook_count + 14
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = strassen_U[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j * p + k][c] = strassen_V[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k * n + i][c] = strassen_W[k * 2 + i][c]
    
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    for c in range(7):
        col = 7 + c
        for i in range(2):
            for j in range(2, 4):
                U[i * m + j][col] = strassen_U[i * 2 + (j - 2)][c]
        for j in range(2, 4):
            for k in range(2, 4):
                V[j * p + k][col] = strassen_V[(j - 2) * 2 + (k - 2)][c]
        for k in range(2, 4):
            for i in range(2):
                W[k * n + i][col] = strassen_W[(k - 2) * 2 + i][c]
    
    # Schoolbook for remaining elements
    col = 14
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if in Block 1
                if i < 2 and j < 2 and k < 2:
                    continue
                # Check if in Block 2
                if i < 2 and 2 <= j < 4 and 2 <= k < 4:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _try_three_strassen(n, m, p, deadline):
    """
    Try three non-overlapping 2x2x2 Strassen blocks.
    """
    if time.time() > deadline:
        return None
    
    # Block 1: i<2, j<2, k<2
    # Block 2: i<2, j in {2,3}, k in {2,3}
    # Block 3: i<2, j=4, k in {4,5} - but this is 1x1x2, not 2x2x2
    # Can't fit a third 2x2x2 block with i<2 since j only has 5 elements
    
    # Alternative: use different i ranges
    # Block 1: i<2, j<2, k<2
    # Block 2: i<2, j in {2,3}, k in {2,3}  
    # Block 3: i=2, j<2, k<2 - but this is 1x2x2, not 2x2x2
    
    # For 3x5x6, max two non-overlapping 2x2x2 blocks with i<2
    return None

def _try_gradient_descent(n, m, p, rank, max_iters=5000, seed=42, lr_init=0.01, deadline=None):
    """
    Try to find a rank-`rank` decomposition via gradient descent.
    """
    if deadline and time.time() > deadline:
        return None
    
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    scale = (rank / (n * m * p)) ** (1/3)
    U = U / scale
    V = V / scale
    W = W / scale
    
    lr = lr_init
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if it % 500 == 499:
            lr *= 0.7
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        err = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    grad = s - 1.0
                    err += grad * grad
                    dU[i * m + j] += grad * V[j * p + k] * W[k * n + i]
                    dV[j * p + k] += grad * U[i * m + j] * W[k * n + i]
                    dW[k * n + i] += grad * U[i * m + j] * V[j * p + k]
        
        rms_err = np.sqrt(err / (n * m * p))
        
        if rms_err < best_err:
            best_err = rms_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if rms_err < 1e-9:
            break
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    if best_err < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _try_small_integer_search(n, m, p, rank, max_trials=5000, seed=42, deadline=None):
    """
    Try to find a rank-`rank` decomposition with entries from a small set.
    """
    if deadline and time.time() > deadline:
        return None
    
    np.random.seed(seed)
    random.seed(seed)
    
    vals = [-1, 0, 1]
    
    for t in range(max_trials):
        if deadline and time.time() > deadline:
            break
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(vals) for _ in range(p * n)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try block decomposition approaches first (fast, guaranteed if valid)
    result = _try_block_decomposition(n, m, p, deadline)
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank and _verify(result["U"], result["V"], result["W"], n, m, p):
            best = result
            best_rank = rank
    
    result = _try_multi_strassen(n, m, p, deadline)
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank and _verify(result["U"], result["V"], result["W"], n, m, p):
            best = result
            best_rank = rank
    
    # Try to find even better decompositions via search
    for rank in range(best_rank - 1, 48, -1):
        if time.time() > deadline - 20:
            break
        
        found = None
        
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            result = _try_small_integer_search(n, m, p, rank, max_trials=1000, seed=seed, deadline=deadline)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    found = (U, V, W)
                    break
        
        if found is None:
            for seed in range(5):
                if time.time() > deadline - 20:
                    break
                result = _try_gradient_descent(n, m, p, rank, max_iters=3000, seed=seed, deadline=deadline)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, n, m, p):
                        found = (U, V, W)
                        break
        
        if found is not None:
            U, V, W = found
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best = {"U": U, "V": V, "W": W}
                break
    
    return best
# EVOLVE-BLOCK-END
