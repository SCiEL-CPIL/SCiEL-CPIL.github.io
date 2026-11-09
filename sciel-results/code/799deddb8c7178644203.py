"""Matrix multiplication 3,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]
import time
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s

# EVOLVE-BLOCK-START
import random
import math
import numpy as np
from itertools import product

def _build_schoolbook():
    """Standard schoolbook matrix multiplication tensor decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate_decomposition(result, ntest=5, seed=42):
    """Validate that the decomposition computes exact matrix multiplication."""
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    if not U or not V or not W:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rng = random.Random(seed)
    for t in range(ntest):
        A = np.array([[rng.uniform(-3, 3) for _ in range(m)] for _ in range(n)])
        B = np.array([[rng.uniform(-3, 3) for _ in range(p)] for _ in range(m)])
        Ctrue = A @ B
        
        # Compute Ccomp
        # Ccomp[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
        Ccomp = np.zeros((n, p))
        for r in range(len(U[0])):
            u_col = U_np[:, r].reshape(n, m)
            v_col = V_np[:, r].reshape(m, p)
            w_mat = W_np[:, r].reshape(p, n)
            
            uv_prod = u_col @ v_col
            term = uv_prod * w_mat.T
            Ccomp += term
            
        if not np.allclose(Ccomp, Ctrue, atol=1e-9):
            return False
    return True

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 * 2x2.
    Returns U, V, W each of shape (4, 7) for the 2x2 case.
    The index mapping is:
    U: rows are (i,j) for i in {0,1}, j in {0,1}
    V: rows are (j,k) for j in {0,1}, k in {0,1}
    W: rows are (k,i) for k in {0,1}, i in {0,1}
    
    Strassen formulas:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    # Indices:
    # U: 00->0, 01->1, 10->2, 11->3
    # V: 00->0, 01->1, 10->2, 11->3
    # W: 00->0, 01->1, 10->2, 11->3
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # M2: (A10+A11)B00
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0; W[3][1] = -1.0
    
    # M3: A00(B10-B11)
    U[0][2] = 1.0
    V[2][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0; W[3][2] = 1.0
    
    # M4: A11(B01-B10)
    U[3][3] = 1.0
    V[1][3] = 1.0; V[2][3] = -1.0
    W[0][3] = 1.0; W[2][3] = 1.0
    
    # M5: (A00+A01)B11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0
    
    # M6: (A10-A00)(B00+B10)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[2][5] = 1.0
    W[3][5] = 1.0
    
    # M7: (A01-A11)(B01+B11)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[1][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0
    
    return U, V, W

def _build_block_decomposition():
    """
    Build a block decomposition for 3x4 * 4x8 using Strassen on 2x2 blocks.
    
    We decompose:
    A is 3x4, B is 4x8.
    Split A into 2x2 blocks? No, 3x4 doesn't fit 2x2 nicely.
    
    Let's try a different approach:
    Split the middle dimension m=4 into two halves of size 2.
    A = [A1, A2] where A1, A2 are 3x2.
    B = [B1; B2] where B1, B2 are 2x8.
    C = A1*B1 + A2*B2.
    
    This is a sum of two matrix products: (3x2)*(2x8) and (3x2)*(2x8).
    Rank of (3x2)*(2x8) is 3*2*8 = 48 schoolbook.
    Total rank = 48 + 48 = 96. No improvement.
    
    Let's try splitting n=3 into 1+2, and m=4 into 2+2, and p=8 into 4+4.
    This is complex.
    
    Alternative: Use the fact that R(3,4,8) <= R(3,4,4) + R(3,4,4) if we split p.
    But we need to exploit the structure.
    
    Let's try splitting the 4x8 matrix multiplication into 2x4 * 2x2 blocks.
    Actually, let's look at the inspiration: "factor the 4x8 multiplication into a (2x4)x(2x2) structure".
    This suggests decomposing the problem into independent sub-problems.
    
    Let's try a simpler block structure:
    Split A (3x4) into top 2 rows and bottom 1 row.
    A = [A_top; A_bottom] where A_top is 2x4, A_bottom is 1x4.
    Split B (4x8) into left 4 cols and right 4 cols.
    B = [B_left, B_right] where both are 4x4.
    C = [A_top*B_left, A_top*B_right; A_bottom*B_left, A_bottom*B_right].
    
    This is 4 independent matrix multiplications:
    1. (2x4) * (4x4) -> 2x4
    2. (2x4) * (4x4) -> 2x4
    3. (1x4) * (4x4) -> 1x4
    4. (1x4) * (4x4) -> 1x4
    
    Rank for (2x4)*(4x4): 2*4*4 = 32.
    Rank for (1x4)*(4x4): 1*4*4 = 16.
    Total rank = 32 + 32 + 16 + 16 = 96. Still schoolbook.
    
    To improve, we need to use Strassen or similar on the sub-blocks.
    For (2x4)*(4x4), we can split the 4x4 into 2x2 blocks and use Strassen?
    A 2x4 times 4x4 is not a standard square block multiplication.
    
    Let's try a different decomposition.
    Use the known bound R(3,4,8) <= 2*R(3,2,8) or similar.
    
    Actually, let's try to implement a generic block decomposition with Strassen on 2x2 blocks where possible.
    
    For now, let's try a random local search to improve the schoolbook rank.
    """
    # Fallback to schoolbook if block decomposition is complex
    return _build_schoolbook()

def _random_local_search(initial_result, max_iter=100, seed=42):
    """
    Try to reduce rank by finding zero columns or linear dependencies.
    This is a heuristic search.
    """
    n, m, p = DIMENSIONS
    U = np.array(initial_result['U'], dtype=float)
    V = np.array(initial_result['V'], dtype=float)
    W = np.array(initial_result['W'], dtype=float)
    
    rank = U.shape[1]
    rng = random.Random(seed)
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Check for zero columns
    for r in range(rank):
        if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
            # This column is zero, so it contributes nothing.
            # We can remove it.
            pass # In a real implementation, we'd remove and reindex
    
    # For simplicity, we return the initial result if we can't find improvements quickly.
    # A more sophisticated search would perturb U, V, W and check validity.
    
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME
    budget = _TIME_LIMIT - elapsed
    if budget <= 10:
        return _build_schoolbook()
    
    best = _build_schoolbook()
    best_rank = len(best['U'][0])
    
    # Try local search if we have time
    if budget > 30:
        local_result = _random_local_search(best, max_iter=50, seed=int(time.time() % 1000))
        if _validate_decomposition(local_result):
            local_rank = len(local_result['U'][0])
            if local_rank < best_rank:
                best = local_result
                best_rank = local_rank
    
    return {
        "U": [[float(x) for x in row] for row in best['U']],
        "V": [[float(x) for x in row] for row in best['V']],
        "W": [[float(x) for x in row] for row in best['W']]
    }
# EVOLVE-BLOCK-END
