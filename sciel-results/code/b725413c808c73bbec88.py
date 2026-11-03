"""Matrix multiplication 2x3 x 3x4 rank-minimal decomposition.
Combines structured Strassen-based construction with ALS refinement and
random search. The key insight: partition the 3x4 matrix into 3x2 blocks
and use Strassen's algorithm for the even sub-problems to potentially
achieve rank < 24 with integer/half-integer entries.
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p, tol=1e-8):
    """Exact verification of the decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (ip == i and jp == j and kp == k) else 0.0
                            val = np.dot(U[i * m + j], V[jp * p + k] * W[kp * n + ip])
                            if abs(val - expected) > tol:
                                return False
    return True

def _snap_to_half_int(arr):
    return np.round(np.asarray(arr, dtype=float) * 2) / 2.0

def _is_half_integer(arr, tol=1e-7):
    arr = np.asarray(arr, dtype=float)
    return np.all(np.abs(arr * 2 - np.round(arr * 2)) < tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _tensor_error(U, V, W, n, m, p):
    """Compute the total squared error of the decomposition."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (ip == i and jp == j and kp == k) else 0.0
                            val = np.dot(U[i * m + j], V[jp * p + k] * W[kp * n + ip])
                            err += (val - expected) ** 2
    return err

def _als_step_U(U, V, W, n, m, p, rank):
    """Solve for U given V, W via least squares."""
    new_U = U.copy()
    for i in range(n):
        for j in range(m):
            A_mat = np.zeros((p, rank))
            for k in range(p):
                A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                new_U[i * m + j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
            except:
                pass
    return new_U

def _als_step_V(U, V, W, n, m, p, rank):
    """Solve for V given U, W via least squares."""
    new_V = V.copy()
    for j in range(m):
        for k in range(p):
            A_mat = np.zeros((n, rank))
            for i in range(n):
                A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                new_V[j * p + k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
            except:
                pass
    return new_V

def _als_step_W(U, V, W, n, m, p, rank):
    """Solve for W given U, V via least squares."""
    new_W = W.copy()
    for k in range(p):
        for i in range(n):
            A_mat = np.zeros((m, rank))
            for j in range(m):
                A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                new_W[k * n + i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
            except:
                pass
    return new_W

def _als_solve(n, m, p, rank, U, V, W, max_iter=500, tol=1e-12, deadline=None):
    """Run alternating least squares to minimize tensor error."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        U = _als_step_U(U, V, W, n, m, p, rank)
        V = _als_step_V(U, V, W, n, m, p, rank)
        W = _als_step_W(U, V, W, n, m, p, rank)
        err = _tensor_error(U, V, W, n, m, p)
        if err < tol:
            return U, V, W, True
    return U, V, W, False

def _try_rank_als(n, m, p, rank, n_restarts, deadline, rng):
    """Try to find a rank-r decomposition via ALS with random restarts."""
    for restart in range(n_restarts):
        if time.time() > deadline:
            return None
        
        scale = 1.0 / np.sqrt(rank)
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        U, V, W, converged = _als_solve(n, m, p, rank, U, V, W, max_iter=300, tol=1e-12, deadline=deadline)
        
        if converged:
            U_s = _snap_to_half_int(U)
            V_s = _snap_to_half_int(V)
            W_s = _snap_to_half_int(W)
            
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                if _is_half_integer(U_s) and _is_half_integer(V_s) and _is_half_integer(W_s):
                    return (U_s, V_s, W_s)
            
            if _verify_exact(U, V, W, n, m, p, tol=1e-6):
                U_s = _snap_to_half_int(U)
                V_s = _snap_to_half_int(V)
                W_s = _snap_to_half_int(W)
                if _verify_exact(U_s, V_s, W_s, n, m, p):
                    return (U_s, V_s, W_s)
        
        err = _tensor_error(U, V, W, n, m, p)
        if err < 1e-6:
            U_s = _snap_to_half_int(U)
            V_s = _snap_to_half_int(V)
            W_s = _snap_to_half_int(W)
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                return (U_s, V_s, W_s)
    
    return None

def _strassen_2x2_decomposition():
    """
    Construct the Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4, 7), (4, 7), (4, 7) respectively.
    
    For C = A * B where A, B are 2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    Strassen's 7 products:
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))  # (4, 7)
    V = np.zeros((m * p, rank))  # (4, 7)
    W = np.zeros((p * n, rank))  # (4, 7)
    
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    
    # m1 = (a+d)(e+h)
    # U contributions: a (0,0) -> +1, d (1,1) -> +1
    # V contributions: e (0,0) -> +1, h (1,1) -> +1
    # W contributions: C[0,0] (0,0) -> +1, C[1,1] (1,1) -> +1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    # U: c (1,0) -> +1, d (1,1) -> +1
    # V: e (0,0) -> +1
    # W: C[1,0] (0,1) -> +1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # m3 = a(f-h)
    # U: a (0,0) -> +1
    # V: f (0,1) -> +1, h (1,1) -> -1
    # W: C[0,1] (1,0) -> +1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1
    
    # m4 = d(g-e)
    # U: d (1,1) -> +1
    # V: g (1,0) -> +1, e (0,0) -> -1
    # W: C[0,0] (0,0) -> +1, C[1,0] (0,1) -> +1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # m5 = (a+b)h
    # U: a (0,0) -> +1, b (0,1) -> +1
    # V: h (1,1) -> +1
    # W: C[0,0] (0,0) -> -1, C[0,1] (1,0) -> +1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[0, 5] = 1  # Wait, need to fix W indexing
    
    # Let me redo this more carefully
    # W[k*n+i, c] where k is row of B (0,1), i is row of A (0,1)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # W index: k*n + i = k*2 + i
    
    # m5 = (a+b)h
    # U: a (0,0) -> +1, b (0,1) -> +1
    # V: h (1,1) -> +1
    # W: C[0,0] (k=0,i=0) -> -1, C[0,1] (k=1,i=0) -> +1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # k=0,i=0 -> 0; k=1,i=0 -> 2
    
    # m6 = (c-a)(e+f)
    # U: c (1,0) -> +1, a (0,0) -> -1
    # V: e (0,0) -> +1, f (0,1) -> +1
    # W: C[1,1] (k=1,i=1) -> +1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # k=1,i=1 -> 3
    
    # m7 = (b-d)(g+h)
    # U: b (0,1) -> +1, d (1,1) -> -1
    # V: g (1,0) -> +1, h (1,1) -> +1
    # W: C[0,0] (k=0,i=0) -> +1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # k=0,i=0 -> 0
    
    return U, V, W

def _try_strassen_based(n, m, p, deadline):
    """
    Try to construct a decomposition using Strassen's algorithm.
    For 2x3 * 3x4, we can partition B into [B1 | B2] where B1, B2 are 3x2.
    Then C = A * B = A * [B1 | B2] = [A*B1 | A*B2].
    
    Each A*B1 and A*B2 is 2x3 * 3x2.
    We can further decompose each 2x3 * 3x2 product.
    
    Actually, let's think differently. The 3x4 matrix B can be written as:
    B = [B_top; B_bottom] where B_top is 2x4, B_bottom is 1x4.
    Then A*B = A_top*B + A_bottom*B where A_top is 2x2, A_bottom is 2x1... 
    This doesn't work cleanly.
    
    Alternative: Use the fact that 3 = 2+1 and 4 = 2+2.
    Partition A as [A1; A2] where A1 is 2x2, A2 is 1x2 (wait, A is 2x3).
    
    Let me try: A is 2x3, B is 3x4.
    Write A = [A_left | A_right] where A_left is 2x2, A_right is 2x1.
    Write B = [B_top; B_bottom] where B_top is 2x4, B_bottom is 1x4.
    
    Then A*B = A_left*B_top + A_right*B_bottom.
    
    A_left*B_top is 2x2 * 2x4 = 2x4. This can be done with Strassen if we split B_top into two 2x2 blocks.
    A_right*B_bottom is 2x1 * 1x4 = 2x4. This is just outer product, rank 1 per column... actually rank 4 total.
    
    Let's construct this explicitly.
    """
    # A is 2x3: indices (i,j) with i=0,1 and j=0,1,2
    # B is 3x4: indices (j,k) with j=0,1,2 and k=0,1,2,3
    # C is 2x4: indices (i,k) with i=0,1 and k=0,1,2,3
    
    # Split A = [A1 | a2] where A1 is 2x2 (j=0,1), a2 is 2x1 (j=2)
    # Split B = [B1; b2] where B1 is 2x4 (j=0,1), b2 is 1x4 (j=2)
    
    # C = A1*B1 + a2*b2
    
    # A1*B1: 2x2 * 2x4. Split B1 = [B1a | B1b] where B1a, B1b are 2x2.
    # A1*B1a: 2x2 * 2x2 -> use Strassen (rank 7)
    # A1*B1b: 2x2 * 2x2 -> use Strassen (rank 7)
    # Total for A1*B1: rank 14
    
    # a2*b2: 2x1 * 1x4 -> outer product, rank 4 (one per column)
    
    # Total rank: 14 + 4 = 18
    
    # Let's build this
    rank = 18
    U = np.zeros((n * m, rank))  # (6, 18)
    V = np.zeros((m * p, rank))  # (12, 18)
    W = np.zeros((p * n, rank))  # (8, 18)
    
    # Strassen for A1*B1a (columns k=0,1 of B)
    # A1: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices: 0,1,3,4
    # B1a: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices: 0,1,4,5
    # C1a: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices: 0,1,4,5
    
    S_U, S_V, S_W = _strassen_2x2_decomposition()
    
    # Map Strassen U (4,7) to our U for A1*B1a
    # Strassen U indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Our U indices for A1: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # So Strassen row 0 -> our row 0, row 1 -> our row 1, row 2 -> our row 3, row 3 -> our row 4
    U[0, 0:7] = S_U[0, :]  # A[0,0]
    U[1, 0:7] = S_U[1, :]  # A[0,1]
    U[3, 0:7] = S_U[2, :]  # A[1,0]
    U[4, 0:7] = S_U[3, :]  # A[1,1]
    
    # Map Strassen V (4,7) to our V for B1a
    # Strassen V indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Our V indices for B1a: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    V[0, 0:7] = S_V[0, :]  # B[0,0]
    V[1, 0:7] = S_V[1, :]  # B[0,1]
    V[4, 0:7] = S_V[2, :]  # B[1,0]
    V[5, 0:7] = S_V[3, :]  # B[1,1]
    
    # Map Strassen W (4,7) to our W for C1a
    # Strassen W indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Our W indices for C1a: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    W[0, 0:7] = S_W[0, :]  # C[0,0]
    W[1, 0:7] = S_W[1, :]  # C[0,1]
    W[4, 0:7] = S_W[2, :]  # C[1,0]
    W[5, 0:7] = S_W[3, :]  # C[1,1]
    
    # Strassen for A1*B1b (columns k=2,3 of B)
    # B1b: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices: 2,3,6,7
    # C1b: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices: 2,3,6,7
    
    U[0, 7:14] = S_U[0, :]
    U[1, 7:14] = S_U[1, :]
    U[3, 7:14] = S_U[2, :]
    U[4, 7:14] = S_U[3, :]
    
    V[2, 7:14] = S_V[0, :]  # B[0,2]
    V[3, 7:14] = S_V[1, :]  # B[0,3]
    V[6, 7:14] = S_V[2, :]  # B[1,2]
    V[7, 7:14] = S_V[3, :]  # B[1,3]
    
    W[2, 7:14] = S_W[0, :]  # C[0,2]
    W[3, 7:14] = S_W[1, :]  # C[0,3]
    W[6, 7:14] = S_W[2, :]  # C[1,2]
    W[7, 7:14] = S_W[3, :]  # C[1,3]
    
    # a2*b2: A[0,2]*B[2,k] and A[1,2]*B[2,k] for k=0,1,2,3
    # U indices for a2: (0,2)=2, (1,2)=5
    # V indices for b2: (2,0)=8, (2,1)=9, (2,2)=10, (2,3)=11
    # W indices for C: (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,0)=4, (1,1)=5, (1,2)=6, (1,3)=7
    
    # For each k, we need: U[0,2]*V[8+k] contributes to C[0,k]
    # and U[1,2]*V[8+k] contributes to C[1,k]
    
    # Column 14: A[0,2]*B[2,0] -> C[0,0]
    U[2, 14] = 1
    V[8, 14] = 1
    W[0, 14] = 1
    
    # Column 15: A[0,2]*B[2,1] -> C[0,1]
    U[2, 15] = 1
    V[9, 15] = 1
    W[1, 15] = 1
    
    # Column 16: A[0,2]*B[2,2] -> C[0,2]
    U[2, 16] = 1
    V[10, 16] = 1
    W[2, 16] = 1
    
    # Column 17: A[0,2]*B[2,3] -> C[0,3]
    U[2, 17] = 1
    V[11, 17] = 1
    W[3, 17] = 1
    
    # Wait, we also need A[1,2]*B[2,k] -> C[1,k]
    # But we only have 4 more columns if we want rank 18 total... 
    # Actually 14 + 4 = 18, but we need 8 more for the bottom part.
    # Let me recalculate: A1*B1a (7) + A1*B1b (7) + a2*b2 (4) = 18
    # But a2*b2 needs 4 products for the top row and 4 for the bottom row = 8 total.
    # So total rank is 7 + 7 + 8 = 22.
    
    # Let me redo with rank 22
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen for A1*B1a (columns 0-6)
    U[0, 0:7] = S_U[0, :]
    U[1, 0:7] = S_U[1, :]
    U[3, 0:7] = S_U[2, :]
    U[4, 0:7] = S_U[3, :]
    V[0, 0:7] = S_V[0, :]
    V[1, 0:7] = S_V[1, :]
    V[4, 0:7] = S_V[2, :]
    V[5, 0:7] = S_V[3, :]
    W[0, 0:7] = S_W[0, :]
    W[1, 0:7] = S_W[1, :]
    W[4, 0:7] = S_W[2, :]
    W[5, 0:7] = S_W[3, :]
    
    # Strassen for A1*B1b (columns 7-13)
    U[0, 7:14] = S_U[0, :]
    U[1, 7:14] = S_U[1, :]
    U[3, 7:14] = S_U[2, :]
    U[4, 7:14] = S_U[3, :]
    V[2, 7:14] = S_V[0, :]
    V[3, 7:14] = S_V[1, :]
    V[6, 7:14] = S_V[2, :]
    V[7, 7:14] = S_V[3, :]
    W[2, 7:14] = S_W[0, :]
    W[3, 7:14] = S_W[1, :]
    W[6, 7:14] = S_W[2, :]
    W[7, 7:14] = S_W[3, :]
    
    # a2*b2 top row (columns 14-17): A[0,2]*B[2,k] -> C[0,k]
    for k in range(4):
        U[2, 14 + k] = 1
        V[8 + k, 14 + k] = 1
        W[k, 14 + k] = 1
    
    # a2*b2 bottom row (columns 18-21): A[1,2]*B[2,k] -> C[1,k]
    for k in range(4):
        U[5, 18 + k] = 1
        V[8 + k, 18 + k] = 1
        W[4 + k, 18 + k] = 1
    
    # Verify
    if _verify_exact(U, V, W, n, m, p):
        return (U, V, W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    np.random.seed(42)
    
    start_time = time.time()
    deadline = start_time + 130
    
    # Schoolbook fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_fb, V_fb, W_fb)
    
    # Try Strassen-based construction first
    if time.time() < deadline - 30:
        result = _try_strassen_based(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            best_rank = U.shape[1]
            best_result = (U, V, W)
    
    # Try ALS from rank 12 upward
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank_als(n, m, p, target_rank, n_restarts=20, deadline=deadline - 10, rng=rng)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
