"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must equal 1.
    """
    r = len(U[0])
    # Build the multiplication tensor
    # T[i,j,k] should be 1 for all i,j,k
    # Check a few random indices or all
    # For efficiency, check all if small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_like_3x3x8(n, m, p):
    """
    Attempt to find a lower-rank decomposition.
    For 3x3 * 3x8, we can think of it as 8 independent 3x3 * 3x1 multiplications.
    However, there might be a shared structure.
    
    A known lower bound for matrix multiplication is related to the rank of the tensor.
    For 2x2, rank is 7 (Strassen).
    For 3x3, the minimal rank is not known exactly but is <= 23 (schoolbook) and >= 19.
    For 3x3 * 3x8, we can potentially do better than 72 (3*3*8).
    
    Let's try a specific construction.
    We can split the 3x8 matrix into two 3x4 blocks? Or use a 3x3 Strassen-like approach for each column?
    Actually, let's look at the structure.
    A (3x3) * (3x8) = (3x8).
    Let A be 3x3, B be 3x8.
    We can compute C = A * B.
    
    Consider the tensor T_{i,j,k} where i is row of A, j is col of A (row of B), k is col of B.
    T_{i,j,k} = sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c.
    
    Let's try to use a rank-1 decomposition for each of the 8 columns of B, but share some factors.
    Actually, a simpler approach:
    Since n=3, m=3, p=8.
    Schoolbook rank = 72.
    
    Let's try to construct a decomposition with rank < 72.
    One idea: Use the fact that we're multiplying by a 3x8 matrix.
    We can write B as [B1, B2, B3] where each Bi is 3x1? No, that's not helpful.
    
    Let's try a generic optimization approach.
    We want to minimize r such that there exist U (9xr), V (24xr), W (24xr) satisfying the constraints.
    This is a hard problem.
    
    Alternative: Use a known algorithm for 3x3x3 and extend it.
    For 3x3x3, the best known rank is 23 (Landsberg).
    For 3x3x8, we can think of it as 8 separate 3x3x1 multiplications.
    But we can share the U factors.
    Let U be 9xr.
    For each column k of B (k=0..7), we have a 3x3x1 multiplication.
    The tensor for column k is T^k_{i,j} = sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c.
    
    If we use the same U for all columns, then for each k, we need V_k (3x8? no, 3x1 per column) and W_k (3x1).
    Actually, V is (m*p, r) = (3*8, r) = (24, r).
    V_{j,k} is the (j,k)-th element of the 3x8 block of V.
    W is (p*n, r) = (8*3, r) = (24, r).
    W_{k,i} is the (k,i)-th element of the 8x3 block of W.
    
    So for each k, we have a 3x3 matrix multiplication with:
    U (shared), V_k (3x1 part of V), W_k (1x3 part of W).
    
    This is equivalent to finding a rank-r decomposition of the 3x3x3 tensor that works for 8 different (V_k, W_k) pairs, sharing the same U.
    
    This is a constrained problem. Let's try to solve it using numerical optimization.
    """
    # We will try to find a rank-r decomposition for r < 72.
    # Start with r = 50 and try to optimize.
    # We will use a simple gradient-free method or random restarts.
    # Given the time limit, we will try a few random initializations and optimize.
    
    # Actually, let's first check if there's a simple combinatorial construction.
    # For 2x2x2, Strassen uses 7 rank-1 tensors.
    # For 3x3x3, the best known is 23.
    # For 3x3x8, we might be able to use 8 * 23 / 3 = 61.33, so maybe 62?
    # But sharing U might help more.
    
    # Let's try a numerical search for a small rank.
    # We will use the fact that the problem is symmetric in some ways.
    # We will initialize U, V, W randomly and then use a few steps of optimization.
    # However, optimization of tensor rank is non-convex and hard.
    
    # Given the complexity, let's first return the schoolbook if we can't find better quickly.
    # But let's try one specific construction:
    # Use the 3x3x3 rank-23 decomposition for the first 3 columns, and replicate for the rest?
    # That would be 23 * 8 / 3 = 61.33, so 62 if we can share.
    
    # Actually, let's try a simpler idea:
    # For each of the 8 columns, use a 3x3x1 multiplication.
    # The minimal rank for 3x3x1 is 9 (schoolbook) but can be lower?
    # No, 3x3x1 is just a linear map, so rank is 9.
    # But if we share U, we might do better.
    
    # Let's try to find a decomposition with r = 60.
    # We will use random initialization and a few iterations of a simple update rule.
    # But this is risky and might not converge.
    
    # Given the time constraints and the difficulty, let's return the schoolbook for now.
    # But let's try one more thing:
    # Can we use a 2x2 Strassen-like block decomposition?
    # Split A into 2x2 blocks? No, 3 is odd.
    
    # Let's try to use a known result:
    # The rank of the 3x3x3 tensor is at most 23.
    # For 3x3x8, we can use 8 * 23 = 184, which is worse than 72.
    # So sharing is crucial.
    
    # Let's try a random search for r = 50.
    # We will initialize U, V, W randomly and then check if we can adjust them to satisfy the constraints.
    # This is a system of 3*3*8 = 72 equations in 9*r + 24*r + 24*r = 57*r variables.
    # For r = 50, we have 2850 variables and 72 equations, so it should be solvable.
    # We can use a linear least squares approach for each rank-1 term.
    
    # Actually, we can solve this as a linear problem for fixed U, V, W structure.
    # But the structure is nonlinear.
    
    # Let's try a simple approach:
    # Initialize U, V, W with random values.
    # Then, for each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    # This is a system of 72 equations.
    # We can try to solve it by fixing U and V and solving for W, then fixing U and W and solving for V, etc.
    
    # Let's try r = 50.
    r = 50
    # Initialize
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    # Normalize
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    # Iterate
    for _ in range(100):
        # Update W
        for k in range(p):
            for i in range(n):
                # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                # This is a system of m equations in W[k*n+i, :]
                # Let A_j = U[i*m+j, :] * V[j*p+k, :] (element-wise product)
                # We need A_j @ W[k*n+i, :] = 1 for all j
                # This is a least squares problem.
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve A @ w = 1
                w, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k * n + i, :] = w
        
        # Update V
        for j in range(m):
            for k in range(p):
                # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                # Let A_i = U[i*m+j, :] * W[k*n+i, :]
                # We need A_i @ V[j*p+k, :] = 1 for all i
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                v, residuals, rank, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V[j * p + k, :] = v
        
        # Update U
        for i in range(n):
            for j in range(m):
                # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # Let A_k = V[j*p+k, :] * W[k*n+i, :]
                # We need A_k @ U[i*m+j, :] = 1 for all k
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                u, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i * m + j, :] = u
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        if err < 1e-6:
            break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe
    
    # Try schoolbook first
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    
    # Try to find a better decomposition
    try:
        better = _try_strassen_like_3x3x8(n, m, p)
        if better is not None:
            best = better
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best
    
    return best

# EVOLVE-BLOCK-END
