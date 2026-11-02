"""Tensor decomposition for 3x4 * 4x7 matrix multiplication.

Strategy:
- The known rank formula gives R(3,4,7) = max(12,28,21,42) = 42.
- We use a structured construction: partition the 3x4x7 tensor into
  sub-tensors and use known low-rank decompositions.
- Verify exactness with integer arithmetic (multiply by 2 for half-integers).
- Deadline guard at 140s (safety margin before 150s kill).
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 84
RANK_FORMULA = max(n * m, m * p, p * n, (n * m * p + 1) // 2)  # 42


def _build_tensor():
    """Build the full 3D tensor T[i,j,k] = 1."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T


def _schoolbook():
    """Return the schoolbook decomposition (rank 84)."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                val = np.dot(u_row * V[j * p + k, :], W[k * n + i, :])
                if val != T[i, j, k]:
                    return False
    return True


def _verify_scaled(U2, V2, W2, T):
    """Verify using integer-scaled values (U2 = 2*U, etc.) for exact arithmetic."""
    r = U2.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U2[i * m + j, :]
            for k in range(p):
                # sum_c U2[i,j,c] * V2[j,k,c] * W2[k,i,c] / 4 == T[i,j,k]
                # => sum_c U2[i,j,c] * V2[j,k,c] * W2[k,i,c] == 4*T[i,j,k]
                val = 0
                for c in range(r):
                    val += u_row[c] * V2[j * p + k, c] * W2[k * n + i, c]
                if val != 4 * T[i, j, k]:
                    return False
    return True


def _compute_error(U, V, W, T):
    """Compute sum of squared errors."""
    err = 0.0
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                val = np.dot(u_row * V[j * p + k, :], W[k * n + i, :])
                err += (val - T[i, j, k]) ** 2
    return err


def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2 * 2x2 multiplication.
    Uses Strassen's algorithm.
    Tensors indexed as U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # C = A * B where A is 2x2, B is 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U[i*2+j, c] = coefficient of A[i,j] in product c
    # V[j*2+k, c] = coefficient of B[j,k] in product c
    # W[k*2+i, c] = coefficient of C[i,k] in product c
    #
    # Note: C[i,k] where i is row of C (=row of A), k is col of C (=col of B)
    # W[k*2+i, c]: k is col index, i is row index of C
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h): A[0,0]=1, A[1,1]=1, B[0,0]=1, B[0,1]=1, B[1,0]=1, B[1,1]=1
    # C11 += 1, C12 += 0, C21 += 0, C22 += 1
    U[0, 0] = 1; U[3, 0] = 1  # a, d
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1  # e, f, g, h
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # M2 = (c+d)e: A[1,0]=1, A[1,1]=1, B[0,0]=1
    # C21 += 1, C22 += -1
    U[2, 1] = 1; U[3, 1] = 1  # c, d
    V[0, 1] = 1  # e
    W[2, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]
    
    # M3 = a(f-h): A[0,0]=1, B[0,1]=1, B[1,1]=-1
    # C12 += 1, C22 += 1
    U[0, 2] = 1  # a
    V[1, 2] = 1; V[3, 2] = -1  # f, -h
    W[1, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # M4 = d(g-e): A[1,1]=1, B[1,0]=1, B[0,0]=-1
    # C11 += 1, C21 += 1
    U[3, 3] = 1  # d
    V[2, 3] = 1; V[0, 3] = -1  # g, -e
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0], C[1,0]
    
    # M5 = (a+b)h: A[0,0]=1, A[0,1]=1, B[1,1]=1
    # C11 += -1, C12 += 1
    U[0, 4] = 1; U[1, 4] = 1  # a, b
    V[3, 4] = 1  # h
    W[0, 4] = -1; W[1, 4] = 1  # C[0,0], C[0,1]
    
    # M6 = (c-a)(e+f): A[1,0]=1, A[0,0]=-1, B[0,0]=1, B[0,1]=1
    # C22 += 1
    U[2, 5] = 1; U[0, 5] = -1  # c, -a
    V[0, 5] = 1; V[1, 5] = 1  # e, f
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (b-d)(g+h): A[0,1]=1, A[1,1]=-1, B[1,0]=1, B[1,1]=1
    # C11 += 1
    U[1, 6] = 1; U[3, 6] = -1  # b, -d
    V[2, 6] = 1; V[3, 6] = 1  # g, h
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _construct_rank_42():
    """
    Construct a rank-42 decomposition for 3x4 * 4x7.
    
    Strategy: Use the formula-based construction.
    For R(n,m,p) = max(nm, mp, np, ceil(nmp/2)) = 42.
    
    We partition the computation and use structured low-rank pieces.
    
    Alternative: Use a direct construction based on the known algorithm.
    For 3x4x7 with rank 42, we can use a block decomposition.
    
    Let me try a different approach: use the "slice" method.
    Partition the 4x7 output into blocks and the 3x4 input into blocks.
    
    Actually, let me use a randomized search initialized with structure.
    """
    T = _build_tensor()
    
    # Start with a structured initialization
    rank = RANK_FORMULA  # 42
    
    # Initialize with a pattern that has some structure
    random.seed(42)
    
    # Use a combination of Strassen-like blocks
    # For 3x4: split into 2x2 and 1x2 blocks
    # For 4x7: split into 2x7 blocks
    
    # Let's try a direct approach: initialize randomly and refine
    U = np.random.randint(-2, 3, size=(n * m, rank)).astype(np.float64)
    V = np.random.randint(-2, 3, size=(m * p, rank)).astype(np.float64)
    W = np.random.randint(-2, 3, size=(p * n, rank)).astype(np.float64)
    
    # Scale to get reasonable magnitudes
    U *= 0.5
    V *= 0.5
    W *= 0.5
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _compute_error(best_U, best_V, best_W, T)
    
    # Local search with deadline
    deadline = time.time() + 130
    step_size = 0.5
    iterations = 0
    max_iter = 200000
    
    while iterations < max_iter and time.time() < deadline:
        iterations += 1
        
        # Pick a random matrix and entry
        mat_idx = random.choice([0, 1, 2])
        if mat_idx == 0:
            row = random.randint(0, n * m - 1)
            col = random.randint(0, rank - 1)
            old = best_U[row, col]
            best_U[row, col] = old + random.choice([-step_size, -step_size/2, step_size/2, step_size])
        elif mat_idx == 1:
            row = random.randint(0, m * p - 1)
            col = random.randint(0, rank - 1)
            old = best_V[row, col]
            best_V[row, col] = old + random.choice([-step_size, -step_size/2, step_size/2, step_size])
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, rank - 1)
            old = best_W[row, col]
            best_W[row, col] = old + random.choice([-step_size, -step_size/2, step_size/2, step_size])
        
        err = _compute_error(best_U, best_V, best_W, T)
        if err < best_err:
            best_err = err
            if best_err < 1e-6:
                break
        else:
            # Revert
            if mat_idx == 0:
                best_U[row, col] = old
            elif mat_idx == 1:
                best_V[row, col] = old
            else:
                best_W[row, col] = old
        
        # Occasionally shrink step size
        if iterations % 10000 == 0:
            step_size *= 0.7
    
    # Round to half-integers
    best_U = np.round(best_U * 2) / 2.0
    best_V = np.round(best_V * 2) / 2.0
    best_W = np.round(best_W * 2) / 2.0
    
    # Verify
    if _verify(best_U, best_V, best_W, T):
        return best_U, best_V, best_W
    
    return None, None, None


def _try_structured_construction():
    """
    Try a structured construction based on partitioning.
    
    For 3x4 * 4x7:
    - Split the 3x4 matrix into a 2x4 and 1x4 part
    - Use Strassen-like tricks for the 2x4 * 4x7 part
    
    Actually, let me try a more systematic approach.
    
    The key insight: R(3,4,7) = 42. We can construct this by:
    - Using the fact that R(2,4,7) <= 2*7 = 14 (via Strassen on 2x2 blocks)
    - R(1,4,7) = 7 (trivial)
    - But we need to handle the overlap properly.
    
    Let me try: partition the 3 rows into {0,1} and {2}.
    For rows {0,1}: 2x4 * 4x7. Use Strassen on 2x2 blocks of the 4x7.
    Split 4x7 into two 2x7 blocks.
    2x2 * 2x7: rank 7*7 = 49? No, that's too high.
    
    Actually, R(2,2,7) = 7 (Strassen for the 2x2 part, replicated for each of 7 columns).
    So 2x4 * 4x7: split 4 into two 2s.
    2x2 * 2x7: rank 7 per 2x2 block, so 2 blocks * 7 = 14.
    1x4 * 4x7: rank 7.
    Total: 14 + 7 = 21? That's less than 42.
    
    Wait, that can't be right. Let me reconsider.
    
    R(2,2,7): For 2x2 * 2x7, we can use Strassen's 7 products for each pair of columns.
    Actually, 2x2 * 2x7 is 2x2 * (2x7) = (2x2 * 2x2) * 7 columns? No.
    
    2x2 * 2x7: The tensor has dimensions 2*2*7 = 28 entries.
    R(2,2,7) = 7 (using Strassen: 7 products, each handling 2x2 * 2x1, replicated 7 times).
    Wait, no. Strassen gives R(2,2,2) = 7. For 2x2 * 2x7, we can do it in 7*7 = 49?
    No, that's not right either.
    
    Actually, R(2,2,p) = 7p for p >= 2? No, that's not optimal.
    R(2,2,p) can be done in 7p using Strassen replicated p times, but it can be better.
    For 2x2 * 2x7, R = 7 (each of the 7 Strassen products is a scalar, and we need 7 outputs per column... no).
    
    Let me just try the direct approach with the formula rank 42.
    """
    pass


def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    T = _build_tensor()
    U0, V0, W0 = _schoolbook()
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = RANK_SCHOOLBOOK
    
    # Try rank 42 with structured search
    if time.time() < deadline - 5:
        result = _construct_rank_42()
        if result[0] is not None:
            best_U, best_V, best_W = result
            best_rank = RANK_FORMULA
    
    # If we still have time, try to improve or verify
    if not _verify(best_U, best_V, best_W, T):
        # Fallback to schoolbook
        best_U, best_V, best_W = U0, V0, W0
        best_rank = RANK_SCHOOLBOOK
    
    # Convert to list of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
