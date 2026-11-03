"""
Rank-minimal tensor decomposition for 2x4 * 4x7 matrix multiplication.

The trivial decomposition has rank n*p = 14. We attempt to find a lower rank
by using Strassen-like block decompositions on the 2x2 sub-blocks, then
optimising the remaining free parameters with a local numerical search.

The tensor T has entries T_{(i,j),(j,k),(k,i)} = 1 for all valid i,j,k
(0-indexed), with i in [0,1], j in [0,3], k in [0,6].

A rank-r decomposition consists of matrices U (8 x r), V (28 x r), W (14 x r)
such that for all (i,j),(j,k),(k,i):
    sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c} = 1

We start with the rank-14 seed and try to reduce rank by finding a valid
rank-13, 12, ... decomposition using:
1. Analytic Strassen-style constructions on 2x2 blocks.
2. Numerical local search (Gauss-Newton style) to adjust parameters.
"""
import time
import numpy as np
from itertools import product

n, m, p = 2, 4, 7
DEADLINE = time.time() + 140.0  # safety margin within 150s


def _build_tensor_equations(rank, U, V, W):
    """
    Compute residuals for the tensor decomposition.
    For each (i, j, k) with the constraint that the (i,j) index of A,
    (j,k) index of B, and (k,i) index of C must be consistent:
      residual = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] - 1
    Returns a flat array of residuals (size n*m*p = 56).
    """
    residuals = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i
                s = np.dot(U[i*m + j], V[j*p + k]) * W[k*n + i]
                # Actually the tensor is sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Not dot product of U and V rows; it's elementwise product then sum
                s = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i])
                residuals.append(s - 1.0)
    return np.array(residuals)


def _verify(U, V, W, tol=1e-6):
    """Check if the decomposition is exact."""
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err


def _rank_np_seed():
    """Rank n*p = 14 seed decomposition."""
    rank = n * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for k in range(p):
            c = k * n + i
            for j in range(m):
                U[i*m + j, c] = 1.0
                V[j*p + k, c] = 1.0
            W[k*n + i, c] = 1.0
    return U, V, W, rank


def _strassen_222():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 tensor.
    
    Standard Strassen:
    Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    p1 = (a+d)(e+h)
    p2 = (c+d)(e)
    p3 = (a)(f-h)
    p4 = (d)(g-c)
    p5 = (a+b)(h)
    p6 = (c-a)(e+f)
    p7 = (b-d)(g+h)
    
    C = [[p1+p4-p5+p7, p3+p5],
         [p2+p4, p1-p2+p3+p6]]
    """
    # U rows: A indices (0,0),(0,1),(1,0),(1,1) -> a,b,c,d
    # V rows: B indices (0,0),(0,1),(1,0),(1,1) -> e,f,g,h
    # W rows: C indices (0,0),(0,1),(1,0),(1,1)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a+d)(e+h): U has a=1,d=1; V has e=1,h=1; W: C00 += 1, C11 -= 1? No.
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 and C11
    
    # p2 = (c+d)(e)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # p3 = (a)(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # p4 = (d)(g-c)
    U[3, 3] = 1
    V[2, 3] = 1; V[1, 3] = -1  # g - c
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # p5 = (a+b)(h)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C00
    W[1, 4] = 1   # C01
    
    # p6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1   # C11
    
    # p7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1   # C00
    
    return U, V, W


def _construct_strassen_blocks():
    """
    Construct a decomposition using Strassen on 2x2 blocks.
    
    We have A (2x4), B (4x7), C (2x7).
    
    Split B into left (4x4) and right (4x3) parts? Or split differently.
    
    Approach: Split the m=4 dimension into two groups of 2: j in {0,1} and j in {2,3}.
    Split the p=7 dimension into groups: k in {0,1,2,3} and k in {4,5,6}.
    
    Actually, let's think of it as:
    C = A @ B where A is 2x4, B is 4x7.
    
    We can write B as [B1 | B2] where B1 is 4x4, B2 is 4x3.
    C = [A1 @ B1_left | ...] -- this gets complicated.
    
    Alternative: Use the fact that for each pair of columns of B (forming a 4x2 block),
    we can use a 2x2x2 Strassen-like decomposition if we also group rows of A.
    
    Let's try: Group the 4 columns of A (and 4 rows of B) into two 2-groups: {0,1} and {2,3}.
    Group the 7 columns of B (and 7 columns of C) into pairs: {0,1}, {2,3}, {4,5}, and {6}.
    
    For each pair of columns (k, k+1) where k is even (0,2,4), we have a 2x2x2 subproblem
    with A being 2x2 (rows 0,1 of A for columns 0,1 or 2,3), and B being 4x2.
    
    This is still not clean. Let me try a different approach.
    
    Direct construction: For the 2x4x7 problem, note that n=2 is small.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can precompute linear combinations of A's rows and B's columns.
    For rank r, we need r scalars p_c such that:
    C[i,k] = sum_c W[k,i,c] * p_c
    where p_c = (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k])
    
    For the n*p=14 seed, each c corresponds to a unique (i,k) pair, and
    U[i,j,c] = 1 for all j when i matches, V[j,k,c] = 1 for all j when k matches,
    W[k,i,c] = 1 when (i,k) matches c, else 0.
    
    To reduce rank, we need to find cases where two (i,k) pairs can share a term.
    This requires that the "pattern" of U and V for those two pairs is compatible.
    """
    
    # Let's try to build a rank-13 or lower by starting from rank 14
    # and trying to merge terms.
    
    U, V, W, rank = _rank_np_seed()
    
    # Try to eliminate one term by expressing it as a combination of others.
    # This is equivalent to finding a linear dependence among the rank-1 tensors.
    
    # Actually, let's use a numerical approach: start from rank 14 and try to
    # find a valid rank-13 decomposition by setting one column to a linear
    # combination of others and adjusting.
    
    return U, V, W, rank


def _local_search(U, V, W, rank, max_iter=1000, tol=1e-8):
    """
    Local search to minimize the sum of squared residuals.
    Uses a simple gradient descent with backtracking.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute residuals
        res = _build_tensor_equations(rank, U, V, W)
        sse = np.sum(res**2)
        if sse < tol:
            break
        
        # Compute gradient (approximate)
        eps = 1e-6
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # This is expensive for large rank. Let's use a simpler approach:
        # just do a random perturbation search (simulated annealing style).
        
        # Actually, for rank 13-14, the number of parameters is manageable.
        # But gradient computation is O(n*m*p*rank) per parameter.
        # Let's use coordinate descent instead.
        
        # Coordinate descent on U
        improved = False
        for idx in np.random.choice(U.size, min(20, U.size), replace=False):
            i, j = np.unravel_index(idx, U.shape)
            old_val = U[i, j]
            for delta in [-0.01, 0.01]:
                U[i, j] = old_val + delta
                res2 = _build_tensor_equations(rank, U, V, W)
                sse2 = np.sum(res2**2)
                if sse2 < sse:
                    sse = sse2
                    improved = True
                    break
                U[i, j] = old_val
        
        # Coordinate descent on V
        for idx in np.random.choice(V.size, min(20, V.size), replace=False):
            i, j = np.unravel_index(idx, V.shape)
            old_val = V[i, j]
            for delta in [-0.01, 0.01]:
                V[i, j] = old_val + delta
                res2 = _build_tensor_equations(rank, U, V, W)
                sse2 = np.sum(res2**2)
                if sse2 < sse:
                    sse = sse2
                    improved = True
                    break
                V[i, j] = old_val
        
        # Coordinate descent on W
        for idx in np.random.choice(W.size, min(20, W.size), replace=False):
            i, j = np.unravel_index(idx, W.shape)
            old_val = W[i, j]
            for delta in [-0.01, 0.01]:
                W[i, j] = old_val + delta
                res2 = _build_tensor_equations(rank, U, V, W)
                sse2 = np.sum(res2**2)
                if sse2 < sse:
                    sse = sse2
                    improved = True
                    break
                W[i, j] = old_val
        
        if not improved:
            break
    
    return U, V, W, sse


def construct():
    start_time = time.time()
    
    # Start with rank 14 seed
    U14, V14, W14, rank14 = _rank_np_seed()
    ok, err = _verify(U14, V14, W14)
    best_U, best_V, best_W, best_rank = U14, V14, W14, rank14
    if not ok:
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n*m, rank_school), dtype=int)
        V = np.zeros((m*p, rank_school), dtype=int)
        W = np.zeros((p*n, rank_school), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, c] = 1
                    V[j*p + k, c] = 1
                    W[k*n + i, c] = 1
                    c += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Try to find lower rank using local search from rank 14
    # First, try rank 13 by removing one column and optimizing
    if time.time() < DEADLINE - 30:
        for target_rank in [13, 12, 11]:
            if time.time() > DEADLINE - 10:
                break
            
            # Start from rank 14 and try to zero out one column
            for col_to_zero in range(rank14):
                if time.time() > DEADLINE - 10:
                    break
                
                U_try = U14.copy()
                V_try = V14.copy()
                W_try = W14.copy()
                U_try[:, col_to_zero] = 0
                V_try[:, col_to_zero] = 0
                W_try[:, col_to_zero] = 0
                
                # Local search
                U_opt, V_opt, W_opt, sse = _local_search(
                    U_try, V_try, W_try, target_rank, max_iter=200
                )
                
                ok, err = _verify(U_opt, V_opt, W_opt, tol=1e-5)
                if ok:
                    best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, target_rank
                    break
            
            if best_rank < rank14:
                break
    
    # Try Strassen-based construction
    if time.time() < DEADLINE - 20 and best_rank >= 14:
        # Try to build using 2x2 Strassen blocks
        # Split j into {0,1} and {2,3}, k into {0,1}, {2,3}, {4,5}, {6}
        # For each pair of k's, use a 2x2x2 Strassen decomposition
        # This would give rank = 3 * 7 + 7 = 28? No, that's worse.
        
        # Alternative: Use the n=2 structure.
        # C[0,k] = sum_j A[0,j] B[j,k]
        # C[1,k] = sum_j A[1,j] B[j,k]
        
        # We can define p_c = L_c(A) * R_c(B) and C[i,k] = sum_c W[i,k,c] p_c
        # For rank 14, each (i,k) has its own term.
        # To reduce rank, we need W to have a rank < 14 structure in the (i,k) plane.
        
        # Let's try a different seed: use random initialization and local search
        np.random.seed(42)
        for target_rank in [13, 12]:
            if time.time() > DEADLINE - 10:
                break
            
            for attempt in range(5):
                if time.time() > DEADLINE - 10:
                    break
                
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
                
                U_opt, V_opt, W_opt, sse = _local_search(
                    U_init, V_init, W_init, target_rank, max_iter=300
                )
                
                ok, err = _verify(U_opt, V_opt, W_opt, tol=1e-5)
                if ok:
                    best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, target_rank
                    break
    
    # Convert to integer/half-integer if possible
    def to_clean_list(mat):
        mat = np.asarray(mat, dtype=float)
        # Try to find a scaling factor that makes all entries integers
        for scale in [1, 2, 4, 8, 16]:
            scaled = mat * scale
            rounded = np.round(scaled)
            if np.allclose(scaled, rounded, atol=1e-6):
                rounded = rounded.astype(int)
                # Check if we can divide by 2
                if np.all(rounded % 2 == 0):
                    rounded = rounded // 2
                return rounded.tolist()
        # If not clean, return floats
        return mat.tolist()
    
    return {
        'U': to_clean_list(best_U),
        'V': to_clean_list(best_V),
        'W': to_clean_list(best_W)
    }
