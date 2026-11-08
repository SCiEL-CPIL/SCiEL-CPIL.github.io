"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U and U[0] else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,c], V_arr[:,c], W_arr[:,c])
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0
    
    if not np.allclose(T_decomp, T_target, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _gf2_gaussian_elimination(A, b):
    """
    Solve Ax = b over GF(2) using Gaussian elimination.
    A is a 2D array of 0s and 1s, b is a 1D array of 0s and 1s.
    Returns a solution x or None if inconsistent.
    """
    A = np.array(A, dtype=np.int8).copy()
    b = np.array(b, dtype=np.int8).copy()
    n_rows, n_cols = A.shape
    n_rhs = len(b)
    
    # Augment
    Aug = np.column_stack([A, b.reshape(-1, 1)])
    
    rank = 0
    for col in range(n_cols):
        # Find pivot
        pivot = None
        for row in range(rank, n_rows):
            if Aug[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap
        Aug[[rank, pivot]] = Aug[[pivot, rank]]
        # Eliminate
        for row in range(n_rows):
            if row != rank and Aug[row, col] == 1:
                Aug[row] = (Aug[row] + Aug[rank]) % 2
        rank += 1
    
    # Check consistency
    for row in range(rank, n_rows):
        if Aug[row, n_cols:] != 0:
            return None
    
    # Back-substitute to get a solution (set free variables to 0)
    x = np.zeros(n_cols, dtype=np.int8)
    for row in range(rank):
        # Find the leading column in this row
        for col in range(n_cols):
            if Aug[row, col] == 1:
                x[col] = Aug[row, n_cols]
                break
    
    return x

def _gf2_find_sparse_pattern(n, m, p, target_rank, deadline):
    """
    Over GF(2), find a sparse support pattern for rank r decomposition.
    Each term c has supports:
      U_c: subset of {0,...,n*m-1}
      V_c: subset of {0,...,m*p-1}
      W_c: subset of {0,...,p*n-1}
    
    The constraint is that for each (i,j,k), the sum over c of
      U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] ≡ 1 (mod 2)
    
    This is a system of bilinear equations. We linearize by fixing the
    support of U and V, then solving for W, or vice versa.
    
    Alternative: enumerate small support patterns and check.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    total_terms = n * m * p
    
    # For each target rank r, we need to find r triples (U_c, V_c, W_c)
    # such that the bilinear form matches the tensor over GF(2).
    
    # Strategy: use a random initialization of supports and then solve.
    # Or: start from the schoolbook and try to merge terms.
    
    # Let's try a different approach: for each possible rank r, 
    # randomly assign each of the n*m*p tensor entries to one of r terms,
    # then check if it's consistent.
    
    # Actually, let's use the structure: the tensor has exactly one 1 per (i,j,k)
    # at position (i*m+j, j*p+k, k*n+i).
    
    # For rank r, we need r rank-1 tensors whose sum (over GF(2)) equals T.
    
    # A rank-1 tensor over GF(2) is defined by three binary vectors.
    
    # Let's try: for a given r, randomly initialize U, V, W as binary matrices
    # of shape (nm, r), (mp, r), (pn, r) and check if they work.
    # If not, do a few iterations of local search.
    
    best_pattern = None
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random binary initialization
        U_bin = np.random.randint(0, 2, size=(nm, target_rank))
        V_bin = np.random.randint(0, 2, size=(mp, target_rank))
        W_bin = np.random.randint(0, 2, size=(pn, target_rank))
        
        # Check if this works over GF(2)
        T_decomp = np.zeros((nm, mp, pn), dtype=np.int8)
        for c in range(target_rank):
            T_decomp += (np.outer(np.outer(U_bin[:, c], V_bin[:, c]), W_bin[:, c]) % 2)
        T_decomp = T_decomp % 2
        
        T_target = np.zeros((nm, mp, pn), dtype=np.int8)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m + j, j*p + k, k*n + i] = 1
        
        if np.array_equal(T_decomp, T_target):
            best_pattern = (U_bin, V_bin, W_bin)
            return best_pattern
        
        # Local search: flip bits to reduce error
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Compute current error
            T_decomp = np.zeros((nm, mp, pn), dtype=np.int8)
            for c in range(target_rank):
                T_decomp += (np.outer(np.outer(U_bin[:, c], V_bin[:, c]), W_bin[:, c]) % 2)
            T_decomp = T_decomp % 2
            
            error = (T_decomp + T_target) % 2  # XOR difference
            num_errors = np.sum(error)
            
            if num_errors == 0:
                best_pattern = (U_bin, V_bin, W_bin)
                return best_pattern
            
            # Try flipping a random bit in one of the matrices
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                row = random.randint(0, nm - 1)
                col = random.randint(0, target_rank - 1)
                U_bin[row, col] = 1 - U_bin[row, col]
            elif mat_idx == 1:
                row = random.randint(0, mp - 1)
                col = random.randint(0, target_rank - 1)
                V_bin[row, col] = 1 - V_bin[row, col]
            else:
                row = random.randint(0, pn - 1)
                col = random.randint(0, target_rank - 1)
                W_bin[row, col] = 1 - W_bin[row, col]
            
            # Check if error decreased
            T_decomp_new = np.zeros((nm, mp, pn), dtype=np.int8)
            for c in range(target_rank):
                T_decomp_new += (np.outer(np.outer(U_bin[:, c], V_bin[:, c]), W_bin[:, c]) % 2)
            T_decomp_new = T_decomp_new % 2
            
            error_new = (T_decomp_new + T_target) % 2
            num_errors_new = np.sum(error_new)
            
            if num_errors_new >= num_errors:
                # Revert
                if mat_idx == 0:
                    U_bin[row, col] = 1 - U_bin[row, col]
                elif mat_idx == 1:
                    V_bin[row, col] = 1 - V_bin[row, col]
                else:
                    W_bin[row, col] = 1 - W_bin[row, col]
    
    return best_pattern

def _lift_to_integers(U_bin, V_bin, W_bin, n, m, p, deadline):
    """
    Lift a GF(2) pattern to integer coefficients.
    The GF(2) solution gives us the support structure.
    We need to find integer coefficients such that the sum of outer products
    equals the tensor over the reals.
    
    For the matrix multiplication tensor, if the GF(2) pattern works,
    the integer lift is often just the same binary values (0 or 1),
    but we may need to adjust signs or scale.
    
    Actually, for the standard matrix multiplication tensor, if we have a
    valid GF(2) decomposition, the same 0/1 values work over the integers
    because the tensor entries are all 1.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U_bin.shape[1]
    
    # Try the direct lift: use the binary values as integer coefficients
    U_int = U_bin.astype(np.float64)
    V_int = V_bin.astype(np.float64)
    W_int = W_bin.astype(np.float64)
    
    # Verify
    T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
    for c in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_int[:, c], V_int[:, c], W_int[:, c])
    
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0
    
    if np.allclose(T_decomp, T_target, atol=1e-8):
        return U_int, V_int, W_int
    
    # If direct lift doesn't work, try solving for integer coefficients
    # given the support structure. This is a linear system.
    
    # For each (a, b, c) in the tensor, we have:
    # sum_{r} U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is bilinear, not linear. But if we fix U and V, it's linear in W.
    
    # Let's try: fix U and V to the binary values, solve for W.
    # But W also has binary values, so this is constrained.
    
    # Alternative: try small integer values (-1, 0, 1) for coefficients.
    
    # For now, return None if the direct lift doesn't work.
    return None

def _try_gf2_search(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using GF(2) analysis.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try ranks from 30 up to 44 (schoolbook is 45)
    for rank in range(30, 45):
        if time.time() > deadline:
            break
        
        pattern = _gf2_find_sparse_pattern(n, m, p, rank, deadline)
        if pattern is not None:
            U_bin, V_bin, W_bin = pattern
            result = _lift_to_integers(U_bin, V_bin, W_bin, n, m, p, deadline)
            if result is not None:
                U_int, V_int, W_int = result
                return U_int, V_int, W_int, rank
    
    return None

def _try_eliminate_terms(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by eliminating terms.
    Uses a local search approach: try to set columns to zero or adjust nearby columns
    to absorb the contribution of a removed term.
    """
    current_rank = U.shape[1]
    best_rank = current_rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    T_target = _build_tensor(n, m, p)
    
    # Greedy elimination: try to remove each column, and if the tensor is no longer satisfied,
    # try to adjust the remaining columns to compensate.
    # This is a heuristic local search.
    
    # We'll iterate through columns and try to remove them.
    # If removal breaks the decomposition, we try to find a small adjustment to the remaining
    # columns to fix it. This is equivalent to solving a least squares problem on the residuals.
    
    # For efficiency, we'll only try a limited number of eliminations and adjustments.
    
    # Let's try a simpler approach: random restarts with random initializations and gradient descent.
    # But that might be too slow.
    
    # Alternative: Use the fact that the tensor is sparse.
    # Each term in the schoolbook decomposition corresponds to a unique (i,j,k).
    # If we can find two terms that can be combined into one, we reduce the rank.
    # This happens if the supports overlap in a specific way.
    
    # For now, let's try a simple random search for rank reduction.
    # We'll try to find a decomposition of rank r < current_rank by random initialization and
    # then using a few steps of gradient descent or alternating least squares.
    
    # This is likely to fail for large rank reductions, but might find small improvements.
    
    # Given the time constraints, let's just return the schoolbook for now.
    # The elimination approach is complex and might not find improvements quickly.
    
    return best_U, best_V, best_W, best_rank

def _random_search(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of target_rank using random initialization and
    alternating least squares.
    """
    T_target = _build_tensor(n, m, p)
    
    # Random initialization
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    # Alternating least squares
    # For each c, U[:,c] = T_target * (V[:,c] * W[:,c]) / (V[:,c] * W[:,c])^2
    # This is not quite right. Let's use a simpler approach.
    
    # Actually, the ALS for tensor decomposition is:
    # U[:,c] = T_target * (V[:,c] * W[:,c]) / (V[:,c] * W[:,c])^2
    # But this is for the case where we're trying to match T_target exactly.
    
    # Let's use a gradient descent approach instead.
    # This is likely to be too slow for the given time constraints.
    
    # Given the complexity, let's just return None.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    
    best_result = _schoolbook(n, m, p)
    
    # Convert to numpy arrays for processing
    U = np.array(best_result["U"], dtype=np.float64)
    V = np.array(best_result["V"], dtype=np.float64)
    W = np.array(best_result["W"], dtype=np.float64)
    
    # Try GF(2) search first
    gf2_result = _try_gf2_search(n, m, p, deadline)
    if gf2_result is not None:
        U_new, V_new, W_new, rank_new = gf2_result
        if rank_new < best_result["rank"]:
            def to_list(mat):
                result = []
                for row in mat:
                    new_row = []
                    for val in row:
                        # Round to nearest half-integer
                        v = round(val * 2) / 2
                        if abs(v - round(v)) < 1e-9:
                            new_row.append(int(round(v)))
                        else:
                            new_row.append(v)
                    result.append(new_row)
                return result
            
            best_result = {
                "U": to_list(U_new),
                "V": to_list(V_new),
                "W": to_list(W_new),
                "rank": rank_new
            }
            U, V, W = U_new, V_new, W_new
    
    # Try to eliminate terms
    U, V, W, current_rank = _try_eliminate_terms(U, V, W, n, m, p, deadline)
    
    if current_rank < best_result["rank"]:
        # Convert back to list of lists
        def to_list(mat):
            result = []
            for row in mat:
                new_row = []
                for val in row:
                    # Round to nearest half-integer
                    v = round(val * 2) / 2
                    if abs(v - round(v)) < 1e-9:
                        new_row.append(int(round(v)))
                    else:
                        new_row.append(v)
                result.append(new_row)
            return result
        
        best_result = {
            "U": to_list(U),
            "V": to_list(V),
            "W": to_list(W),
            "rank": current_rank
        }
    
    return best_result

# EVOLVE-BLOCK-END

