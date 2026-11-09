import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p):
    """Verify that W[k*n+i, c] == U[i*m+j, c] * V[j*p+k, c] for all i,j,k,c."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    lhs = W[k * n + i][c]
                    rhs = U[i * m + j][c] * V[j * p + k][c]
                    if lhs != rhs:
                        return False
    return True


def _to_halfint_list(mat):
    """Convert a numpy array to a list of lists of Fractions (half-integers)."""
    rows = []
    for row in mat:
        r = []
        for x in row:
            xi = int(round(float(x) * 2))
            r.append(Fraction(xi, 2))
        rows.append(r)
    return rows


def _find_nullspace_halfint(B, target_cols, tmax=64):
    """
    Find `target_cols` linearly independent null vectors of B (rows of B^T),
    each with integer entries bounded in [-tmax, tmax].
    B is (m*p) x r with row rank m*p. Nullity should be target_cols.
    Returns (list_of_vectors, ok).
    """
    n_rows, n_cols = B.shape
    Bf = B.astype(object)
    
    # Gaussian elimination to find pivots and free variables
    pivots = []
    used = []
    for c in range(n_cols):
        col = Bf[:, c]
        if np.any(col != 0):
            pivots.append(c)
            used.append(col)
    
    if len(pivots) != n_rows:
        return [], False
        
    free = [c for c in range(n_cols) if c not in pivots]
    if len(free) < target_cols:
        return [], False
    
    # Build a basis of the nullspace: for each free column f, set x_f=1,
    # others free=0, solve for pivot columns.
    basis = []
    for f in free:
        x = np.zeros(n_cols, dtype=object)
        x[f] = 1
        # Back-substitute: for each pivot row r (in reverse), x[p] = -sum_{j>p} a[r,j] x[j]
        for r in range(len(pivots) - 1, -1, -1):
            p_idx = pivots[r]
            s = sum(Bf[r, j] * x[j] for j in range(p_idx + 1, n_cols))
            x[p_idx] = -s
        # Verify B @ x == 0
        if np.any(Bf @ x != 0):
            return [], False
        basis.append(x)
    
    # Try to find `target_cols` integer combinations with bounded entries.
    best = []
    # Use a limited search space for combinations
    search_range = range(-tmax, tmax + 1)
    
    # For efficiency, we only search a subset of combinations
    # We'll try to find independent vectors greedily
    for a in search_range:
        for b in search_range:
            for c in search_range:
                if a == 0 and b == 0 and c == 0:
                    continue
                v = a * basis[0] + b * basis[1] + c * basis[2]
                # Check if all entries are integers
                is_int = True
                for val in v:
                    if not (isinstance(val, int) or (isinstance(val, float) and val == int(val)) or (isinstance(val, Fraction) and val.denominator == 1)):
                        # Try to convert to int
                        try:
                            iv = int(val)
                            if iv != val:
                                is_int = False
                                break
                        except:
                            is_int = False
                            break
                if not is_int:
                    continue
                
                vi = [int(x) for x in v]
                if max(abs(x) for x in vi) > tmax:
                    continue
                
                # Check independence
                indep = True
                for bv in best:
                    if np.array_equal(np.array(vi), np.array(bv)):
                        indep = False
                        break
                if not indep:
                    continue
                
                # Check rank
                stacked = np.array(best + [vi])
                if np.linalg.matrix_rank(stacked.astype(float)) != len(best) + 1:
                    continue
                
                best.append(vi)
                if len(best) == target_cols:
                    return best, True
    
    return best, len(best) == target_cols


def _try_rank(R, n, m, p, max_attempts=50000, seed=12345):
    """
    Try to find a rank-R decomposition using the structured approach.
    Returns dict if found, None otherwise.
    """
    rng = np.random.default_rng(seed)
    
    for attempt in range(max_attempts):
        # Random half-integer matrices C (m x R) and D (m*p x R) with small entries.
        C = (rng.integers(-2, 3, size=(m, R))) / 2.0
        D = (rng.integers(-2, 3, size=(m * p, R))) / 2.0
        B = D.T  # (m*p x R)
        
        # Row rank of B must be m*p (full row rank)
        if np.linalg.matrix_rank(B) != m * p:
            continue
        
        # Get nullspace vectors (integers) of B^T.
        nullvecs, ok = _find_nullspace_halfint(B, target_cols=m, tmax=32)
        if not ok:
            continue
        
        Y = np.array(nullvecs, dtype=float)  # (m x m*p) rows = y1,...,ym
        # Y has shape (m, m*p), each row is a null vector of B^T, so B^T @ y_j = 0 => B @ y_j^T = 0
        
        # We need to construct U (n*m x R) and W (p*n x R) such that:
        # W[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        # where V = B (m*p x R).
        
        # Let U = M @ C^T? No, U is (n*m x R).
        # Let's define a matrix A (n x R) such that U[i*m+j, c] = A[i, c] * C[j, c]?
        # This is an outer product structure.
        # Then W[k*n+i, c] = sum_j A[i, c] C[j, c] B[j*p+k, c]
        # = A[i, c] * sum_j C[j, c] B[j*p+k, c]
        
        # Let Z[k, c] = sum_j C[j, c] B[j*p+k, c]
        # Then W[k*n+i, c] = A[i, c] * Z[k, c]
        
        # We need to check if this is consistent and if W is half-integer.
        # Also, we need the decomposition to be exact.
        # The condition for exactness is that the tensor equation holds.
        # With this structure, it holds by construction if we define W this way.
        # But we need W to be half-integer.
        
        Z = np.zeros((p, R))
        for k in range(p):
            for c in range(R):
                Z[k, c] = sum(C[j, c] * B[j * p + k, c] for j in range(m))
        
        # Check if Z is half-integer
        Z2 = Z * 2
        if not np.allclose(Z2, np.round(Z2), atol=1e-9):
            continue
        
        # Now choose A (n x R) with half-integer entries.
        # We can choose A randomly and check if U = A * C (element-wise) is half-integer.
        # U[i*m+j, c] = A[i, c] * C[j, c]
        # Since A and C are half-integers, U is half-integer.
        
        A = (rng.integers(-2, 3, size=(n, R))) / 2.0
        A2 = A * 2
        if not np.allclose(A2, np.round(A2), atol=1e-9):
            continue
        
        # Construct U, V, W
        U = np.zeros((n * m, R))
        V = B.copy()
        W = np.zeros((p * n, R))
        
        for i in range(n):
            for j in range(m):
                for c in range(R):
                    U[i * m + j, c] = A[i, c] * C[j, c]
        
        for k in range(p):
            for i in range(n):
                for c in range(R):
                    W[k * n + i, c] = A[i, c] * Z[k, c]
        
        # Check if U and W are half-integers
        U2 = U * 2
        if not np.allclose(U2, np.round(U2), atol=1e-9):
            continue
        
        W2 = W * 2
        if not np.allclose(W2, np.round(W2), atol=1e-9):
            continue
        
        # Verify the decomposition exactly
        U_list = _to_halfint_list(U)
        V_list = _to_halfint_list(V)
        W_list = _to_halfint_list(W)
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # If verification fails, continue searching
        continue
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s safety margin
    
    # Schoolbook decomposition (Rank 42)
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Lower bound is max(n*m, m*p, p*n) = 21
    # We try ranks from 24 up to 41
    ranks_to_try = [24, 25, 26, 27, 28, 30, 35]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        if R >= best_rank:
            continue
        
        # Try to find a decomposition of rank R
        result = _try_rank(R, n, m, p, max_attempts=10000, seed=12345 + R)
        if result is not None:
            best_result = result
            best_rank = R
            # If we found a rank close to the lower bound, we can stop
            if R <= 25:
                break
    
    return best_result
# EVOLVE-BLOCK-END
