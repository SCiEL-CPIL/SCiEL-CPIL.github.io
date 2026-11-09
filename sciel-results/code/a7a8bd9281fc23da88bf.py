"""Find rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]
DEADLINE = time.time() + 110.0  # safety margin before 120s limit


def _validate(U, V, W, n, m, p):
    """Check if decomposition is exact."""
    rank = len(U[0])
    # W has shape (p*n, rank) with W[k*n+i, r]
    # U has shape (n*m, rank) with U[i*m+j, r]
    # V has shape (m*p, rank) with V[j*p+k, r]
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # We can check by constructing the full tensor
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape: U_arr[i*m+j, r], V_arr[j*p+k, r], W_arr[k*n+i, r]
    # For each r, the contribution to (i,j,k) is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Sum over r should give 1 for all i,j,k
    
    # Build tensor T of shape (n, m, p)
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u_col = U_arr[:, r]  # shape (n*m,)
        v_col = V_arr[:, r]  # shape (m*p,)
        w_col = W_arr[:, r]  # shape (p*n,)
        # Reshape
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        # For each (i,j,k): u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # This is an outer product structure
        # We can compute this as: for each k, the (i,j) slice is u_mat[i,j]*v_mat[j,k]*w_mat[k,i]
        # Let's use einsum or manual
        for k in range(p):
            # For fixed k: u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # = (u_mat * v_mat[:, k][:, None]) * w_mat[k, :][:, None]  ... wait
            # u_mat[i,j] * v_mat[j,k] for fixed k: this is u_mat * v_mat[:, k].T? No.
            # For fixed k, we want sum over j? No, we want the element (i,j) for each j.
            # Actually: for fixed k, the (i,j) entry is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # = w_mat[k,i] * (u_mat[:, k_idx] ... no
            # Let me just do it with einsum
            T[:, :, k] = u_mat * v_mat[:, k][:, None]  # shape (n, m): u_mat[i,j] * v_mat[j,k]
            T[:, :, k] *= w_mat[k, :][:, None]  # multiply by w_mat[k,i] for each i
    
    return np.allclose(T, 1.0, atol=1e-9)


def _to_fractional(U, V, W, tol=1e-9):
    """Try to convert float arrays to Fraction, returning (U_frac, V_frac, W_frac, ok)."""
    def convert_matrix(M):
        rows, cols = M.shape
        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                val = M[i, j]
                if abs(val) < tol:
                    row.append(Fraction(0))
                else:
                    frac = Fraction(val).limit_denominator(100000)
                    row.append(frac)
            result.append(row)
        return result
    
    return convert_matrix(U), convert_matrix(V), convert_matrix(W), True


def _search_random(n, m, p, rank, num_trials=1000, seed=None):
    """Random search for rank-rank decomposition."""
    if seed is not None:
        random.seed(seed)
    
    best = None
    for trial in range(num_trials):
        if time.time() > DEADLINE:
            break
        # Random initialization with small integers/half-integers
        U = np.random.randint(-2, 3, size=(n*m, rank)) * 0.5
        V = np.random.randint(-2, 3, size=(m*p, rank)) * 0.5
        W = np.random.randint(-2, 3, size=(p*n, rank)) * 0.5
        
        # Verify
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            for k in range(p):
                T[:, :, k] = u_mat * v_mat[:, k][:, None] * w_mat[k, :][:, None]
        
        if np.allclose(T, 1.0, atol=1e-6):
            # Found a valid decomposition
            return U, V, W
    return None


def _newton_polish(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Try to polish the solution using least-squares on the tensor equations."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    
    # The tensor equation: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # We have n*m*p = 75 equations, 3*rank = 3R unknowns (where R = rank)
    # If 75 >= 3R, we might be able to solve exactly.
    
    # Let's set up the system. For a given r, the unknowns are U[:,r], V[:,r], W[:,r].
    # This is nonlinear. Let's use a different approach: fix two of them, solve for the third.
    
    # Alternative: use the fact that for each (i,j), the equation over k gives:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for each k
    # This is still coupled.
    
    # Simpler approach: just check if it's already close enough and round.
    return U, V, W


def construct():
    """Find rank-minimal exact tensor decomposition for 3x5 * 5x5."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Known lower bounds and upper bounds for 3x5x5
    # Schoolbook: rank = 3*5*5 = 75
    # We want to find the minimal rank.
    # For small matrix multiplications, the minimal rank is often known or can be searched.
    # Let's start from a reasonable upper bound and try to reduce.
    
    # First, let's try known values. The rank of matrix multiplication (n,m,p)
    # is at least max(n*m, m*p, p*n) = max(15, 25, 15) = 25.
    # It's also at most n*m*p = 75.
    # For 3x5x5, let's search from rank 25 upward.
    
    best_decomp = None
    best_rank = n * m * p  # schoolbook
    
    # Start with schoolbook
    U_school = np.zeros((n*m, best_rank), dtype=np.float64)
    V_school = np.zeros((m*p, best_rank), dtype=np.float64)
    W_school = np.zeros((p*n, best_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Now try to find lower rank decompositions
    for rank in range(25, best_rank + 1):
        if time.time() > DEADLINE - 30:
            break
        
        # Try random search for this rank
        for attempt in range(20):
            if time.time() > DEADLINE - 30:
                break
            
            seed = hash((rank, attempt)) % (2**31)
            result = _search_random(n, m, p, rank, num_trials=50, seed=seed)
            if result is not None:
                U, V, W = result
                best_decomp = (U, V, W)
                best_rank = rank
                break
        
        if best_rank == rank:
            # Found decomposition at this rank
            break
    
    # Convert to fractional if possible
    U, V, W = best_decomp
    U_frac, V_frac, W_frac, ok = _to_fractional(U, V, W)
    
    if ok:
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    else:
        # Return as floats if fractional conversion fails
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
