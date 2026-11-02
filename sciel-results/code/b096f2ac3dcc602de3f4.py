"""Matrix multiplication 2x5x6 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 5, 6]
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _is_halfint(x):
    """Check if x is an integer or half-integer."""
    fx = Fraction(x)
    return (fx * 2).denominator == 1

def _to_halfint(x):
    """Convert to half-integer representation as Fraction."""
    return Fraction(x)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    # Build the tensor: T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1 if (i * m + j) >= 0 and (j * p + k) >= 0 and (k * n + i) >= 0 else 0
                # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                # Actually, the multiplication tensor is T[i,j,k] = 1 for the (i,j,k) term
                # Wait - let me reconsider. The tensor for matrix mult C = A*B where A is n x m, B is m x p
                # C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor T has indices (i, j, k) with T[i,j,k] = 1 (it's the identity in the j-dimension)
                # So T[i,j,k] = 1 for all i, j, k
                expected = 1
                if s != expected:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Fast verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    # U: (n*m, rank) -> for each (i,j), we have U[i*m+j, :]
    # V: (m*p, rank) -> for each (j,k), we have V[j*p+k, :]
    # W: (p*n, rank) -> for each (k,i), we have W[k*n+i, :]
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Build T
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_vec = U_arr[:, c]  # (n*m,)
        v_vec = V_arr[:, c]  # (m*p,)
        w_vec = W_arr[:, c]  # (p*n,)
        
        # Reshape
        u_3d = u_vec.reshape(n, m)
        v_3d = v_vec.reshape(m, p)
        w_3d = w_vec.reshape(p, n)
        
        # T[i,j,k] += u_3d[i,j] * v_3d[j,k] * w_3d[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_3d[i, j] * v_3d[j, k] * w_3d[k, i]
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-10):
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, timeout=5.0):
    """
    Try to reduce the rank by removing linearly dependent terms.
    For each term c, check if the rank-1 tensor U[:,c] ⊗ V[:,c] ⊗ W[:,c] 
    can be expressed as a linear combination of the other terms.
    """
    start = time.time()
    rank = len(U[0])
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    active = list(range(rank))
    removed = []
    
    # We'll iteratively try to remove terms
    # For a term to be removable, its rank-1 tensor must be in the span of the others
    
    # Build the "tensor space" representation
    # Each rank-1 term c corresponds to a vector in R^(n*m*p)
    # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We want to find if T_c is in span{T_c' : c' != c}
    
    # For efficiency, we work in the vector space of dimension n*m*p
    # But that's 60-dimensional, which is manageable
    
    dim = n * m * p
    
    def term_vector(c):
        """Get the vector representation of rank-1 term c."""
        vec = [Fraction(0)] * dim
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    vec[idx] = U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
        return vec
    
    # Precompute all term vectors
    term_vecs = [term_vector(c) for c in active]
    
    # Iteratively try to remove terms
    changed = True
    while changed and (time.time() - start) < timeout:
        changed = False
        new_active = []
        for idx, c in enumerate(active):
            # Check if term c is in the span of all other active terms
            others = [term_vecs[i] for i in range(len(active)) if i != idx]
            
            if not others:
                new_active.append(c)
                continue
            
            # Check if term_vecs[idx] is in span(others) using Gaussian elimination
            # Build matrix with others as rows, check if term_vecs[idx] is in row space
            # Actually, check if it's in the column space of the matrix formed by others
            
            # Use rank check: rank([others; term_vecs[idx]]) == rank(others)
            # Build augmented matrix
            mat = others + [term_vecs[idx]]
            
            # Gaussian elimination to check rank
            # We need rank(mat) == rank(others)
            
            def matrix_rank(mat_list):
                """Compute rank of a list of vectors (as list of Fractions) using Gaussian elimination."""
                if not mat_list:
                    return 0
                rows = len(mat_list)
                cols = len(mat_list[0])
                # Work on a copy
                mat = [row[:] for row in mat_list]
                rank = 0
                for col in range(cols):
                    # Find pivot
                    pivot = None
                    for row in range(rank, rows):
                        if mat[row][col] != 0:
                            pivot = row
                            break
                    if pivot is None:
                        continue
                    # Swap
                    mat[rank], mat[pivot] = mat[pivot], mat[rank]
                    # Eliminate
                    pivot_val = mat[rank][col]
                    for row in range(rows):
                        if row != rank:
                            factor = mat[row][col] / pivot_val
                            for c2 in range(col, cols):
                                mat[row][c2] -= factor * mat[rank][c2]
                    rank += 1
                return rank
            
            rank_others = matrix_rank(others)
            rank_aug = matrix_rank(mat)
            
            if rank_aug == rank_others:
                # Term c is dependent, remove it
                removed.append(c)
                changed = True
            else:
                new_active.append(c)
        
        active = new_active
        # Recompute term_vecs for active terms
        term_vecs = [term_vector(c) for c in active]
    
    if not active:
        return None
    
    # Build reduced decomposition
    new_rank = len(active)
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    for new_c, old_c in enumerate(active):
        for i in range(n * m):
            new_U[i][new_c] = U[i][old_c]
        for j in range(m * p):
            new_V[j][new_c] = V[j][old_c]
        for k in range(p * n):
            new_W[k][new_c] = W[k][old_c]
    
    return new_U, new_V, new_W

def _greedy_reduce(U, V, W, n, m, p):
    """
    Iteratively apply rank reduction until no more improvement.
    """
    start = time.time()
    current_U, current_V, current_W = U, V, W
    prev_rank = len(current_U[0])
    
    while _time_left() > 10.0:
        result = _try_reduce_rank(current_U, current_V, current_W, n, m, p, timeout=5.0)
        if result is None:
            break
        new_U, new_V, new_W = result
        new_rank = len(new_U[0])
        
        if new_rank >= prev_rank:
            break
        
        # Verify the new decomposition
        if not _verify_numpy(new_U, new_V, new_W, n, m, p):
            break
        
        current_U, current_V, current_W = new_U, new_V, new_W
        prev_rank = new_rank
    
    return current_U, current_V, current_W

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Apply greedy rank reduction
    U, V, W = _greedy_reduce(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}
