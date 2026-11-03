"""Tensor decomposition for 4x4 x 4x9 matrix multiplication via recursive Kronecker composition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 150.0

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Return U, V, W for M_{2,2,2} (rank 7).
    Shapes: (4,7) each.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # r=0: (A11+A22)(B11+B22) -> C11, C22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: (A21+A22)B11 -> C21, C22
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # r=2: A11(B21-B11) -> C12, C22
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # r=3: A22(B12-B22) -> C11, C21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # r=4: (A11+A12)B22 -> C11, C12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # r=5: (A21-A11)(B11+B12) -> C21
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: (A12-A22)(B21+B22) -> C12
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_terms():
    """
    Return U, V, W for M_{4,4,4} using 2-level Strassen (rank 49).
    Shapes: (16, 49) each.
    """
    # Use Kronecker composition of 2x2 Strassen with itself
    u2, v2, w2 = _strassen_2x2_terms()
    return _tensor_kron(u2, v2, w2)

def _tensor_kron(u, v, w):
    """
    Compose two tensor decompositions via Kronecker product.
    If (u,v,w) decomposes M_{n1,m1,p1} with rank r1,
    and we want M_{n1*n2, m1*m2, p1*p2}, we can compose with another.
    
    But here we use it differently: given a base decomposition for M_{2,2,2},
    we compose it with itself to get M_{4,4,4}.
    
    For M_{n1*n2, m1*m2, p1*p2}:
    U[i1*n2+i2, j1*m2+j2, r1*r2+r2] = u[i1,j1,r1] * u[i2,j2,r2]  -- wait, that's for same tensor.
    
    Actually, for composing M_{n1,m1,p1} with M_{n2,m2,p2} to get M_{n1*n2, m1*m2, p1*p2}:
    The tensor is T[i,j,k] = delta[i1*i2, j1*j2, k1*k2] where i=i1*n2+i2 etc.
    This is NOT a simple Kronecker of two multiplications.
    
    Let me think again. M_{4,4,4} can be seen as block multiplication:
    C = A*B where A is 4x4, B is 4x4.
    Partition into 2x2 blocks of size 2x2:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    This is exactly the tensor product (Kronecker) of two M_{2,2,2} tensors.
    
    For the Kronecker product of two tensor decompositions:
    If T1 = sum_r u1[r] v1[r] w1[r] with rank r1
    and T2 = sum_s u2[s] v2[s] w2[s] with rank r2
    then T1 ⊗ T2 = sum_{r,s} (u1[r]⊗u2[s]) (v1[r]⊗v2[s]) (w1[r]⊗w2[s])
    with rank r1*r2.
    
    For M_{4,4,4} = M_{2,2,2} ⊗ M_{2,2,2}:
    U[i1*2+i2, j1*2+j2, r1*7+r2] = u2x2[i1,j1,r1] * u2x2[i2,j2,r2]
    V[j1*2+j2, k1*2+k2, r1*7+r2] = v2x2[j1,k1,r1] * v2x2[j2,k2,r2]
    W[k1*2+k2, i1*2+i2, r1*7+r2] = w2x2[k1,i1,r1] * w2x2[k2,i2,r2]
    """
    # u, v, w are for M_{2,2,2}, shapes (4,7)
    # We want M_{4,4,4}, shapes (16, 49)
    r1 = u.shape[1]
    n2 = 2  # inner dimension
    
    U = np.zeros((4, r1 * r1))
    V = np.zeros((4, r1 * r1))
    W = np.zeros((4, r1 * r1))
    
    for r1_idx in range(r1):
        for r2_idx in range(r1):
            col = r1_idx * r1 + r2_idx
            # U: outer index (i1,j1), inner index (i2,j2)
            # U[i1*2+i2, j1*2+j2, col] = u[i1,j1,r1_idx] * u[i2,j2,r2_idx]
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            U[i1*2+i2, j1*2+j2, col] = u[i1,j1,r1_idx] * u[i2,j2,r2_idx]
            
            # V: outer (j1,k1), inner (j2,k2)
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            V[j1*2+j2, k1*2+k2, col] = v[j1,k1,r1_idx] * v[j2,k2,r2_idx]
            
            # W: outer (k1,i1), inner (k2,i2)
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            W[k1*2+k2, i1*2+i2, col] = w[k1,i1,r1_idx] * w[k2,i2,r2_idx]
    
    return U, V, W

def _schoolbook_terms(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}. Rank n*m*p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_recursive(n, m, p, time_limit):
    """
    Recursively build a decomposition for M_{n,m,p} by splitting the output dimension p
    into the largest power-of-2 block <= p, and recursing on the remainder.
    
    Returns (U, V, W) or None if time is exceeded.
    """
    if time.time() > time_limit:
        return None
    
    if p == 0:
        return None
    
    # If n and m are both 1, it's just a scalar multiplication: rank p
    if n == 1 and m == 1:
        return _schoolbook_terms(1, 1, p)
    
    # If p is a power of 2 and n=m=p, use Strassen
    if n == m == p and (p & (p - 1)) == 0 and p >= 2:
        if p == 2:
            return _strassen_2x2_terms()
        elif p == 4:
            return _strassen_4x4_terms()
        else:
            # For larger powers of 2, we could recurse, but for now fallback
            return _schoolbook_terms(n, m, p)
    
    # Find the largest power of 2 <= p
    largest_pow2 = 1
    while largest_pow2 * 2 <= p and largest_pow2 * 2 <= n and largest_pow2 * 2 <= m:
        largest_pow2 *= 2
    
    # Actually, we want to split p into blocks. The constraint is that the block size
    # for the "square" part must satisfy n_block = m_block = p_block for Strassen.
    # But n=4, m=4 are fixed. We can only use Strassen for square blocks where p_block <= min(n,m).
    # So max square block size is min(n, m, p) rounded down to power of 2.
    
    max_square = min(n, m, p)
    largest_pow2 = 1
    while largest_pow2 * 2 <= max_square:
        largest_pow2 *= 2
    
    if largest_pow2 >= 2:
        # Use a square block of size largest_pow2
        n_block = largest_pow2
        m_block = largest_pow2
        p_block = largest_pow2
        
        # Get decomposition for the square block
        if n_block == 2:
            U_block, V_block, W_block = _strassen_2x2_terms()
        elif n_block == 4:
            U_block, V_block, W_block = _strassen_4x4_terms()
        else:
            U_block, V_block, W_block = _schoolbook_terms(n_block, m_block, p_block)
        
        r_block = U_block.shape[1]
        
        # Recurse on the remainder
        p_remainder = p - p_block
        if p_remainder > 0:
            result_remainder = _build_recursive(n, m, p_remainder, time_limit)
            if result_remainder is None:
                return None
            U_rem, V_rem, W_rem = result_remainder
            r_rem = U_rem.shape[1]
            
            total_rank = r_block + r_rem
            U = np.zeros((n * m, total_rank))
            V = np.zeros((m * p, total_rank))
            W = np.zeros((p * n, total_rank))
            
            # Copy block part
            U[:, 0:r_block] = U_block
            for j in range(m_block):
                for k in range(p_block):
                    V[j * p + k, 0:r_block] = V_block[j * m_block + k, :]
            for k in range(p_block):
                for i in range(n_block):
                    W[k * n + i, 0:r_block] = W_block[k * n_block + i, :]
            
            # Copy remainder part
            U[:, r_block:r_block + r_rem] = U_rem
            for j in range(m):
                for k in range(p_remainder):
                    V[j * p + (p_block + k), r_block:r_block + r_rem] = V_rem[j * p_remainder + k, :]
            for k in range(p_remainder):
                for i in range(n):
                    W[(p_block + k) * n + i, r_block:r_block + r_rem] = W_rem[k * n + i, :]
            
            return U, V, W
        else:
            # No remainder
            return U_block, V_block, W_block
    else:
        # Cannot use Strassen, use schoolbook
        return _schoolbook_terms(n, m, p)

def _verify(U, V, W, n, m, p):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Reshape for broadcasting
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # For each (i, j, k), check sum over r
    # U_r[i, j, :] shape (rank,)
    # V_r[j, k, :] shape (rank,)
    # W_r[k, i, :] shape (rank,)
    
    # Use einsum or manual
    total = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total[i, j, k] = np.dot(np.dot(U_r[i, j, :], V_r[j, k, :]), W_r[k, i, :])
    
    return np.allclose(total, 1.0, atol=1e-10)

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()
    
    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]
    
    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)
    
    return np.all(sums == 1)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    best_result = None
    best_rank = n * m * p  # schoolbook fallback
    
    # Try recursive decomposition
    try:
        result = _build_recursive(n, m, p, start_time + DEADLINE - 10)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p):
                best_result = (U, V, W)
                best_rank = U.shape[1]
    except Exception:
        pass
    
    # Deadline guard
    if time.time() - start_time > DEADLINE - 5:
        if best_result is not None:
            U, V, W = best_result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            Us, Vs, Ws = _schoolbook_terms(n, m, p)
            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        Us, Vs, Ws = _schoolbook_terms(n, m, p)
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
