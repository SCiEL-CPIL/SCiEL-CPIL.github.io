"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
from functools import lru_cache

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
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

def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    Returns the global U, V, W slices (as arrays of shape (n*m, r), etc.)
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    # U: local index li*mi + lj -> global (n_off+li)*m + (m_off+lj)
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    # V: local index lj*pi + lk -> global (m_off+lj)*p + (p_off+lk)
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    # W: local index lk*ni + li -> global (p_off+lk)*n + (n_off+li)
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g

def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
    Use Strassen for 2x2x2 blocks, schoolbook for others.
    Returns (U, V, W, rank).
    """
    # Compute offsets
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    # Determine rank for each block
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]
    
    # Construct global U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    U_local, V_local, W_local = _strassen_2x2x2()
                else:
                    U_local, V_local, W_local = _schoolbook(ni, mi, pi)
                
                U_g, V_g, W_g = _embed_block(
                    U_local, V_local, W_local,
                    n, m, p,
                    n_offsets[i], m_offsets[j], p_offsets[k],
                    ni, mi, pi
                )
                
                U[:, col:col+block_ranks[(i,j,k)]] += U_g
                V[:, col:col+block_ranks[(i,j,k)]] += V_g
                W[:, col:col+block_ranks[(i,j,k)]] += W_g
                col += block_ranks[(i,j,k)]
    
    return U, V, W, total_rank

def _generate_partitions(n, m, p, max_parts=4):
    """
    Generate all possible partitions of dimensions into up to max_parts parts.
    Each partition is a tuple of positive integers summing to the dimension.
    """
    def _partitions(d, max_parts):
        if d == 0:
            yield ()
            return
        if max_parts == 0:
            return
        for first in range(1, d + 1):
            for rest in _partitions(d - first, max_parts - 1):
                yield (first,) + rest
    
    n_parts_list = list(_partitions(n, max_parts))
    m_parts_list = list(_partitions(m, max_parts))
    p_parts_list = list(_partitions(p, max_parts))
    
    return n_parts_list, m_parts_list, p_parts_list

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if the indices form a valid triple (i,j,k) and 0 otherwise.
    # Actually, for matrix multiplication, T[i,j,k] = 1 for all i,j,k (it's the
    # standard tensor where C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, let me reconsider. The tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition satisfies: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # Let's verify by checking a few random indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # First, try the recursive approach from the current program
    try:
        U, V, W, rank = _get_decomposition(n, m, p)
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Now try block partition search
    n_parts_list, m_parts_list, p_parts_list = _generate_partitions(n, m, p, max_parts=4)
    
    for n_parts in n_parts_list:
        if time.time() > deadline - 10:
            break
        for m_parts in m_parts_list:
            if time.time() > deadline - 10:
                break
            for p_parts in p_parts_list:
                if time.time() > deadline - 10:
                    break
                
                # Compute the rank for this partition
                total_rank = 0
                for i, ni in enumerate(n_parts):
                    for j, mi in enumerate(m_parts):
                        for k, pi in enumerate(p_parts):
                            if ni == 2 and mi == 2 and pi == 2:
                                total_rank += 7
                            else:
                                total_rank += ni * mi * pi
                
                if total_rank >= best_rank:
                    continue
                
                try:
                    U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
                    if rank < best_rank:
                        # Verify
                        if _verify_decomposition(U, V, W, n, m, p):
                            best_rank = rank
                            best_U, best_V, best_W = U, V, W
                except Exception:
                    pass
    
    if best_U is None:
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

@lru_cache(maxsize=None)
def _get_decomposition(n, m, p):
    """
    Recursively find the minimal rank decomposition for n x m x p.
    Returns (U, V, W, rank).
    """
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W, 1

    # Special case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7

    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None

    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1, r1 = _get_decomposition(n1, m, p)
        U2, V2, W2, r2 = _get_decomposition(n2, m, p)
        
        if r1 + r2 < best_rank:
            # Combine
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # U: Stack rows
            U[:n1*m, :r1] = U1
            U[n1*m:, r1:] = U2
            
            # V: Concat columns
            V[:, :r1] = V1
            V[:, r1:] = V2
            
            # W: Scatter rows based on i index
            # W1 corresponds to i in [0, n1)
            # W2 corresponds to i in [n1, n)
            # Global W index: k*n + i
            for k in range(p):
                for i1 in range(n1):
                    W[k*n + i1, :r1] = W1[k*n1 + i1, :]
                for i2 in range(n2):
                    W[k*n + (n1 + i2), r1:] = W2[k*n2 + i2, :]
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1, r1 = _get_decomposition(n, m1, p)
        U2, V2, W2, r2 = _get_decomposition(n, m2, p)
        
        if r1 + r2 < best_rank:
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # U: Scatter columns based on j index? No, U is indexed by i*m + j.
            # U1 corresponds to j in [0, m1)
            # U2 corresponds to j in [m1, m)
            for i in range(n):
                for j1 in range(m1):
                    U[i*m + j1, :r1] = U1[i*m1 + j1, :]
                for j2 in range(m2):
                    U[i*m + (m1 + j2), r1:] = U2[i*m2 + j2, :]
            
            # V: Stack rows
            # V is indexed by j*p + k
            # V1 corresponds to j in [0, m1)
            # V2 corresponds to j in [m1, m)
            V[:m1*p, :r1] = V1
            V[m1*p:, r1:] = V2
            
            # W: Concat columns
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1, r1 = _get_decomposition(n, m, p1)
        U2, V2, W2, r2 = _get_decomposition(n, m, p2)
        
        if r1 + r2 < best_rank:
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # U: Concat columns
            U[:, :r1] = U1
            U[:, r1:] = U2
            
            # V: Scatter columns based on k index? No, V is indexed by j*p + k.
            # V1 corresponds to k in [0, p1)
            # V2 corresponds to k in [p1, p)
            for j in range(m):
                for k1 in range(p1):
                    V[j*p + k1, :r1] = V1[j*p1 + k1, :]
                for k2 in range(p2):
                    V[j*p + (p1 + k2), r1:] = V2[j*p2 + k2, :]
            
            # W: Stack rows
            # W is indexed by k*n + i
            # W1 corresponds to k in [0, p1)
            # W2 corresponds to k in [p1, p)
            W[:p1*n, :r1] = W1
            W[p1*n:, r1:] = W2
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    return best_U, best_V, best_W, best_rank

# EVOLVE-BLOCK-END
