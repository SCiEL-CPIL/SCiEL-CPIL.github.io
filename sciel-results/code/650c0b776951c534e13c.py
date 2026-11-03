<<<<<<< SEARCH
"""Matrix multiplication 3x5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

_START_TIME = time.time()
DEADLINE = _START_TIME + 110.0

def _check_deadline():
    return time.time() > DEADLINE

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    
    Returns {'U': list of lists (15, rank), 'V': list of lists (25, rank), 'W': list of lists (15, rank)}
    where U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] exactly decompose the multiplication tensor.
    """
    n, m, p = 3, 5, 5
    nm = n * m  # 15
    mp = m * p  # 25
    pn = p * n  # 15
    
    # Schoolbook decomposition: rank = n*m*p = 75
    # For each (i, j, k), we have a rank-1 term
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Verify the schoolbook decomposition
    def verify(U_mat, V_mat, W_mat, rank_val):
        """Check if the decomposition correctly computes matrix multiplication."""
        rng = np.random.RandomState(42)
        for trial in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(rank_val):
                u_c = np.array(U_mat)[:, c].reshape(n, m)
                v_c = np.array(V_mat)[:, c].reshape(m, p)
                w_c = np.array(W_mat)[:, c].reshape(p, n)
                z_c = np.sum(u_c * A) * np.sum(v_c * B)
                C_computed += w_c.T * z_c
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    if not verify(U, V, W, rank):
        # Should never happen for schoolbook, but return as-is
        pass
    
    # Try to find a lower rank decomposition using structured block decomposition
    # Strategy: Split the 5x5 middle dimension into 3x3 + 1x1 blocks
    # This gives us terms of type (3,3,3), (3,3,1), (3,1,3), (3,1,1)
    # 
    # Actually, let me try a different approach: use the fact that for matrix
    # multiplication, we can sometimes combine terms.
    #
    # For now, let's try a simple optimization: check if we can reduce by 1
    # by finding a linear dependency in the schoolbook terms.
    
    # Given the complexity and time constraints, return the schoolbook decomposition
    # which is guaranteed correct with rank 75.
    
    return {"U": U, "V": V, "W": W}
=======
"""Matrix multiplication 3x5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
DEADLINE = _START_TIME + 105.0

def _check_deadline():
    return time.time() > DEADLINE

def _verify_decomposition(U, V, W, n, m, p, num_trials=10, atol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rng = np.random.RandomState(42)
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    for trial in range(num_trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        
        for c in range(rank):
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            z_c = np.sum(u_c * A) * np.sum(v_c * B)
            C_computed += w_c.T * z_c
        
        if not np.allclose(C_computed, C_expected, atol=atol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return U, V, W

def _try_block_decomposition(n, m, p, split_m, split_p, time_budget=60.0):
    """
    Try to find a lower-rank decomposition by splitting dimensions.
    
    Split m into blocks and p into blocks, then use known optimal decompositions
    for each block size.
    
    For 3x5x5, we can split:
    - m=5 into 3+2 or 4+1
    - p=5 into 3+2 or 4+1
    """
    n, m, p = 3, 5, 5
    
    # Known optimal ranks for small matrix multiplications:
    # R(1,1,1) = 1
    # R(2,2,2) = 7 (Strassen)
    # R(3,3,3) = 23 (best known)
    # R(1,k,k) = k
    # R(k,1,k) = k
    # R(k,k,1) = k
    # R(2,2,1) = 2
    # R(2,1,2) = 2
    # R(1,2,2) = 2
    # R(2,2,2) = 7
    # R(3,3,3) = 23 (or 21 in some constructions)
    
    def get_block_rank(n1, m1, p1):
        """Get the best known rank for n1 x m1 x p1 matrix multiplication."""
        if n1 == 1 and m1 == 1 and p1 == 1:
            return 1
        if n1 == 1:
            return m1 * p1
        if m1 == 1:
            return n1 * p1
        if p1 == 1:
            return n1 * m1
        if n1 == 2 and m1 == 2 and p1 == 2:
            return 7  # Strassen
        if n1 == 3 and m1 == 3 and p1 == 3:
            return 23  # Best known
        if n1 == 2 and m1 == 2:
            return 4 * p1
        if n1 == 2 and p1 == 2:
            return 4 * m1
        if m1 == 2 and p1 == 2:
            return 4 * n1
        # Fallback to schoolbook
        return n1 * m1 * p1
    
    # Try different block splits
    best_rank = n * m * p  # Start with schoolbook
    best_decomp = None
    
    m_splits = [
        (3, 2),
        (2, 3),
        (4, 1),
        (1, 4),
        (5,),
        (2, 2, 1),
        (1, 2, 2),
        (3, 1, 1),
        (1, 3, 1),
        (1, 1, 3),
    ]
    
    p_splits = [
        (3, 2),
        (2, 3),
        (4, 1),
        (1, 4),
        (5,),
        (2, 2, 1),
        (1, 2, 2),
        (3, 1, 1),
        (1, 3, 1),
        (1, 1, 3),
    ]
    
    start_time = time.time()
    
    for m_split in m_splits:
        for p_split in p_splits:
            if time.time() - start_time > time_budget:
                break
            if _check_deadline():
                return None
            
            total_rank = 0
            for m_block in m_split:
                for p_block in p_split:
                    total_rank += get_block_rank(n, m_block, p_block)
            
            if total_rank < best_rank:
                best_rank = total_rank
                # Build this decomposition
                best_decomp = _build_block_decomp(n, m, p, m_split, p_split)
    
    return best_rank, best_decomp

def _build_block_decomp(n, m, p, m_split, p_split):
    """Build a block decomposition."""
    rank = 0
    for mb in m_split:
        for pb in p_split:
            rank += _get_block_rank(n, mb, pb)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    m_offset = 0
    for mb in m_split:
        p_offset = 0
        for pb in p_split:
            # Build decomposition for this block
            block_U, block_V, block_W, block_rank = _build_simple_block(n, mb, pb)
            
            for c in range(block_rank):
                for i in range(n):
                    for j in range(mb):
                        U[i * m + (m_offset + j)][col + c] = block_U[i * mb + j][c]
                for j in range(mb):
                    for k in range(pb):
                        V[(m_offset + j) * p + (p_offset + k)][col + c] = block_V[j * pb + k][c]
                for k in range(pb):
                    for i in range(n):
                        W[(p_offset + k) * n + i][col + c] = block_W[k * n + i][c]
            
            col += block_rank
            p_offset += pb
        m_offset += mb
    
    return {"U": U, "V": V, "W": W}

def _get_block_rank(n, m, p):
    """Get the rank for a block decomposition."""
    if n == 1:
        return m * p
    if m == 1:
        return n * p
    if p == 1:
        return n * m
    if n == 2 and m == 2 and p == 2:
        return 7
    if n == 3 and m == 3 and p == 3:
        return 23
    return n * m * p

def _build_simple_block(n, m, p):
    """Build a simple (schoolbook) block decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return U, V, W, rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    
    Returns {'U': list of lists (15, rank), 'V': list of lists (25, rank), 'W': list of lists (15, rank)}
    where U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] exactly decompose the multiplication tensor.
    """
    n, m, p = 3, 5, 5
    
    # Start with schoolbook as fallback
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try block decomposition
    if not _check_deadline():
        result = _try_block_decomposition(n, m, p, 5, 5, time_budget=90.0)
        if result is not None:
            block_rank, block_decomp = result
            if block_rank < best_rank and block_decomp is not None:
                # Verify the block decomposition
                if _verify_decomposition(block_decomp['U'], block_decomp['V'], block_decomp['W'], n, m, p):
                    best_rank = block_rank
                    U, V, W = block_decomp['U'], block_decomp['V'], block_decomp['W']
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p, num_trials=5):
        # Fall back to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
