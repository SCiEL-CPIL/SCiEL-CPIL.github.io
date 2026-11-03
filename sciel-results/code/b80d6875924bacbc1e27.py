"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_222():
    """
    Returns Strassen decomposition for 2x2x2 matrix multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)*(B00+B11) -> C00 += M1, C01 -= M3, C10 -= M4, C11 += M1
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[1, 0] = 1  # B11
    W[0, 0] = 1  # C00 (k=0, i=0)
    W[1, 0] = 1  # C01 (k=1, i=0)
    W[2, 0] = 1  # C10 (k=0, i=1)
    W[3, 0] = 1  # C11 (k=1, i=1)
    
    # M2: (A10+A11)*B00 -> C10 += M2
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 (k=0, i=1)
    
    # M3: A00*(B01-B11) -> C01 -= M3
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1  # B11
    W[1, 2] = -1  # C01 (k=1, i=0)
    
    # M4: A11*(B10-B00) -> C10 -= M4
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1  # B00
    W[2, 3] = -1  # C10 (k=0, i=1)
    
    # M5: (A00+A01)*B11 -> C11 += M5
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[3, 4] = 1  # C11 (k=1, i=1)
    
    # M6: (A10-A00)*(B00+B01) -> C01 += M6, C11 += M6
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[1, 5] = 1  # C01 (k=1, i=0)
    W[3, 5] = 1  # C11 (k=1, i=1)
    
    # M7: (A01-A11)*(B10+B11) -> C00 += M7, C10 += M7
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00 (k=0, i=0)
    W[2, 6] = 1  # C10 (k=0, i=1)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Test with random matrices
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    for j in range(m):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _snap_to_half_integer(mat):
    """Snap values to nearest half-integer if close."""
    mat = np.array(mat, dtype=float)
    snapped = np.round(mat * 2) / 2
    if np.allclose(mat, snapped, atol=1e-9):
        return snapped
    return mat

def _build_2x5x6():
    """
    Build 2x5x6 decomposition using block Strassen.
    Split m=5 into [2, 2, 1] and p=6 into [2, 2, 2].
    
    Blocks:
    - 6 blocks of 2x2x2 using Strassen (rank 7 each) -> 42
    - 3 blocks of 1x2x2 using schoolbook (rank 2 each) -> 6
    Total rank: 48
    """
    n, m, p = 2, 5, 6
    
    # m partitions: [2, 2, 1]
    # p partitions: [2, 2, 2]
    
    # For each m_block_size and p_block_size:
    # If m_block=2 and p_block=2: use Strassen (rank 7)
    # Otherwise: use schoolbook (rank m_block * p_block)
    
    sU, sV, sW = _strassen_222()  # 4x7 each
    
    # Calculate total rank
    rank = 0
    m_offsets = [0, 2, 4]  # starting indices for m blocks of size [2, 2, 1]
    p_offsets = [0, 2, 4]  # starting indices for p blocks of size [2, 2, 2]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]
    
    for mi in range(3):
        for pi in range(3):
            ms = m_sizes[mi]
            ps = p_sizes[pi]
            if ms == 2 and ps == 2:
                rank += 7
            else:
                rank += ms * ps
    
    U = np.zeros((n * m, rank))  # 10 x rank
    V = np.zeros((m * p, rank))  # 30 x rank
    W = np.zeros((p * n, rank))  # 12 x rank
    
    col = 0
    for mi in range(3):
        m_start = m_offsets[mi]
        ms = m_sizes[mi]
        for pi in range(3):
            p_start = p_offsets[pi]
            ps = p_sizes[pi]
            
            if ms == 2 and ps == 2:
                # Use Strassen for 2x2x2 block
                for r in range(7):
                    # U: A[i, m_start+j], i in {0,1}, j in {0,1}
                    U[0 * m + m_start + 0, col] = sU[0, r]  # i=0, j=0
                    U[0 * m + m_start + 1, col] = sU[1, r]  # i=0, j=1
                    U[1 * m + m_start + 0, col] = sU[2, r]  # i=1, j=0
                    U[1 * m + m_start + 1, col] = sU[3, r]  # i=1, j=1
                    
                    # V: B[m_start+j, p_start+k], j in {0,1}, k in {0,1}
                    V[(m_start + 0) * p + p_start + 0, col] = sV[0, r]  # j=0, k=0
                    V[(m_start + 0) * p + p_start + 1, col] = sV[1, r]  # j=0, k=1
                    V[(m_start + 1) * p + p_start + 0, col] = sV[2, r]  # j=1, k=0
                    V[(m_start + 1) * p + p_start + 1, col] = sV[3, r]  # j=1, k=1
                    
                    # W: C[p_start+k, i], k in {0,1}, i in {0,1}
                    W[(p_start + 0) * n + 0, col] = sW[0, r]  # k=0, i=0
                    W[(p_start + 0) * n + 1, col] = sW[1, r]  # k=0, i=1
                    W[(p_start + 1) * n + 0, col] = sW[2, r]  # k=1, i=0
                    W[(p_start + 1) * n + 1, col] = sW[3, r]  # k=1, i=1
                    
                    col += 1
            else:
                # Schoolbook for ms x ps block
                for i in range(n):
                    for j in range(ms):
                        for k in range(ps):
                            U[i * m + m_start + j, col] = 1
                            V[(m_start + j) * p + p_start + k, col] = 1
                            W[(p_start + k) * n + i, col] = 1
                            col += 1
    
    assert col == rank, f"Expected col={rank}, got {col}"
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
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

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (A*B)[i,k]
    """
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition first (rank 48)
    U, V, W = _build_2x5x6()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        # Snap to half-integers
        U = _snap_to_half_integer(U)
        V = _snap_to_half_integer(V)
        W = _snap_to_half_integer(W)
        
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    U = _snap_to_half_integer(U)
    V = _snap_to_half_integer(V)
    W = _snap_to_half_integer(W)
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
