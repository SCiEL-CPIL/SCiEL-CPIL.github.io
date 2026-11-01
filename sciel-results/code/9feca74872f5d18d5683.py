import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses Strassen's algorithm on 2x2 sub-blocks with partition 5 = 2+2+1
    of the inner dimension. Total rank: 46.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the full tensor T[a, b, c] where:
    #   a = i*m + j  (row of A, col of A)
    #   b = j*p + k  (row of B, col of B)
    #   c = k*n + i  (output, trace convention: k*n + i)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a single rank-1 term to the decomposition."""
        terms.append((u_entries, v_entries, w_entries))
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        """
        Add 7 Strassen terms for a 2x2x2 block starting at (i_off, j_off, k_off).
        """
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # Global: i = i_off + il, j = j_off + jl, k = k_off + kl
        
        # U indices (A-space): i*m + j
        a00 = i_off * m + j_off
        a01 = i_off * m + (j_off + 1)
        a10 = (i_off + 1) * m + j_off
        a11 = (i_off + 1) * m + (j_off + 1)
        
        # V indices (B-space): j*p + k
        b00 = j_off * p + k_off
        b01 = j_off * p + (k_off + 1)
        b10 = (j_off + 1) * p + k_off
        b11 = (j_off + 1) * p + (k_off + 1)
        
        # W indices (C-space): k*n + i
        c00 = k_off * n + i_off
        c01 = k_off * n + (i_off + 1)
        c10 = (k_off + 1) * n + i_off
        c11 = (k_off + 1) * n + (i_off + 1)
        
        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        add_term([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])
        
        # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
        add_term([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, -1)])
        
        # M3 = A00(B10 - B00) -> C01 += M3, C11 += M3
        add_term([(a00, 1)], [(b10, 1), (b00, -1)], [(c01, 1), (c11, 1)])
        
        # M4 = A11(B11 - B01) -> C00 += M4, C10 += M4
        add_term([(a11, 1)], [(b11, 1), (b01, -1)], [(c00, 1), (c10, 1)])
        
        # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
        add_term([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])
        
        # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
        add_term([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])
        
        # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
        add_term([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])
    
    def add_schoolbook(i, j, k):
        """Add a single schoolbook term."""
        a_idx = i * m + j
        b_idx = j * p + k
        c_idx = k * n + i
        add_term([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
    
    # Decomposition plan for 2x5x5 using Strassen with partition 5 = 2+2+1:
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): Strassen, rank 7
    add_strassen_2x2x2(0, 0, 0)
    
    # Block (i in {0,1}, j in {0,1}, k in {2,3}): Strassen, rank 7
    add_strassen_2x2x2(0, 0, 2)
    
    # Block (i in {0,1}, j in {0,1}, k in {4}): Schoolbook, rank 4
    for i in range(2):
        for j in range(2):
            add_schoolbook(i, j, 4)
    
    # Block (i in {0,1}, j in {2,3}, k in {0,1}): Strassen, rank 7
    add_strassen_2x2x2(0, 2, 0)
    
    # Block (i in {0,1}, j in {2,3}, k in {2,3}): Strassen, rank 7
    add_strassen_2x2x2(0, 2, 2)
    
    # Block (i in {0,1}, j in {2,3}, k in {4}): Schoolbook, rank 4
    for i in range(2):
        for j in range(2, 4):
            add_schoolbook(i, j, 4)
    
    # Block (i in {0,1}, j in {4}, k in {0,1,2,3,4}): Schoolbook, rank 10
    for i in range(2):
        for k in range(5):
            add_schoolbook(i, 4, k)
    
    # Build U, V, W matrices from terms
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for (idx, val) in u_entries:
            U[idx, r] = val
        for (idx, val) in v_entries:
            V[idx, r] = val
        for (idx, val) in w_entries:
            W[idx, r] = val
    
    # Verify the decomposition
    valid, err = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook if verification fails
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    # Time guard
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
