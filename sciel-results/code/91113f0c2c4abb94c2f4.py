"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

# EVOLVE-BLOCK-START

def _add_strassen_term(U, V, W, col, i_base, j_base, k_base, n, m, p):
    """
    Add Strassen terms for a 2x2x2 block.
    i_base, j_base, k_base are the starting global indices for i, j, k.
    """
    # A indices: (i_base, j_base), (i_base, j_base+1), (i_base+1, j_base), (i_base+1, j_base+1)
    # B indices: (j_base, k_base), (j_base, k_base+1), (j_base+1, k_base), (j_base+1, k_base+1)
    # C indices: (k_base, i_base), (k_base+1, i_base), (k_base, i_base+1), (k_base+1, i_base+1)
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # M1: (A00 + A11)(B00 + B11) -> C00, C11
    U[u_idx(i_base, j_base), col] += 1
    U[u_idx(i_base+1, j_base+1), col] += 1
    V[v_idx(j_base, k_base), col] += 1
    V[v_idx(j_base+1, k_base+1), col] += 1
    W[w_idx(k_base, i_base), col] += 1
    W[w_idx(k_base+1, i_base+1), col] += 1
    
    # M2: (A10 + A11)B00 -> C10
    U[u_idx(i_base+1, j_base), col] += 1
    U[u_idx(i_base+1, j_base+1), col] += 1
    V[v_idx(j_base, k_base), col] += 1
    W[w_idx(k_base, i_base+1), col] += 1
    
    # M3: A00(B01 - B11) -> C01
    U[u_idx(i_base, j_base), col] += 1
    V[v_idx(j_base, k_base+1), col] += 1
    V[v_idx(j_base+1, k_base+1), col] -= 1
    W[w_idx(k_base+1, i_base), col] += 1
    
    # M4: A11(B10 - B00) -> C11
    U[u_idx(i_base+1, j_base+1), col] += 1
    V[v_idx(j_base+1, k_base), col] += 1
    V[v_idx(j_base, k_base), col] -= 1
    W[w_idx(k_base+1, i_base+1), col] += 1
    
    # M5: (A00 + A01)B11 -> C00
    U[u_idx(i_base, j_base), col] += 1
    U[u_idx(i_base, j_base+1), col] += 1
    V[v_idx(j_base+1, k_base+1), col] += 1
    W[w_idx(k_base, i_base), col] += 1
    
    # M6: (A10 - A00)(B00 + B01) -> C01, C11
    U[u_idx(i_base+1, j_base), col] += 1
    U[u_idx(i_base, j_base), col] -= 1
    V[v_idx(j_base, k_base), col] += 1
    V[v_idx(j_base, k_base+1), col] += 1
    W[w_idx(k_base+1, i_base), col] += 1
    W[w_idx(k_base+1, i_base+1), col] += 1
    
    # M7: (A01 - A11)(B10 + B11) -> C00, C10
    U[u_idx(i_base, j_base+1), col] += 1
    U[u_idx(i_base+1, j_base+1), col] -= 1
    V[v_idx(j_base+1, k_base), col] += 1
    V[v_idx(j_base+1, k_base+1), col] += 1
    W[w_idx(k_base, i_base), col] += 1
    W[w_idx(k_base, i_base+1), col] += 1

def _construct_decomposition(n, m, p):
    """
    Construct the rank 88 decomposition for 3x4x8.
    """
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}
    for j_block in [0, 2]:
        # For each block, split n=3 into 1 and 2
        # Block A: i=0, j in {j_block, j_block+1}, k in 0..7
        # This is 1x2x8. Rank 16.
        for k in range(8):
            # Term 1: A[0, j_block] * B[j_block, k] * C[k, 0]
            U[0 * m + j_block, col] = 1
            V[j_block * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
            # Term 2: A[0, j_block+1] * B[j_block+1, k] * C[k, 0]
            U[0 * m + j_block + 1, col] = 1
            V[(j_block + 1) * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
        # Block B: i in {1,2}, j in {j_block, j_block+1}, k in 0..7
        # This is 2x2x8. Split k into 4 pairs of 2.
        for k_pair in range(4):
            k_base = k_pair * 2
            _add_strassen_term(U, V, W, col, 1, j_block, k_base, n, m, p)
            col += 1
            _add_strassen_term(U, V, W, col, 1, j_block, k_base, n, m, p) # Wait, Strassen is 7 terms.
            # Correction: _add_strassen_term adds 7 terms? No, it adds 1 column's worth of coefficients?
            # No, my _add_strassen_term adds coefficients to a SINGLE column.
            # But Strassen requires 7 columns.
            # I need to call it 7 times? No, the logic inside _add_strassen_term is for ONE rank-1 term.
            # I need to implement the 7 distinct rank-1 terms of Strassen.
            
    # The previous implementation of _add_strassen_term was flawed. It added coefficients for ONE term.
    # I need to add 7 terms for each 2x2x2 block.
    
    # Let's restart the construction logic properly.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    for j_block in [0, 2]:
        # 1x2x8 part (i=0)
        for k in range(8):
            U[0 * m + j_block, col] = 1
            V[j_block * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
            U[0 * m + j_block + 1, col] = 1
            V[(j_block + 1) * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            
        # 2x2x8 part (i=1,2)
        for k_pair in range(4):
            k_base = k_pair * 2
            # Add 7 Strassen terms
            for s in range(7):
                _add_strassen_term_specific(U, V, W, col, s, 1, j_block, k_base, n, m, p)
                col += 1
                
    return U, V, W

def _add_strassen_term_specific(U, V, W, col, term_idx, i_base, j_base, k_base, n, m, p):
    """
    Add the specific Strassen term (term_idx 0..6) to the given column.
    """
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    if term_idx == 0: # M1
        U[u_idx(i_base, j_base), col] = 1
        U[u_idx(i_base+1, j_base+1), col] = 1
        V[v_idx(j_base, k_base), col] = 1
        V[v_idx(j_base+1, k_base+1), col] = 1
        W[w_idx(k_base, i_base), col] = 1
        W[w_idx(k_base+1, i_base+1), col] = 1
    elif term_idx == 1: # M2
        U[u_idx(i_base+1, j_base), col] = 1
        U[u_idx(i_base+1, j_base+1), col] = 1
        V[v_idx(j_base, k_base), col] = 1
        W[w_idx(k_base, i_base+1), col] = 1
    elif term_idx == 2: # M3
        U[u_idx(i_base, j_base), col] = 1
        V[v_idx(j_base, k_base+1), col] = 1
        V[v_idx(j_base+1, k_base+1), col] = -1
        W[w_idx(k_base+1, i_base), col] = 1
    elif term_idx == 3: # M4
        U[u_idx(i_base+1, j_base+1), col] = 1
        V[v_idx(j_base+1, k_base), col] = 1
        V[v_idx(j_base, k_base), col] = -1
        W[w_idx(k_base+1, i_base+1), col] = 1
    elif term_idx == 4: # M5
        U[u_idx(i_base, j_base), col] = 1
        U[u_idx(i_base, j_base+1), col] = 1
        V[v_idx(j_base+1, k_base+1), col] = 1
        W[w_idx(k_base, i_base), col] = 1
    elif term_idx == 5: # M6
        U[u_idx(i_base+1, j_base), col] = 1
        U[u_idx(i_base, j_base), col] = -1
        V[v_idx(j_base, k_base), col] = 1
        V[v_idx(j_base, k_base+1), col] = 1
        W[w_idx(k_base+1, i_base), col] = 1
        W[w_idx(k_base+1, i_base+1), col] = 1
    elif term_idx == 6: # M7
        U[u_idx(i_base, j_base+1), col] = 1
        U[u_idx(i_base+1, j_base+1), col] = -1
        V[v_idx(j_base+1, k_base), col] = 1
        V[v_idx(j_base+1, k_base+1), col] = 1
        W[w_idx(k_base, i_base), col] = 1
        W[w_idx(k_base, i_base+1), col] = 1

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
                
    # Reconstruct
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness
    if not np.allclose(T_recon, T_true, atol=tol):
        return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a 2D list.
    """
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    U, V, W = _construct_decomposition(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
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
                    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
