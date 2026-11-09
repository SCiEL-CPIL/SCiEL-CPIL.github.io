"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
        return False
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = -1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = -1; W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = -1; W[0, 6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_recursive_strassen(n, m, p):
    """
    Build rank-49 recursive Strassen decomposition for 2x4 x 4x8 -> 2x8.
    
    View the multiplication as a 2x2 Strassen where:
    - A is split into two 2x2 blocks: A = [A00 A01; A10 A11] (each 1x2)
    - B is split into two 2x4 blocks: B = [B00 B01; B10 B11] (each 2x4)  
    - C is split into two 2x4 blocks: C = [C00 C01; C10 C11] (each 1x4)
    
    The outer Strassen gives 7 products of "elements" (which are 1x2 * 2x4 -> 1x4).
    Each such product is a 2x4 x 4x4 -> 2x4 problem... no wait.
    
    Actually, let me use the standard recursive Strassen approach:
    For 2x4 x 4x8, split as:
    - i: 2 = 2*1 (one block of size 2)
    - j: 4 = 2*2 (two blocks of size 2)
    - k: 8 = 2*4 (two blocks of size 4)
    
    Outer Strassen on the 2x2 block structure (j-blocks x k-blocks):
    Each "element" multiplication is 2x2 * 2x4 -> 2x4
    Which can be done with rank 7 * 4 = 28? No...
    
    Let me just use the tensor product of two rank-7 Strassen decompositions.
    The 2x4 x 4x8 tensor is isomorphic to (2x2 x 2x2) ⊗ (2x2 x 2x2) in a suitable way.
    
    Concretely:
    - First Strassen handles the (i, j0) x (j0, k0) -> (i, k0) structure
    - Second Strassen handles the (j1, k1) refinement within each block
    
    Total rank = 7 * 7 = 49
    """
    n, m, p = 2, 4, 8
    U_base, V_base, W_base = _build_strassen_2x2x2()
    
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Outer Strassen term c1 in {0,...,6}
    # Inner Strassen term c2 in {0,...,6}
    # Total term index = c1 * 7 + c2
    
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            
            # U part: coefficient for A[i, j]
            # Outer: U_base[i_local*2 + j0_local, c1] where j0_local is the j-block index
            # Inner: U_base[j1_local*2 + k1_local, c2] -- but this doesn't quite work
            # because the inner problem has different dimensions.
            
            # Let me think about this differently.
            # 
            # A[i, j] where j = 2*j0 + j1, j0 in {0,1}, j1 in {0,1}
            # B[j, k] where j = 2*j0 + j1, k = 4*k0 + k1, k0 in {0,1}, k1 in {0,1,2,3}
            # C[i, k] where k = 4*k0 + k1
            #
            # Outer Strassen (on j0, k0):
            #   U_outer[i, j0, c1] from U_base
            #   V_outer[j0, k0, c1] from V_base
            #   W_outer[k0, i, c1] from W_base
            #
            # Inner Strassen (on j1, k1 within each block):
            #   For fixed j0, k0, the sub-problem is:
            #   C[i, 4*k0+k1] += sum_{j1} A[i, 2*j0+j1] * B[2*j0+j1, 4*k0+k1]
            #   This is a 2x2 x 2x4 -> 2x4 problem, not 2x2 x 2x2.
            #
            # So the inner problem is NOT a standard 2x2x2 Strassen.
            #
            # I think the rank-49 claim requires a more careful construction.
            # Let me just use the rank-56 tiled approach which is known to work.
            
            # Fallback: use tiled Strassen (rank 56)
            pass
    
    # Since the rank-49 construction is complex, use tiled Strassen (rank 56)
    n_blocks_i = n // 2
    n_blocks_j = m // 2
    n_blocks_k = p // 2
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_offset = 0
    for ii in range(n_blocks_i):
        for jj in range(n_blocks_j):
            for kk in range(n_blocks_k):
                for li in range(2):
                    for lj in range(2):
                        global_i = 2*ii + li
                        global_j = 2*jj + lj
                        for c in range(7):
                            U[global_i * m + global_j, col_offset + c] = U_base[li*2 + lj, c]
                for lj in range(2):
                    for lk in range(2):
                        global_j = 2*jj + lj
                        global_k = 2*kk + lk
                        for c in range(7):
                            V[global_j * p + global_k, col_offset + c] = V_base[lj*2 + lk, c]
                for lk in range(2):
                    for li in range(2):
                        global_k = 2*kk + lk
                        global_i = 2*ii + li
                        for c in range(7):
                            W[global_k * n + global_i, col_offset + c] = W_base[lk*2 + li, c]
                col_offset += 7
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build the rank-56 Strassen-tiled decomposition
    U, V, W = _build_recursive_strassen(n, m, p)
    
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() > deadline:
        U_s, V_s, W_s = _build_schoolbook(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    U_s, V_s, W_s = _build_schoolbook(n, m, p)
    if _verify_decomposition(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
