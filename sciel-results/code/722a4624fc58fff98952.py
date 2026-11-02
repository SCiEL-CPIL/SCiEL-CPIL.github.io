"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def _build_target():
    """Build the matrix multiplication tensor T in 3D form.
    T[i,j,k] = 1 iff the (i,j) entry of A and (j,k) entry of B
    contribute to (i,k) entry of C.
    Dimensions: n=2 (rows of A), m=2 (cols of A / rows of B), p=5 (cols of B).
    """
    n, m, p = 2, 2, 5
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T, n, m, p

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    T, n, m, p = _build_target()
    rank = U.shape[1]
    for r in range(rank):
        # Build rank-1 slice: U_col[:, :, :] * V_col * W_col
        # U_col is (n*m,), V_col is (m*p,), W_col is (p*n,)
        uc = U[:, r]  # (n*m,)
        vc = V[:, r]  # (m*p,)
        wc = W[:, r]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = T[i, j, k]
                    # The contribution to T[i,j,k] from rank r is uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
                    # We just check the full sum
    # Better: compute the full tensor reconstruction
    T_recon = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        uc = U[:, r]
        vc = V[:, r]
        wc = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    return np.allclose(T, T_recon, atol=tol, rtol=tol)

def _build_rank18():
    """Build a rank-18 decomposition using Strassen for 2x2x2 blocks.
    Split p=5 into 2+2+1.
    Block 1: 2x2x2 (cols 0,1) -> Strassen rank 7
    Block 2: 2x2x2 (cols 2,3) -> Strassen rank 7
    Block 3: 2x2x1 (col 4)    -> schoolbook rank 4
    Total: 18
    """
    n, m, p = 2, 2, 5
    rank = 18
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen 2x2x2 decomposition (standard)
    # The 7 products are:
    # P1 = (A11)(B12-B22)
    # P2 = (A11+A12)(B22)
    # P3 = (A21+A22)(B11)
    # P4 = (A22)(B21-B11)
    # P5 = (A11+A22)(B11+B22)
    # P6 = (A12-A22)(B21+B22)
    # P7 = (A21-A11)(B11+B12)
    # C11 = P5 + P4 - P2 - P7
    # C12 = P1 + P2
    # C21 = P3 + P4
    # C22 = P5 + P1 - P3 - P6
    
    strassen_U = [
        # i*m+j index: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # P1: (A11)(B12-B22) -> A: [1,0,0,0], B: [0,1,-1,0], C: [0,0,0,0] (contributes to C12)
        [1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 0],
        # P2: (A11+A12)(B22) -> A: [1,1,0,0], B: [0,0,0,1], C: [0,0,0,0]
        [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
        # P3: (A21+A22)(B11) -> A: [0,0,1,1], B: [1,0,0,0], C: [0,0,0,0]
        [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        # P4: (A22)(B21-B11) -> A: [0,0,0,1], B: [-1,0,1,0], C: [0,0,0,0]
        [0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, -1, 0],
        # P5: (A11+A22)(B11+B22) -> A: [1,0,0,1], B: [1,0,0,1], C: [0,0,0,0]
        [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1],
        # P6: (A12-A22)(B21+B22) -> A: [0,1,0,-1], B: [0,1,0,1], C: [0,0,0,0]
        [0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 0, -1],
        # P7: (A21-A11)(B11+B12) -> A: [0,0,1,-1], B: [1,1,0,0], C: [0,0,0,0]
        [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, -1],
    ]
    # Wait, I need to be more careful. The W factor should encode how P's combine into C.
    # Let me redo this properly.
    
    # For a 2x2x2 block with output columns starting at col_offset:
    # The 7 rank-1 terms have:
    # U factor: (4,) vector for A indices
    # V factor: (4,) vector for B indices (j*p+k, but within the block, k is 0 or 1)
    # W factor: (4,) vector for C indices (k*n+i, within the block, k is 0 or 1)
    
    # Let me define the 7 Strassen terms properly:
    # Term r: U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] contributes to T[i,j,k]
    
    # For the block at columns [c0, c1]:
    # k_local = 0 -> k_global = c0, k_local = 1 -> k_global = c1
    
    def add_strassen_block(U, V, W, col_offset, start_rank):
        n, m, p = 2, 2, 5
        # 7 terms
        # Each term is (u_vec, v_vec, w_vec)
        # u_vec has 4 entries (i*m+j for i,j in 0,1)
        # v_vec has 4 entries (j*p+k for j in 0,1, k_local in 0,1) -> but globally j*p+k_global
        # w_vec has 4 entries (k*n+i for k_local in 0,1, i in 0,1) -> globally k_global*n+i
        
        terms = [
            # P1 = (A11)(B12-B22)
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            # P2 = (A11+A12)(B22)
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),
            # P3 = (A21+A22)(B11)
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]),
            # P4 = (A22)(B21-B11)
            ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, -1, 0]),
            # P5 = (A11+A22)(B11+B22)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # P6 = (A12-A22)(B21+B22)
            ([0, 1, 0, -1], [0, 1, 0, 1], [0, 0, 0, -1]),
            # P7 = (A21-A11)(B11+B12)
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, -1]),
        ]
        
        for r, (u_vec, v_vec, w_vec) in enumerate(terms):
            col = start_rank + r
            for idx in range(4):
                U[idx, col] = u_vec[idx]
            # V indices: j*p + k_global, j in 0,1, k_local in 0,1
            # v_vec[0] = j=0, k_local=0 -> j*p + (col_offset+0)
            # v_vec[1] = j=0, k_local=1 -> j*p + (col_offset+1)
            # v_vec[2] = j=1, k_local=0 -> j*p + (col_offset+0)
            # v_vec[3] = j=1, k_local=1 -> j*p + (col_offset+1)
            V[0 * p + col_offset + 0, col] = v_vec[0]
            V[0 * p + col_offset + 1, col] = v_vec[1]
            V[1 * p + col_offset + 0, col] = v_vec[2]
            V[1 * p + col_offset + 1, col] = v_vec[3]
            # W indices: k*n + i, k_local in 0,1, i in 0,1
            # w_vec[0] = k_local=0, i=0 -> (col_offset+0)*n + 0
            # w_vec[1] = k_local=0, i=1 -> (col_offset+0)*n + 1
            # w_vec[2] = k_local=1, i=0 -> (col_offset+1)*n + 0
            # w_vec[3] = k_local=1, i=1 -> (col_offset+1)*n + 1
            W[(col_offset + 0) * n + 0, col] = w_vec[0]
            W[(col_offset + 0) * n + 1, col] = w_vec[1]
            W[(col_offset + 1) * n + 0, col] = w_vec[2]
            W[(col_offset + 1) * n + 1, col] = w_vec[3]
    
    # Block 1: cols 0,1
    add_strassen_block(U, V, W, 0, 0)
    # Block 2: cols 2,3
    add_strassen_block(U, V, W, 2, 7)
    
    # Block 3: col 4, schoolbook 2x2x1, rank 4
    # For single column k=4:
    # (i,j) in 0..1 x 0..1
    # T[i,j,4] = 1 for all i,j
    # Schoolbook: for each (i,j,k), U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    for i in range(2):
        for j in range(2):
            col = 14 + i * 2 + j
            U[i * 2 + j, col] = 1.0
            V[j * 5 + 4, col] = 1.0
            W[4 * 2 + i, col] = 1.0
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    """Build the schoolbook rank-20 decomposition."""
    n, m, p = 2, 2, 5
    rank = n * m * p  # 20
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 2x2x5 matrix multiplication.
    Uses Strassen-based rank-18 decomposition.
    """
    n, m, p = 2, 2, 5
    
    # Start with rank-18
    result = _build_rank18()
    
    # Verify
    if _check_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    result = _build_schoolbook()
    return result
# EVOLVE-BLOCK-END
