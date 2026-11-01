"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

_START_TIME = time.time()
_TIMEOUT = 140  # seconds, safety margin before 150s kill

def _get_target_tensor(n, m, p):
    """
    Create the sparse target tensor T for matrix multiplication.
    T[i, j, k] = 1 if i corresponds to A[a,b], j to B[b,f], k to C[a,f].
    i = a*m + b
    j = b*p + f
    k = a*p + f  <-- WAIT.
    The task says: "output factor follows the official trace convention W[k*n+i,r]"
    This implies the output index is $k \cdot n + i$?
    Let's re-read carefully: "W[k*n+i,r]".
    Usually W is the output matrix.
    If W has shape (p*n, rank), then the index is $k \cdot n + i$.
    Here $n=2, p=9$. So index is $9k + i$? Or $k \cdot 2 + i$?
    The prompt says: "W[k*n+i,r]".
    If $k$ is the column index of B (0..8) and $i$ is the row index of A (0..1)?
    Then $W$ index is $k \cdot 2 + i$.
    This matches $C_{i,k}$ being stored in row $k$, col $i$? Or just flat index.
    Let's check the shape: (p*n, rank) = (18, rank).
    If index is $k \cdot n + i$, with $k \in 0..8, i \in 0..1$, max index is $8 \cdot 2 + 1 = 17$. Correct.
    So $W$ index $k_{out} = k_{col} \cdot n + i_{row}$.
    
    Let's check U and V.
    U shape (n*m, rank) = (4, rank).
    V shape (m*p, rank) = (18, rank).
    
    Standard multiplication: $C = A B$.
    $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    Here $i \in 0..1$ (row A), $j \in 0..1$ (col A / row B), $k \in 0..8$ (col B).
    
    U index: Corresponds to A. $i_U = i \cdot m + j = 2i + j$.
    V index: Corresponds to B. $j_V = j \cdot p + k = 9j + k$.
    W index: Corresponds to C. $k_W = k \cdot n + i = 9k + i$. (Based on "W[k*n+i,r]").
    
    So non-zero entries of T:
    For each $i \in 0..1, j \in 0..1, k \in 0..8$:
    $T[2i+j, 9j+k, 9k+i] = 1$.
    """
    n, m, p = 2, 2, 9
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition.
    4 blocks of 2x2x2 (Strassen) + 1 block of 2x2x1 (Schoolbook).
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen formulas for 2x2x2
    # Inputs: A (2x2), B (2x2). Outputs: C (2x2).
    # Indices in local block:
    # A: 0,1,2,3 (row-major)
    # B: 0,1,2,3 (row-major)
    # C: 0,1,2,3 (row-major)
    # Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U coeffs (A), V coeffs (B), W coeffs (C)
    # Note: W index in local block is $k \cdot n + i$.
    # For 2x2x2 block, $n=2, p=2$.
    # W local index: $k_{local} \cdot 2 + i_{local}$.
    # C00 (i=0,k=0) -> idx 0
    # C01 (i=0,k=1) -> idx 1
    # C10 (i=1,k=0) -> idx 2
    # C11 (i=1,k=1) -> idx 3
    
    strassen_terms = [
        # M1
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (1, 1), (2, 1), (3, 1) ]), # C00+=, C01+=, C10+=, C11+=
        # M2
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]), # C10+=, C11-=
        # M3
        ([ (0, 1) ], [ (2, 1), (3, -1) ], [ (1, 1), (3, 1) ]), # C01+=, C11+=
        # M4
        ([ (3, 1) ], [ (1, 1), (0, -1) ], [ (0, 1), (2, -1) ]), # C00+=, C10-=
        # M5
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]), # C01+=, C11+=
        # M6
        ([ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ], [ (0, -1), (2, 1) ]), # C00-=, C10+=
        # M7
        ([ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ], [ (0, 1), (1, 1) ]), # C00+=, C01+=
    ]
    
    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx_off, coeff in v_coeffs:
                # B index: j*p + k. Block covers k=2*block .. 2*block+1
                # v_idx_off is 0..3 (local B index).
                # Local B index $b_{loc} = j_{loc} \cdot 2 + k_{loc}$.
                # Global B index $j_{glob} \cdot 9 + k_{glob}$.
                # $j_{glob} = j_{loc}$ (since j is 0 or 1, and blocks are in k).
                # $k_{glob} = 2 \cdot block + k_{loc}$.
                # So $V_{glob} = j_{loc} \cdot 9 + 2 \cdot block + k_{loc}$.
                # $b_{loc} = 2 j_{loc} + k_{loc}$.
                # This mapping is not simply $2 \cdot block + v_{loc}$.
                # We need to map local B index to global B index.
                j_loc = v_idx_off // 2
                k_loc = v_idx_off % 2
                v_idx = j_loc * 9 + (2 * block + k_loc)
                V[v_idx, col] += coeff
            for w_idx_off, coeff in w_coeffs:
                # W index: k*n + i.
                # Local W index $w_{loc} = k_{loc} \cdot 2 + i_{loc}$.
                # Global W index $k_{glob} \cdot 2 + i_{loc}$.
                # $k_{glob} = 2 \cdot block + k_{loc}$.
                k_loc = w_idx_off // 2
                i_loc = w_idx_off % 2
                w_idx = (2 * block + k_loc) * 2 + i_loc
                W[w_idx, col] += coeff
    
    # Schoolbook for 2x2x1 (block 4, k=8)
    schoolbook_terms = [
        # C08 = A00 B08
        ([ (0, 1) ], [ (8, 1) ], [ (16, 1) ]), # W idx: 8*2+0 = 16
        # C08 += A01 B18
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]), # B idx: 1*9+8=17
        # C18 = A10 B08
        ([ (2, 1) ], [ (8, 1) ], [ (17, 1) ]), # W idx: 8*2+1 = 17
        # C18 += A11 B18
        ([ (3, 1) ], [ (17, 1) ], [ (17, 1) ]),
    ]
    
    col_start = 4 * 7
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
    
    return U, V, W

def _validate_decomposition(U, V, W, n, m, p):
    T = _get_target_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T, T_recon, atol=1e-10)

def _try_rank_31(n, m, p):
    """
    Try to find a rank-31 decomposition by optimizing.
    """
    U32, V32, W32 = _build_rank_32(n, m, p)
    T_target = _get_target_tensor(n, m, p)
    
    # Flatten target
    T_flat = T_target.flatten()
    
    # Try removing each term
    for removed in range(32):
        if time.time() - _START_TIME > _TIMEOUT - 10:
            break
        
        others = [i for i in range(32) if i != removed]
        U_init = U32[:, others].flatten()
        V_init = V32[:, others].flatten()
        W_init = W32[:, others].flatten()
        
        x0 = np.concatenate([U_init, V_init, W_init])
        
        n_u = n * m * 31
        n_v = m * p * 31
        n_w = p * n * 31
        
        def residual(x):
            U = x[:n_u].reshape((n*m, 31))
            V = x[n_u:n_u+n_v].reshape((m*p, 31))
            W = x[n_u+n_v:].reshape((p*n, 31))
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            return T_recon.flatten() - T_flat
        
        try:
            # Use least_squares
            res = opt.least_squares(residual, x0, method='lm', max_nfev=200)
            if res.cost < 1e-10:
                U = res.x[:n_u].reshape((n*m, 31))
                V = res.x[n_u:n_u+n_v].reshape((m*p, 31))
                W = res.x[n_u+n_v:].reshape((p*n, 31))
                return U, V, W
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 31
    if time.time() - _START_TIME < _TIMEOUT - 20:
        result = _try_rank_31(n, m, p)
        if result is not None:
            U, V, W = result
            if _validate_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 32
    U, V, W = _build_rank_32(n, m, p)
    if _validate_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    T = _get_target_tensor(n, m, p)
    idx = 0
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                if T[i,j,k] != 0:
                    U[i, idx] = 1
                    V[j, idx] = 1
                    W[k, idx] = 1
                    idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
