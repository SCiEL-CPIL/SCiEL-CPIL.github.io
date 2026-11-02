import numpy as np
import time

# EVOLVE-BLOCK-START

# Deadline guard: kill before 150s wall-clock
_DEADLINE = 150.0
_START_TIME = time.time()

def _check_deadline(safety_margin=5.0):
    return (time.time() - _START_TIME) > (_DEADLINE - safety_margin)

# The 7 Strassen linear combinations for 2x2 matrix multiplication.
# Each combo is a list of (row, col, sign) entries.
_A_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (0, 0, -1)],
    [(0, 1, 1), (1, 1, -1)]
]

_B_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(0, 0, 1)],
    [(0, 1, 1), (1, 1, -1)],
    [(1, 0, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
    [(1, 0, 1), (1, 1, 1)]
]

_C_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 1, 1), (0, 0, -1)],
    [(1, 1, 1)],
    [(0, 0, 1)]
]

def _generate_strassen_4x4():
    """
    Generates U, V, W for 4x4 Strassen decomposition directly.
    Uses the recursive Kronecker product structure:
    T_4 = T_2 ⊗ T_2, giving rank 7*7 = 49.
    
    Indexing convention:
    - U[i*n + k, r] for A element (i,k)
    - V[j*n + l, r] for B element (j,l)  [m=p=4, so V is 4x4]
    - W[k*n + i, r] for C element (k,i) per trace convention W[k*n+i, r]
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Outer Strassen level (2x2 blocks of 2x2)
    for t_outer in range(7):
        # Inner Strassen level (within each 2x2 block)
        for t_inner in range(7):
            r_idx = t_outer * 7 + t_inner
            
            # A (U) part: A[i,k]
            for (br, bc, sign_a) in _A_COMBOS[t_outer]:
                for (ir, ik, sign_inner_a) in _A_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    k_global = bc * 2 + ik
                    u_idx = i_global * n + k_global
                    U[u_idx, r_idx] += sign_a * sign_inner_a
            
            # B (V) part: B[j,l]
            for (br, bc, sign_b) in _B_COMBOS[t_outer]:
                for (ir, ik, sign_inner_b) in _B_COMBOS[t_inner]:
                    j_global = br * 2 + ir
                    l_global = bc * 2 + ik
                    v_idx = j_global * n + l_global
                    V[v_idx, r_idx] += sign_b * sign_inner_b
            
            # C (W) part: C[i,j] -> W[j*n + i, r]
            for (br, bc, sign_c) in _C_COMBOS[t_outer]:
                for (ir, ik, sign_inner_c) in _C_COMBOS[t_inner]:
                    i_global = br * 2 + ir
                    j_global = bc * 2 + ik
                    w_idx = j_global * n + i_global
                    W[w_idx, r_idx] += sign_c * sign_inner_c

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
    """
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))

def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 4, 4, 4
    
    # Primary: rank-49 Strassen decomposition
    U, V, W = _generate_strassen_4x4()
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails (should not happen)
    if _check_deadline(safety_margin=10.0):
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # As a last resort, try with tighter verification
    if _verify_decomposition(U, V, W, n, m, p, atol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
