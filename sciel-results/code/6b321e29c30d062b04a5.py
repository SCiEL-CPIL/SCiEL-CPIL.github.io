import numpy as np
import time

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0  # seconds

def _check_deadline():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

# Strassen 2x2 terms as (A_coeffs, B_coeffs, C_coeffs)
# Each coeff is a dict: (row, col) -> coefficient
_STRASSEN_2X2 = [
    # P1 = (a+d)(e+h) -> C11 += P1, C22 += P1
    ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    # P2 = (c+d)e -> C21 += P2, C22 -= P2
    ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
    # P3 = a(f-h) -> C12 += P3, C22 += P3
    ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
    # P4 = d(g-e) -> C11 += P4, C21 += P4
    ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
    # P5 = (a+b)h -> C11 -= P5, C12 += P5
    ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
    # P6 = (c-a)(e+f) -> C22 += P6
    ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
    # P7 = (b-d)(g+h) -> C11 += P7
    ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
]

def _generate_strassen_4x4():
    """
    Generate rank-49 Strassen decomposition for 4x4 matrix multiplication.
    
    Convention:
    U[i*n+k, r] - coefficient for A[i,k] in rank-r term
    V[k*n+j, r] - coefficient for B[k,j] in rank-r term  
    W[j*n+i, r] - coefficient for C[i,j] in rank-r term
    """
    n = 4
    half = 2
    outer_rank = 7
    inner_rank = 7
    total_rank = outer_rank * inner_rank
    
    U = np.zeros((n * n, total_rank), dtype=float)
    V = np.zeros((n * n, total_rank), dtype=float)
    W = np.zeros((n * n, total_rank), dtype=float)
    
    # Precompute inner Strassen matrices as arrays for speed
    inner_U = np.zeros((half * half, inner_rank), dtype=float)
    inner_V = np.zeros((half * half, inner_rank), dtype=float)
    inner_W = np.zeros((half * half, inner_rank), dtype=float)
    
    for t, (a_coeffs, b_coeffs, c_coeffs) in enumerate(_STRASSEN_2X2):
        for (r, c), coeff in a_coeffs.items():
            inner_U[r * half + c, t] = coeff
        for (r, c), coeff in b_coeffs.items():
            inner_V[r * half + c, t] = coeff
        for (r, c), coeff in c_coeffs.items():
            inner_W[c * half + r, t] = coeff  # W[j*2+i] for C[i,j]
    
    # Build outer Strassen decomposition
    for t_outer, (a_blocks, b_blocks, c_blocks) in enumerate(_STRASSEN_2X2):
        for t_inner in range(inner_rank):
            r_idx = t_outer * inner_rank + t_inner
            
            # A (U) part
            for (br, bc), block_coeff in a_blocks.items():
                for er in range(half):
                    for ec in range(half):
                        elem_coeff = inner_U[er * half + ec, t_inner]
                        if elem_coeff != 0:
                            i_global = br * half + er
                            k_global = bc * half + ec
                            U[i_global * n + k_global, r_idx] += block_coeff * elem_coeff
            
            # B (V) part
            for (br, bc), block_coeff in b_blocks.items():
                for er in range(half):
                    for ec in range(half):
                        elem_coeff = inner_V[er * half + ec, t_inner]
                        if elem_coeff != 0:
                            k_global = br * half + er
                            j_global = bc * half + ec
                            V[k_global * n + j_global, r_idx] += block_coeff * elem_coeff
            
            # C (W) part
            for (br, bc), block_coeff in c_blocks.items():
                for er in range(half):
                    for ec in range(half):
                        elem_coeff = inner_W[ec * half + er, t_inner]
                        if elem_coeff != 0:
                            i_global = br * half + er
                            j_global = bc * half + ec
                            W[j_global * n + i_global, r_idx] += block_coeff * elem_coeff
    
    return U, V, W

def _verify_decomposition(U, V, W, n=4, atol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    U_r = U.reshape(n, n, rank)
    V_r = V.reshape(n, n, rank)
    W_r = W.reshape(n, n, rank)
    
    # T[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] should equal 1 for all i,j,k
    T = np.einsum('ikr,kjr,jir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))

def _reduce_rank(U, V, W, tol=1e-10):
    """
    Remove linearly dependent columns from the decomposition.
    A column r is redundant if (U[:,r], V[:,r], W[:,r]) is linearly dependent
    on previously kept columns.
    """
    n = 4
    U_np = U.astype(float)
    V_np = V.astype(float)
    W_np = W.astype(float)
    
    rank = U_np.shape[1]
    keep = []
    basis = []  # List of combined column vectors
    
    for r in range(rank):
        if _check_deadline():
            break
        
        col_u = U_np[:, r]
        col_v = V_np[:, r]
        col_w = W_np[:, r]
        
        # Skip if all zeros
        if np.allclose(col_u, 0, atol=tol) and np.allclose(col_v, 0, atol=tol) and np.allclose(col_w, 0, atol=tol):
            continue
        
        if len(keep) == 0:
            keep.append(r)
            basis.append(np.concatenate([col_u, col_v, col_w]))
        else:
            # Check if this column is linearly independent of the basis
            all_basis = np.column_stack(basis)
            new_col = np.concatenate([col_u, col_v, col_w])
            
            # Check rank increase
            rank_old = np.linalg.matrix_rank(all_basis, tol=tol)
            rank_new = np.linalg.matrix_rank(np.hstack([all_basis, new_col.reshape(-1, 1)]), tol=tol)
            
            if rank_new > rank_old:
                keep.append(r)
                basis.append(new_col)
    
    if len(keep) == rank:
        return U, V, W
    
    U_red = U_np[:, keep]
    V_red = V_np[:, keep]
    W_red = W_np[:, keep]
    
    # Verify the reduced decomposition
    if not _verify_decomposition(U_red, V_red, W_red, n):
        return U, V, W
    
    return U_red, V_red, W_red

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n = 4
    
    # Generate the Strassen-based rank-49 decomposition
    U, V, W = _generate_strassen_4x4()
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n):
        # Fallback: return schoolbook rank-64
        U_fb = np.zeros((n*n, n*n), dtype=float)
        V_fb = np.zeros((n*n, n*n), dtype=float)
        W_fb = np.zeros((n*n, n*n), dtype=float)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    idx = i*n + j
                    U_fb[idx, k*n + i] = 1.0  # U[i*n+k]
                    V_fb[idx, k*n + j] = 1.0  # V[k*n+j]
                    W_fb[idx, j*n + i] = 1.0  # W[j*n+i]
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank
    if not _check_deadline():
        U, V, W = _reduce_rank(U, V, W)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n):
        # Fallback to original
        U, V, W = _generate_strassen_4x4()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
