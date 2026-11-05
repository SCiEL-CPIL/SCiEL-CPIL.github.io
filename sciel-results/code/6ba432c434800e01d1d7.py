import numpy as np
import time

def construct():
    """
    Construct an exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The decomposition satisfies:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    where T[i*m+j, j*p+k, k*n+i] = 1 and 0 otherwise.
    
    Strategy:
    Split the dimensions into blocks to utilize Strassen's algorithm for 2x2x2 blocks.
    n=4 -> [2, 2]
    m=5 -> [2, 2, 1]
    p=6 -> [2, 2, 2]
    
    Blocks of type 2x2x2 use Strassen (rank 7).
    Blocks of type 2x1x2 use Schoolbook (rank 4).
    
    Total rank = 12 * 7 + 6 * 4 = 84 + 24 = 108.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = 4, 5, 6
    
    # Block offsets and sizes
    i_offsets = [0, 2]
    i_sizes = [2, 2]
    
    j_offsets = [0, 2, 4]
    j_sizes = [2, 2, 1]
    
    k_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    # Pre-allocate max possible rank (schoolbook is 120, we expect 108)
    max_rank = 120
    U = np.zeros((n * m, max_rank), dtype=int)
    V = np.zeros((m * p, max_rank), dtype=int)
    W = np.zeros((p * n, max_rank), dtype=int)
    
    col = 0
    
    for i_idx, i_0 in enumerate(i_offsets):
        n_b = i_sizes[i_idx]
        if time.time() > deadline:
            break
        for j_idx, j_0 in enumerate(j_offsets):
            m_b = j_sizes[j_idx]
            if time.time() > deadline:
                break
            for k_idx, k_0 in enumerate(k_offsets):
                p_b = k_sizes[k_idx]
                
                # Helper to add a term
                def add_term(u_indices, v_indices, w_indices, coeffs):
                    nonlocal col
                    # coeffs is a list of (u_idx, v_idx, w_idx, coeff)
                    # But we need to add to the column 'col'
                    for u_idx, v_idx, w_idx, coeff in coeffs:
                        U[u_idx, col] += coeff
                        V[v_idx, col] += coeff
                        W[w_idx, col] += coeff
                    col += 1

                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Strassen's algorithm for 2x2x2
                    # Local indices 0,1 for i, j, k
                    # Global indices:
                    # i: i_0, i_0+1
                    # j: j_0, j_0+1
                    # k: k_0, k_0+1
                    
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    
                    u00 = (i_0+0)*m + (j_0+0)
                    u01 = (i_0+0)*m + (j_0+1)
                    u10 = (i_0+1)*m + (j_0+0)
                    u11 = (i_0+1)*m + (j_0+1)
                    
                    v00 = (j_0+0)*p + (k_0+0)
                    v01 = (j_0+0)*p + (k_0+1)
                    v10 = (j_0+1)*p + (k_0+0)
                    v11 = (j_0+1)*p + (k_0+1)
                    
                    w00 = (k_0+0)*n + (i_0+0)
                    w01 = (k_0+0)*n + (i_0+1)
                    w10 = (k_0+1)*n + (i_0+0)
                    w11 = (k_0+1)*n + (i_0+1)
                    
                    # M1 = (A00+A11)(B00+B11)
                    add_term([
                        (u00, v00, w00, 1),
                        (u00, v00, w01, 0), # Wait, W indices are for C.
                        # Let's re-verify the mapping.
                        # T[a,b,c] = sum U[a,r] V[b,r] W[c,r]
                        # a = i*m+j, b = j*p+k, c = k*n+i
                        # Term is rank 1 tensor: u_vec * v_vec * w_vec
                        # u_vec has 1s at positions corresponding to A entries.
                        # v_vec has 1s at positions corresponding to B entries.
                        # w_vec has 1s at positions corresponding to C entries.
                        
                        # For Strassen M1:
                        # A_part: A00, A11
                        # B_part: B00, B11
                        # C_part: C00, C11 (with +1, +1)
                        # So:
                        # U: 1 at u00, 1 at u11
                        # V: 1 at v00, 1 at v11
                        # W: 1 at w00, 1 at w11
                    ], [], []) # The helper above is slightly misaligned with standard add_term logic if I pass lists of indices. 
                    # Let's just write it out explicitly for clarity and safety, as done in inspiration 1.
                    
                    # Term 1: M1 = (A00+A11)(B00+B11) -> C00, C11
                    U[u00, col] = 1
                    U[u11, col] = 1
                    V[v00, col] = 1
                    V[v11, col] = 1
                    W[w00, col] = 1
                    W[w11, col] = 1
                    col += 1
                    
                    # Term 2: M2 = (A10+A11)B00 -> C10, C11(-)
                    U[u10, col] = 1
                    U[u11, col] = 1
                    V[v00, col] = 1
                    W[w10, col] = 1
                    W[w11, col] = -1
                    col += 1
                    
                    # Term 3: M3 = A00(B01-B11) -> C01, C11
                    U[u00, col] = 1
                    V[v01, col] = 1
                    V[v11, col] = -1
                    W[w10, col] = 1
                    W[w11, col] = 1
                    col += 1
                    
                    # Term 4: M4 = A11(B10-B00) -> C00, C10
                    U[u11, col] = 1
                    V[v10, col] = 1
                    V[v00, col] = -1
                    W[w00, col] = 1
                    W[w10, col] = 1
                    col += 1
                    
                    # Term 5: M5 = (A00+A01)B11 -> C01, C00(-)
                    U[u00, col] = 1
                    U[u01, col] = 1
                    V[v11, col] = 1
                    W[w10, col] = 1
                    W[w00, col] = -1
                    col += 1
                    
                    # Term 6: M6 = (A10-A00)(B00+B01) -> C11
                    U[u10, col] = 1
                    U[u00, col] = -1
                    V[v00, col] = 1
                    V[v01, col] = 1
                    W[w11, col] = 1
                    col += 1
                    
                    # Term 7: M7 = (A01-A11)(B10+B11) -> C00
                    U[u01, col] = 1
                    U[u11, col] = -1
                    V[v10, col] = 1
                    V[v11, col] = 1
                    W[w00, col] = 1
                    col += 1
                    
                elif n_b == 2 and m_b == 1 and p_b == 2:
                    # Schoolbook for 2x1x2
                    # A is 2x1, B is 1x2, C is 2x2
                    # C_{u,w} = A_{u,0} * B_{0,w}
                    
                    u0 = (i_0+0)*m + (j_0+0)
                    u1 = (i_0+1)*m + (j_0+0)
                    
                    v0 = (j_0+0)*p + (k_0+0)
                    v1 = (j_0+0)*p + (k_0+1)
                    
                    w00 = (k_0+0)*n + (i_0+0)
                    w10 = (k_0+0)*n + (i_0+1)
                    w01 = (k_0+1)*n + (i_0+0)
                    w11 = (k_0+1)*n + (i_0+1)
                    
                    # Term 1: A00 * B00 -> C00
                    U[u0, col] = 1
                    V[v0, col] = 1
                    W[w00, col] = 1
                    col += 1
                    
                    # Term 2: A00 * B01 -> C01
                    U[u0, col] = 1
                    V[v1, col] = 1
                    W[w01, col] = 1
                    col += 1
                    
                    # Term 3: A10 * B00 -> C10
                    U[u1, col] = 1
                    V[v0, col] = 1
                    W[w10, col] = 1
                    col += 1
                    
                    # Term 4: A10 * B01 -> C11
                    U[u1, col] = 1
                    V[v1, col] = 1
                    W[w11, col] = 1
                    col += 1
                else:
                    # Fallback: Schoolbook for any other size
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0+u)*m + (j_0+v), col] = 1
                                V[(j_0+v)*p + (k_0+w), col] = 1
                                W[(k_0+w)*n + (i_0+u), col] = 1
                                col += 1

    # Trim arrays to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
