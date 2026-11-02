"""Matrix multiplication 4x4x4 tensor decomposition using recursive Strassen (rank 49) with rank reduction check."""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
_STRASSEN_2X2 = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

_STRASSEN_C_COMBINATION = [
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication.
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    block_products = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    ]

    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    w_block = [[[0]*4 for _ in range(2)] for _ in range(7)]
    for (cbr, cbc, c_list) in c_block_combination:
        for (pi, wi) in c_list:
            w_block[pi][cbr][cbc] = wi

    v_local = [[[0]*2 for _ in range(2)] for _ in range(7)]
    for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
        for (si, vs) in v_list:
            v_local[si][lr][lc] = vs

    for i in range(7):
        a_combo, b_combo = block_products[i]
        for s in range(7):
            r = i * 7 + s
            a_local, b_local = _STRASSEN_2X2[s]

            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

            for cbr in range(2):
                for cbc in range(2):
                    wi = w_block[i][cbr][cbc]
                    if wi == 0:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            vs = v_local[s][lr][lc]
                            if vs == 0:
                                continue
                            c_row = 2 * cbr + lr
                            c_col = 2 * cbc + lc
                            W[c_col * n + c_row][r] += wi * vs

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy for speed."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    return bool(np.allclose(result, 1.0, atol=1e-10))


def _try_reduce_rank(U, V, W, n, m, p, rank):
    """
    Check if the rank-1 terms are linearly dependent.
    If so, construct a decomposition with rank-1.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Flatten the rank-1 tensors into vectors of length 16*16*16 = 4096
    # Term r is u_r (v_r (w_r))
    # We check if sum c_r T_r = 0 has a non-trivial solution.
    # This is equivalent to checking if the 49 vectors T_r are linearly dependent.
    
    T = np.zeros((rank, n*m*p))
    for r in range(rank):
        u_r = U_arr[:, r].reshape(n, m)
        v_r = V_arr[:, r].reshape(m, p)
        w_r = W_arr[:, r].reshape(p, n)
        
        # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Note: W index is k*n + i. So W_arr[k*n+i, r] = w_r[k,i]
        # The tensor entry is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can compute the vector by iterating or using einsum
        # T_vec[r, (i,j,k)] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # It is faster to compute the full tensor T_r and flatten.
        # But 49 * 4096 is small.
        
        # Let's build the matrix of coefficients directly.
        # We want to solve C @ T_mat = 0 where T_mat is (rank, 4096).
        
        # Efficient construction:
        # T_mat[r, idx] where idx = (i*m+j)*p*k + ... no, just iterate.
        
        pass # We'll do this in a more vectorized way below if needed, but simple loop is fine for 49x4096.
        
    # Vectorized approach:
    # T_r is a 4x4x4 tensor.
    # We can stack them.
    # U_3d shape (4,4,49), V_3d (4,4,49), W_3d (4,4,49)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T_r[i,j,k] = U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # We want to form a matrix M of shape (4096, 49) where M[idx, r] = T_r[idx]
    
    # This can be done by broadcasting, but 4096*49*49 is too big? No, 4096*49 is ~200k.
    # We can compute the matrix by summing over the product.
    
    # Let's just loop over r to build the matrix, it's only 49 iterations.
    T_mat = np.zeros((n*m*p, rank))
    for r in range(rank):
        u_r = U_3d[:, :, r]
        v_r = V_3d[:, :, r]
        w_r = W_3d[:, :, r]
        
        # Compute tensor T_r
        # T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
        # We can use einsum to get the 3D array, then flatten.
        t_r = np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
        T_mat[:, r] = t_r.flatten()
        
    # Now find null space of T_mat
    # If there is a null vector c, then sum c_r T_r = 0.
    # We need c_r != 0 for some r to reduce rank.
    # Actually, if the rank of T_mat is < 49, there is a null space.
    
    try:
        _, s, vh = np.linalg.svd(T_mat, full_matrices=False)
        # Find the smallest singular value
        if s[-1] < 1e-9:
            # There is a null space.
            # The right singular vector corresponding to the smallest singular value is in the null space.
            # vh has shape (49, 49). The last row is the null vector.
            c = vh[-1]
            
            # Check if c has a significant component.
            if np.max(np.abs(c)) > 1e-6:
                # We can eliminate one term.
                # Find the index with the largest coefficient to eliminate (numerical stability).
                r_elim = np.argmax(np.abs(c))
                
                # Construct new U, V, W with rank 48.
                # T = sum_{r!=elim} T_r + T_elim
                # T_elim = - sum_{r!=elim} (c_r / c_elim) T_r
                # So T = sum_{r!=elim} (1 + c_r/c_elim) T_r
                # New term for r (r != elim) is (1 + c_r/c_elim) * T_r
                # T_r = u_r \otimes v_r \otimes w_r
                # New u_r = (1 + c_r/c_elim) * u_r
                # New v_r = v_r
                # New w_r = w_r
                
                new_rank = rank - 1
                new_U = [[0] * new_rank for _ in range(n*m)]
                new_V = [[0] * new_rank for _ in range(m*p)]
                new_W = [[0] * new_rank for _ in range(p*n)]
                
                new_idx = 0
                for r in range(rank):
                    if r == r_elim:
                        continue
                    coeff = 1.0 + c[r] / c[r_elim]
                    for i in range(n*m):
                        new_U[i][new_idx] = U[i][r] * coeff
                    for j in range(m*p):
                        new_V[j][new_idx] = V[j][r]
                    for k in range(p*n):
                        new_W[k][new_idx] = W[k][r]
                    new_idx += 1
                
                # Verify the new decomposition
                if _verify_decomposition(new_U, new_V, new_W, n, m, p, new_rank):
                    return {"U": new_U, "V": new_V, "W": new_W}
                    
    except Exception as e:
        pass
        
    return None


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    t0 = time.time()
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p, rank):
        return _schoolbook_decomposition(n, m, p)

    # Try to reduce rank by checking for linear dependencies
    reduced_result = _try_reduce_rank(U, V, W, n, m, p, rank)
    if reduced_result is not None:
        return reduced_result

    return result

# EVOLVE-BLOCK-END
