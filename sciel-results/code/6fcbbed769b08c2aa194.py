"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    deadline = 135  # safety margin before 150s kill
    
    # First, build the rank-28 Strassen decomposition as baseline
    rank28_U, rank28_V, rank28_W = _build_rank28()
    
    # Try to find rank-27 by checking if we can eliminate one multiplication
    # through linear dependency analysis
    best_result = {"U": rank28_U, "V": rank28_V, "W": rank28_W}
    
    # Verify rank-28 is correct
    if time.time() - start_time < deadline:
        if _verify_decomposition(rank28_U, rank28_V, rank28_W, n, m, p):
            # Try rank 27: check if any column in the tensor decomposition is redundant
            if time.time() - start_time < deadline:
                rank27_result = _try_rank27(rank28_U, rank28_V, rank28_W, n, m, p)
                if rank27_result is not None:
                    best_result = rank27_result
    
    return best_result

def _build_rank28():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1
    
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_k U[:,k] outer V[:,k] outer W[:,k] = matmul tensor"""
    import numpy as np
    rank = len(U[0])
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build expected tensor
    expected = np.zeros((nm, mp, pn))
    for i in range(nm):
        a_row, a_col = divmod(i, m)
        for j in range(mp):
            b_row, b_col = divmod(j, p)
            for k in range(pn):
                c_col, c_row = divmod(k, n)
                if a_col == b_row and a_row == c_row and b_col == c_col:
                    expected[i, j, k] = 1.0
    
    # Compute actual tensor
    U_arr = np.array(U, dtype=float)  # (nm, rank)
    V_arr = np.array(V, dtype=float)  # (mp, rank)
    W_arr = np.array(W, dtype=float)  # (pn, rank)
    
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Use einsum
    actual = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    if not np.allclose(actual, expected, atol=1e-9):
        return False
    return True

def _try_rank27(U28, V28, W28, n, m, p):
    """Try to find a rank-27 decomposition by checking for linear dependencies"""
    import numpy as np
    rank = len(U28[0])
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the 28 tensor vectors as flattened vectors of length nm*mp*pn
    # Tensor vector for term k: T_k[i,j,l] = U[i,k]*V[j,k]*W[l,k]
    vectors = []
    for k in range(rank):
        vec = np.zeros(nm * mp * pn)
        for i in range(nm):
            for j in range(mp):
                for l in range(pn):
                    vec[i * mp * pn + j * pn + l] = U28[i][k] * V28[j][k] * W28[l][k]
        vectors.append(vec)
    
    vectors = np.array(vectors)  # shape (rank, nm*mp*pn)
    
    # Check if any vector is linearly dependent on the others
    # For each k, check if vectors[k] is in the span of vectors without k
    for k in range(rank):
        others = np.delete(vectors, k, axis=0)
        # Check if vectors[k] is in the column space of others
        # Use SVD or rank check
        try:
            # Compute the rank of others and of [others, vectors[k]]
            rank_others = np.linalg.matrix_rank(others.T)  # others is (rank-1, N), so others.T is (N, rank-1)
            combined = np.vstack([others.T, vectors[k].reshape(1, -1)])
            rank_combined = np.linalg.matrix_rank(combined)
            if rank_combined == rank_others:
                # vectors[k] is linearly dependent!
                # Remove term k
                U_new = []
                V_new = []
                W_new = []
                for i in range(nm):
                    U_new.append([U28[i][j] for j in range(rank) if j != k])
                for j in range(mp):
                    V_new.append([V28[j][j2] for j2 in range(rank) if j2 != k])
                for l in range(pn):
                    W_new.append([W28[l][j] for j in range(rank) if j != k])
                
                # Verify the new decomposition
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return {"U": U_new, "V": V_new, "W": W_new}
                
                # Verify the new decomposition
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return {"U": U_new, "V": V_new, "W": W_new}
        except Exception:
            continue
    
    return None
# EVOLVE-BLOCK-END
