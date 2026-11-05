import time
import numpy as np

DIMENSIONS = [3, 5, 7]


def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Check that the factor matrices define the matrix multiplication tensor.
    Constraint: for all i in [0,n), a in [0,m), b in [0,m), k in [0,p):
        sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] = (1 if a==b else 0)
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]

    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)

    try:
        result = np.einsum('iac,bkc,kic->iabk', U_reshaped, V_reshaped, W_reshaped)
    except:
        for i in range(n):
            for a in range(m):
                for b in range(m):
                    for k in range(p):
                        s = np.dot(W_reshaped[k, i, :], U_reshaped[i, a, :] * V_reshaped[b, k, :])
                        expected = 1.0 if a == b else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    expected = np.zeros((n, m, m, p))
    for a in range(m):
        for b in range(m):
            if a == b:
                expected[:, a, b, :] = 1.0

    if np.max(np.abs(result - expected)) > tol:
        return False

    return True


def _strassen_2x2x2():
    """
    Returns the Strassen decomposition for 2x2x2 matrix multiplication.
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1

    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1

    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[2][2] = 1
    W[3][2] = -1

    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[1][3] = 1
    W[0][3] = -1

    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1

    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[0][5] = 1

    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[3][6] = 1

    return U, V, W


def _schoolbook_block(n_block, m_block, p_block):
    """
    Returns the schoolbook decomposition for an n_block x m_block x p_block block.
    """
    rank = n_block * m_block * p_block
    U = [[0] * rank for _ in range(n_block * m_block)]
    V = [[0] * rank for _ in range(m_block * p_block)]
    W = [[0] * rank for _ in range(p_block * n_block)]

    c = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i * m_block + j][c] = 1
                V[j * p_block + k][c] = 1
                W[k * n_block + i][c] = 1
                c += 1

    return U, V, W


def _try_block_decomposition(n, m, p):
    """
    Constructs a decomposition using block Strassen for 2x2x2 sub-blocks.
    """
    I_blocks = [(0, 2), (2, 1)]
    J_blocks = [(0, 2), (2, 2), (4, 1)]
    K_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]

    S_U, S_V, S_W = _strassen_2x2x2()

    total_rank = 0
    for i_start, i_size in I_blocks:
        for j_start, j_size in J_blocks:
            for k_start, k_size in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    total_rank += 7
                else:
                    total_rank += i_size * j_size * k_size

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col_offset = 0

    for i_start, i_size in I_blocks:
        for j_start, j_size in J_blocks:
            for k_start, k_size in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    block_U, block_V, block_W = S_U, S_V, S_W
                    block_rank = 7
                else:
                    block_U, block_V, block_W = _schoolbook_block(i_size, j_size, k_size)
                    block_rank = i_size * j_size * k_size

                for local_i in range(i_size):
                    for local_a in range(j_size):
                        global_i = i_start + local_i
                        global_a = j_start + local_a
                        for c in range(block_rank):
                            U[global_i * m + global_a][col_offset + c] = block_U[local_i * j_size + local_a][c]

                for local_b in range(j_size):
                    for local_k in range(k_size):
                        global_b = j_start + local_b
                        global_k = k_start + local_k
                        for c in range(block_rank):
                            V[global_b * p + global_k][col_offset + c] = block_V[local_b * k_size + local_k][c]

                for local_k in range(k_size):
                    for local_i in range(i_size):
                        global_k = k_start + local_k
                        global_i = i_start + local_i
                        for c in range(block_rank):
                            W[global_k * n + global_i][col_offset + c] = block_W[local_k * i_size + local_i][c]

                col_offset += block_rank

    return {"U": U, "V": V, "W": W}


def _schoolbook_decomposition(n, m, p):
    """
    Constructs the schoolbook decomposition for matrix multiplication.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1

    return {"U": U, "V": V, "W": W}


def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Try the block Strassen decomposition
    try:
        result = _try_block_decomposition(n, m, p)

        # Check deadline
        if time.time() > deadline - 5:
            return result

        # Validate the result
        if _validate_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
        else:
            return _schoolbook_decomposition(n, m, p)
    except Exception:
        return _schoolbook_decomposition(n, m, p)
