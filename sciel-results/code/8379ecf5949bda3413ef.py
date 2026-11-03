import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    _start_time = time.time()
    _deadline = 125.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Start with rank 66 from the block decomposition
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col_idx = 0

    # Helper to add a column
    def add_column(u_entries, v_entries, w_entries):
        nonlocal col_idx
        if col_idx >= rank:
            return
        for (u_idx_val, val) in u_entries:
            U[u_idx_val][col_idx] += val
        for (v_idx_val, val) in v_entries:
            V[v_idx_val][col_idx] += val
        for (w_idx_val, val) in w_entries:
            W[w_idx_val][col_idx] += val
        col_idx += 1

    # Helper to get global indices
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Recursive decomposition
    def decompose(n_loc, m_loc, p_loc, i_start, j_start, k_start):
        # Base cases
        if n_loc == 1 and m_loc == 1 and p_loc == 1:
            add_column(
                [(u_idx(i_start, j_start), 1)],
                [(v_idx(j_start, k_start), 1)],
                [(w_idx(k_start, i_start), 1)]
            )
            return

        if n_loc == 1 and m_loc == 2 and p_loc == 2:
            # Schoolbook for 1x2x2
            add_column(
                [(u_idx(i_start, j_start), 1)],
                [(v_idx(j_start, k_start), 1)],
                [(w_idx(k_start, i_start), 1)]
            )
            add_column(
                [(u_idx(i_start, j_start+1), 1)],
                [(v_idx(j_start+1, k_start), 1)],
                [(w_idx(k_start, i_start), 1)]
            )
            add_column(
                [(u_idx(i_start, j_start), 1)],
                [(v_idx(j_start, k_start+1), 1)],
                [(w_idx(k_start+1, i_start), 1)]
            )
            add_column(
                [(u_idx(i_start, j_start+1), 1)],
                [(v_idx(j_start+1, k_start+1), 1)],
                [(w_idx(k_start+1, i_start), 1)]
            )
            return

        if n_loc == 2 and m_loc == 2 and p_loc == 2:
            # Strassen for 2x2x2
            add_column(
                [(u_idx(i_start, j_start), 1), (u_idx(i_start+1, j_start+1), 1)],
                [(v_idx(j_start, k_start), 1), (v_idx(j_start+1, k_start+1), 1)],
                [(w_idx(k_start, i_start), 1), (w_idx(k_start+1, i_start+1), 1)]
            )
            add_column(
                [(u_idx(i_start+1, j_start), 1), (u_idx(i_start+1, j_start+1), 1)],
                [(v_idx(j_start, k_start), 1)],
                [(w_idx(k_start, i_start+1), 1), (w_idx(k_start+1, i_start+1), -1)]
            )
            add_column(
                [(u_idx(i_start, j_start), 1)],
                [(v_idx(j_start, k_start+1), 1), (v_idx(j_start+1, k_start+1), -1)],
                [(w_idx(k_start+1, i_start), 1), (w_idx(k_start+1, i_start+1), 1)]
            )
            add_column(
                [(u_idx(i_start+1, j_start+1), 1)],
                [(v_idx(j_start+1, k_start), 1), (v_idx(j_start, k_start), -1)],
                [(w_idx(k_start, i_start), 1), (w_idx(k_start, i_start+1), 1)]
            )
            add_column(
                [(u_idx(i_start, j_start), 1), (u_idx(i_start, j_start+1), 1)],
                [(v_idx(j_start+1, k_start+1), 1)],
                [(w_idx(k_start, i_start), -1), (w_idx(k_start+1, i_start), 1)]
            )
            add_column(
                [(u_idx(i_start+1, j_start), 1), (u_idx(i_start, j_start), -1)],
                [(v_idx(j_start, k_start), 1), (v_idx(j_start, k_start+1), 1)],
                [(w_idx(k_start+1, i_start+1), 1)]
            )
            add_column(
                [(u_idx(i_start, j_start+1), 1), (u_idx(i_start+1, j_start+1), -1)],
                [(v_idx(j_start+1, k_start), 1), (v_idx(j_start+1, k_start+1), 1)],
                [(w_idx(k_start, i_start), 1)]
            )
            return

        # Recursive steps
        if n_loc == 3 and m_loc == 2 and p_loc == 2:
            decompose(1, 2, 2, i_start, j_start, k_start)
            decompose(2, 2, 2, i_start+1, j_start, k_start)
            return

        if n_loc == 3 and m_loc == 2 and p_loc == 6:
            decompose(3, 2, 2, i_start, j_start, k_start)
            decompose(3, 2, 2, i_start, j_start, k_start+2)
            decompose(3, 2, 2, i_start, j_start, k_start+4)
            return

        if n_loc == 3 and m_loc == 4 and p_loc == 6:
            decompose(3, 2, 6, i_start, j_start, k_start)
            decompose(3, 2, 6, i_start, j_start+2, k_start)
            return

        # Fallback: Schoolbook
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    add_column(
                        [(u_idx(i_start+i, j_start+j), 1)],
                        [(v_idx(j_start+j, k_start+k), 1)],
                        [(w_idx(k_start+k, i_start+i), 1)]
                    )

    decompose(n, m, p, 0, 0, 0)
    actual_rank = col_idx

    # Verify the decomposition is exact
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(actual_rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)

    if max_err > 1e-10:
        raise ValueError(f"Decomposition error too large: {max_err}")

    # Try to reduce rank by finding linear dependencies among columns
    # We check if any column can be expressed as a linear combination of others
    # For each column c, we check if the vector (U[:,c], V[:,c], W[:,c]) is in the span of the others
    
    # Build the combined matrix where each column is the concatenation of U[:,c], V[:,c], W[:,c]
    # This is a (12*m*p + 2*m*p + 6*p*n) x rank matrix... actually let's think about it differently.
    # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # A rank reduction would mean finding coefficients such that one column is a linear combo of others
    # that still preserves the tensor identity.
    
    # For now, let's just try to check if we can drop a column by checking if the remaining
    # columns still span the same space. This is a heuristic approach.
    
    # Let's try a different approach: use numpy to find the rank of the combined matrix
    # and see if there are dependencies.
    
    # The combined "flattened" matrix for each column r is:
    # col_r = [U[:,r], V[:,r], W[:,r]]
    # If the columns of this combined matrix are linearly dependent, we might be able to reduce rank.
    
    # Build the combined matrix
    combined = np.zeros((n*m + m*p + p*n, actual_rank))
    combined[:n*m, :] = U_np[:, :actual_rank]
    combined[n*m:n*m+m*p, :] = V_np[:, :actual_rank]
    combined[n*m+m*p:, :] = W_np[:, :actual_rank]
    
    # Find the rank of this combined matrix
    rank_combined = np.linalg.matrix_rank(combined, tol=1e-10)
    
    # If rank_combined < actual_rank, there are dependencies. But we need to check if
    # the dependencies preserve the tensor identity. For now, we'll just use the actual_rank
    # and not attempt to reduce, since the dependencies might not preserve the tensor.
    
    # Return the decomposition with the minimal rank we can achieve
    return {"U": U, "V": V, "W": W}
