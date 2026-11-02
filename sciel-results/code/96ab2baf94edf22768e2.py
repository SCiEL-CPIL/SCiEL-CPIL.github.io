import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size  # 3
    
    # Total rank is n_blocks^3 * 7
    rank = n_blocks ** 3 * 7
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0

    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                # M1
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M2
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] -= 1
                col += 1

                # M3
                U[u_idx(ib, kb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M4
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                col += 1

                # M5
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                col += 1

                # M6
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 0, 0), col] -= 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M7
                U[u_idx(ib, kb, 0, 1), col] += 1
                U[u_idx(ib, kb, 1, 1), col] -= 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                col += 1

    # Linear dependency check
    # We check if any column k can be expressed as a linear combination of previous columns.
    # This is expensive, so we do it only if time permits.
    if time.time() < deadline - 10:
        keep = [True] * rank
        for k in range(rank):
            if not keep[k]:
                continue
            # Check if U[:,k], V[:,k], W[:,k] tensor is in span of previous kept tensors
            # This is hard. We can check if the W vector is in the span of previous W vectors
            # AND the corresponding U/V allow it.
            # Actually, if W_k is in span of W_j, it doesn't mean the tensor is in span.
            # We skip this complex check to avoid bugs and timeouts.
            # The Block-Strassen rank 189 is likely minimal for this family.
            pass

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure integer/half-integer
    # The entries are already integers.
    
    return {"U": U_list, "V": V_list, "W": W_list}
