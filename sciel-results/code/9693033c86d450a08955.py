"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strassen 2x2x2 factors (4 positions: (0,0),(0,1),(1,0),(1,1))
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],  # pos 0: (i=0,j=0)
        [0, 0, 0, 0, 1, 0, 1],   # pos 1: (i=0,j=1)
        [0, 1, 0, 0, 0, 1, 0],   # pos 2: (i=1,j=0)
        [1, 1, 0, 1, 0, 0, -1]   # pos 3: (i=1,j=1)
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],  # pos 0: (j=0,k=0)
        [0, 0, 1, 0, 0, 1, 0],   # pos 1: (j=0,k=1)
        [0, 0, 0, 1, 0, 0, 1],   # pos 2: (j=1,k=0)
        [1, 0, -1, 0, 1, 0, 1]   # pos 3: (j=1,k=1)
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],  # pos 0: (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],   # pos 1: (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],   # pos 2: (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0]   # pos 3: (k=1,i=1)
    ]

    # Partition: n=[2,1], m=[2,2,1], p=[2,2,1]
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Compute offsets
    n_offsets = [0]
    for sz in n_parts[:-1]:
        n_offsets.append(n_offsets[-1] + sz)
    m_offsets = [0]
    for sz in m_parts[:-1]:
        m_offsets.append(m_offsets[-1] + sz)
    p_offsets = [0]
    for sz in p_parts[:-1]:
        p_offsets.append(p_offsets[-1] + sz)
    
    u_entries = {}
    v_entries = {}
    w_entries = {}
    col_offset = 0
    
    for ni, nsz in enumerate(n_parts):
        for mi, msz in enumerate(m_parts):
            for pi, psz in enumerate(p_parts):
                i0 = n_offsets[ni]
                j0 = m_offsets[mi]
                k0 = p_offsets[pi]
                
                if nsz == 2 and msz == 2 and psz == 2:
                    # Use Strassen for 2x2x2 block
                    # 4 positions: (di,dj,dk) where di,dj,dk in {0,1}
                    # Position index = di*2 + dj for U, dj*2+dk for V, dk*2+di for W
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                u_pos = di * 2 + dj
                                v_pos = dj * 2 + dk
                                w_pos = dk * 2 + di
                                
                                u_row = (i0 + di) * m + (j0 + dj)
                                v_row = (j0 + dj) * p + (k0 + dk)
                                w_row = (k0 + dk) * n + (i0 + di)
                                
                                for c in range(7):
                                    u_entries[(u_row, col_offset + c)] = u_entries.get((u_row, col_offset + c), 0) + U_strassen[u_pos][c]
                                    v_entries[(v_row, col_offset + c)] = v_entries.get((v_row, col_offset + c), 0) + V_strassen[v_pos][c]
                                    w_entries[(w_row, col_offset + c)] = w_entries.get((w_row, col_offset + c), 0) + W_strassen[w_pos][c]
                    col_offset += 7
                else:
                    # Schoolbook for this block
                    for di in range(nsz):
                        for dj in range(msz):
                            for dk in range(psz):
                                u_row = (i0 + di) * m + (j0 + dj)
                                v_row = (j0 + dj) * p + (k0 + dk)
                                w_row = (k0 + dk) * n + (i0 + di)
                                
                                u_entries[(u_row, col_offset)] = u_entries.get((u_row, col_offset), 0) + 1
                                v_entries[(v_row, col_offset)] = v_entries.get((v_row, col_offset), 0) + 1
                                w_entries[(w_row, col_offset)] = w_entries.get((w_row, col_offset), 0) + 1
                                col_offset += 1
    
    rank_total = col_offset
    
    # Assemble matrices
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    for (row, col), val in u_entries.items():
        U[row][col] = val
    for (row, col), val in v_entries.items():
        V[row][col] = val
    for (row, col), val in w_entries.items():
        W[row][col] = val

    # Verify correctness
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        U = np.array(U_mat)
        V = np.array(V_mat)
        W = np.array(W_mat)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
