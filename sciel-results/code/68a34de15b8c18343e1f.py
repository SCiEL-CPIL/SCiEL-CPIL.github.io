"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def strassen_decomposition():
        """
        Rank-84 decomposition using Strassen for 2x2x2 blocks.
        Splits:
        n: [0, 2), [2, 3)
        m: [0, 2), [2, 4), [4, 5)
        p: [0, 2), [2, 4), [4, 6)
        """
        if time.time() > deadline:
            return None
            
        U_cols = []
        V_cols = []
        W_cols = []
        
        n_splits = [(0, 2), (2, 3)]
        m_splits = [(0, 2), (2, 4), (4, 5)]
        p_splits = [(0, 2), (2, 4), (4, 6)]
        
        def add_term(U_idx, V_idx, W_idx):
            """
            Add a single tensor term.
            U_idx: list of (i, j) tuples with coefficient 1 (or -1)
            V_idx: list of (j, k) tuples with coefficient 1 (or -1)
            W_idx: list of (k, i) tuples with coefficient 1 (or -1)
            """
            U_col = np.zeros(n * m)
            V_col = np.zeros(m * p)
            W_col = np.zeros(p * n)
            
            for (i, j), val in U_idx:
                U_col[i * m + j] += val
            for (j, k), val in V_idx:
                V_col[j * p + k] += val
            for (k, i), val in W_idx:
                W_col[k * n + i] += val
                
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
            
        def add_strassen_block(i_start, i_end, j_start, j_end, k_start, k_end):
            """
            Add Strassen decomposition for the block defined by ranges.
            Assumes sizes are 2x2x2.
            """
            i0, i1 = i_start, i_start + 1
            j0, j1 = j_start, j_start + 1
            k0, k1 = k_start, k_start + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            add_term(
                [((i0, j0), 1), ((i1, j1), 1)],
                [((j0, k0), 1), ((j1, k1), 1)],
                [((k0, i0), 1), ((k1, i1), 1)]
            )
            
            # M2 = (A10 + A11)B00
            add_term(
                [((i1, j0), 1), ((i1, j1), 1)],
                [((j0, k0), 1)],
                [((k0, i1), 1)]
            )
            
            # M3 = A00(B10 - B11)
            add_term(
                [((i0, j0), 1)],
                [((j1, k0), 1), ((j1, k1), -1)],
                [((k0, i0), 1)]
            )
            
            # M4 = A11(B01 - B00)
            add_term(
                [((i1, j1), 1)],
                [((j0, k1), 1), ((j0, k0), -1)],
                [((k0, i1), 1), ((k1, i1), 1)]
            )
            
            # M5 = (A00 + A01)B11
            add_term(
                [((i0, j0), 1), ((i0, j1), 1)],
                [((j1, k1), 1)],
                [((k1, i0), -1), ((k1, i1), 1)]
            )
            
            # M6 = (A10 - A00)(B00 + B10)
            add_term(
                [((i1, j0), 1), ((i0, j0), -1)],
                [((j0, k0), 1), ((j1, k0), 1)],
                [((k1, i1), 1)]
            )
            
            # M7 = (A01 - A11)(B01 + B11)
            add_term(
                [((i0, j1), 1), ((i1, j1), -1)],
                [((j0, k1), 1), ((j1, k1), 1)],
                [((k0, i0), 1), ((k1, i0), 1)]
            )

        def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
            """
            Add schoolbook decomposition for the block.
            """
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        add_term(
                            [((i, j), 1)],
                            [((j, k), 1)],
                            [((k, i), 1)]
                        )

        for i_start, i_end in n_splits:
            for j_start, j_end in m_splits:
                for k_start, k_end in p_splits:
                    i_size = i_end - i_start
                    j_size = j_end - j_start
                    k_size = k_end - k_start
                    
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        add_strassen_block(i_start, i_end, j_start, j_end, k_start, k_end)
                    else:
                        add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end)
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        rank = U.shape[1]
        
        if verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist(), 'rank': rank}
        
        return None

    result = strassen_decomposition()
    if result is not None:
        rank = result.pop('rank')
        if rank < best_rank:
            best_rank = rank
            best_result = result
            
    return best_result
# EVOLVE-BLOCK-END
