import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    initial_rank = 189
    
    U = np.zeros((n * m, initial_rank))
    V = np.zeros((m * p, initial_rank))
    W = np.zeros((p * n, initial_rank))
    
    strassen_terms = [
        {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
        {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(1, 0, 1), (1, 1, -1)]},
        {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(0, 1, 1), (1, 1, 1)]},
        {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (1, 0, 1)]},
        {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (0, 1, 1)]},
        {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
        {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]}
    ]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    term = strassen_terms[t]
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * i + lr) * m + (2 * k + lc)
                        U[u_row, col_idx] += coeff
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * k + lr) * p + (2 * j + lc)
                        V[v_row, col_idx] += coeff
                    for lr, lc, coeff in term['W']:
                        w_row = (2 * j + lc) * n + (2 * i + lr)
                        W[w_row, col_idx] += coeff

    # Check rank
    U_svd = np.linalg.svd(U, full_matrices=False)
    rank_U = np.sum(U_svd > 1e-10)
    
    if rank_U < initial_rank:
        # Try to find a basis of size rank_U
        # Greedy selection of columns
        selected_cols = []
        current_basis = []
        
        for col in range(initial_rank):
            if len(selected_cols) == rank_U:
                break
            # Check if column is independent
            # Use a simple check: is the column not in the span of current_basis?
            # For speed, we just pick the first rank_U columns that are non-zero and hope they are independent?
            # No, that's unsafe.
            
            # Let's just return the full 189 to be safe.
            pass

    U_list = [[int(val) if val == int(val) else float(val) for val in row] for row in U]
    V_list = [[int(val) if val == int(val) else float(val) for val in row] for row in V]
    W_list = [[int(val) if val == int(val) else float(val) for val in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
