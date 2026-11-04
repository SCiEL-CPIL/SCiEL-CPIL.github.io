import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    _t0 = time.time()
    _DEADLINE = 115.0  # Safety margin

    # 1. Fallback: Rank 36 Structural Decomposition
    def get_rank_36_solution():
        rank = 36
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
            for (i, j), val in u_coeffs.items():
                U[i * m + j, idx] += val
            for (j, k), val in v_coeffs.items():
                V[j * p + k, idx] += val
            for (k, i), val in w_coeffs.items():
                W[k * n + i, idx] += val

        def strassen_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            # Strassen 2x2x2
            add_term(idx_start,
                     {(0, a_row_offset): 1, (1, a_row_offset + 1): 1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1},
                     {(c_col_offset, 0): 1, (c_col_offset + 1, 1): 1})
            add_term(idx_start + 1,
                     {(1, a_row_offset): 1, (1, a_row_offset + 1): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1, (c_col_offset + 1, 1): -1})
            add_term(idx_start + 2,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset + 1): 1, (b_row_offset + 1, c_col_offset + 1): -1},
                     {(c_col_offset + 1, 0): 1, (c_col_offset + 1, 1): 1})
            add_term(idx_start + 3,
                     {(1, a_row_offset + 1): 1},
                     {(b_row_offset + 1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},
                     {(c_col_offset, 0): 1, (c_col_offset, 1): 1})
            add_term(idx_start + 4,
                     {(0, a_row_offset): 1, (0, a_row_offset + 1): 1},
                     {(b_row_offset + 1, c_col_offset + 1): 1},
                     {(c_col_offset, 0): -1, (c_col_offset + 1, 0): 1})
            add_term(idx_start + 5,
                     {(1, a_row_offset): 1, (0, a_row_offset): -1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset + 1): 1},
                     {(c_col_offset + 1, 1): 1})
            add_term(idx_start + 6,
                     {(0, a_row_offset + 1): 1, (1, a_row_offset + 1): -1},
                     {(b_row_offset + 1, c_col_offset): 1, (b_row_offset + 1, c_col_offset + 1): 1},
                     {(c_col_offset, 0): 1})

        def schoolbook_2x2x1_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            add_term(idx_start,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start + 1,
                     {(0, a_row_offset + 1): 1},
                     {(b_row_offset + 1, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start + 2,
                     {(1, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1})
            add_term(idx_start + 3,
                     {(1, a_row_offset + 1): 1},
                     {(b_row_offset + 1, c_col_offset): 1},
                     {(c_col_offset, 1): 1})

        idx = 0
        strassen_block(0, 0, 0, idx); idx += 7
        strassen_block(0, 0, 2, idx); idx += 7
        schoolbook_2x2x1_block(0, 0, 4, idx); idx += 4
        strassen_block(2, 2, 0, idx); idx += 7
        strassen_block(2, 2, 2, idx); idx += 7
        schoolbook_2x2x1_block(2, 2, 4, idx); idx += 4
        
        return U, V, W, rank

    # 2. Search for Rank 34 Solution (using Rank 13 for M_2,2,4)
    def try_find_rank_34_solution():
        # Target: M_2,2,4 rank 13
        # T[i,j,k] = 1 for i in 0..1, j in 0..1, k in 0..3
        # Variables: U (2*2*13), V (2*4*13), W (4*2*13) -> 52 + 104 + 104 = 260 vars
        # Equations: 2*2*4 = 16
        
        def residual(params):
            # Unpack
            U_flat = params[:52]
            V_flat = params[52:156]
            W_flat = params[156:260]
            
            U = U_flat.reshape(2, 2, 13) # i, j, r
            V = V_flat.reshape(2, 4, 13) # j, k, r
            W = W_flat.reshape(4, 2, 13) # k, i, r
            
            res = []
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        # Sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        best_res = np.inf
        best_params = None
        
        # Random restarts
        np.random.seed(42)
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > _DEADLINE - 10:
                break
            # Initialize with small random values
            init = np.random.randn(260) * 0.1
            try:
                sol = least_squares(residual, init, method='lm', max_nfev=1000)
                if sol.cost < best_res:
                    best_res = sol.cost
                    best_params = sol.x
                    if best_res < 1e-8:
                        break
            except Exception:
                continue
                
        if best_res < 1e-8:
            # Construct full rank 34 solution
            # Block 1: A cols 0,1 * B cols 0,3 -> C cols 0,3 (M_2,2,4)
            # Block 2: A cols 2,3 * B cols 0,3 -> C cols 0,3 (M_2,2,4)
            # Block 3: A cols 0,3 * B col 4 -> C col 4 (M_2,4,1)
            
            rank = 34
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Extract M_2,2,4 coefficients
            U_13 = best_params[:52].reshape(2, 2, 13)
            V_13 = best_params[52:156].reshape(2, 4, 13)
            W_13 = best_params[156:260].reshape(4, 2, 13)
            
            # Place Block 1 (A rows 0,1; A cols 0,1; B rows 0,1; B cols 0,3; C cols 0,3)
            for i in range(2):
                for j in range(2):
                    for r in range(13):
                        U[i * m + j, r] = U_13[i, j, r]
            for j in range(2):
                for k in range(4):
                    for r in range(13):
                        V[j * p + k, r] = V_13[j, k, r]
            for k in range(4):
                for i in range(2):
                    for r in range(13):
                        W[k * n + i, r] = W_13[k, i, r]
                        
            # Place Block 2 (A rows 0,1; A cols 2,3; B rows 2,3; B cols 0,3; C cols 0,3)
            # Note: The M_2,2,4 decomposition is for generic 2x2 A, 2x4 B.
            # We can reuse the same coefficients for the second A block?
            # Yes, because the structure is identical.
            for i in range(2):
                for j in range(2):
                    for r in range(13):
                        U[i * m + (2 + j), 13 + r] = U_13[i, j, r]
            for j in range(2):
                for k in range(4):
                    for r in range(13):
                        V[(2 + j) * p + k, 13 + r] = V_13[j, k, r]
            for k in range(4):
                for i in range(2):
                    for r in range(13):
                        W[k * n + i, 13 + r] = W_13[k, i, r]
                        
            # Place Block 3 (M_2,4,1) Rank 8
            # Schoolbook for last column
            idx = 26
            for i in range(2):
                for j in range(4):
                    U[i * m + j, idx] = 1
                    V[j * p + 4, idx] = 1
                    W[4 * n + i, idx] = 1
                    idx += 1
                    
            return U, V, W, rank
        else:
            return None

    # Try to find better solution
    better_solution = try_find_rank_34_solution()
    
    if better_solution:
        U, V, W, rank = better_solution
    else:
        U, V, W, rank = get_rank_36_solution()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
