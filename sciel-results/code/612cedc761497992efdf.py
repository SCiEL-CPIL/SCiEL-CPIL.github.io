"""Matrix multiplication 2x4 x 4x5 rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

_start_time = time.time()
TIME_LIMIT = 140  # seconds, safety margin before 150s kill

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i
    
    # First, build the rank-36 baseline using Strassen for 2x2x2 blocks
    rank_base = 36
    U_base = [[0] * rank_base for _ in range(n * m)]
    V_base = [[0] * rank_base for _ in range(m * p)]
    W_base = [[0] * rank_base for _ in range(p * n)]
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00+y11)(z00+z11) -> X00, X11
        U_base[u_idx(y00[0], y00[1])][col] = 1
        U_base[u_idx(y11[0], y11[1])][col] = 1
        V_base[v_idx(z00[0], z00[1])][col] = 1
        V_base[v_idx(z11[0], z11[1])][col] = 1
        W_base[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W_base[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10+y11)z00 -> X10, -X11
        U_base[u_idx(y10[0], y10[1])][col] = 1
        U_base[u_idx(y11[0], y11[1])][col] = 1
        V_base[v_idx(z00[0], z00[1])][col] = 1
        W_base[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W_base[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00(z01-z11) -> X01, X11
        U_base[u_idx(y00[0], y00[1])][col] = 1
        V_base[v_idx(z01[0], z01[1])][col] = 1
        V_base[v_idx(z11[0], z11[1])][col] = -1
        W_base[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W_base[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11(z10-z00) -> X00, X10
        U_base[u_idx(y11[0], y11[1])][col] = 1
        V_base[v_idx(z10[0], z10[1])][col] = 1
        V_base[v_idx(z00[0], z00[1])][col] = -1
        W_base[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W_base[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00+y01)z11 -> -X00, X01
        U_base[u_idx(y00[0], y00[1])][col] = 1
        U_base[u_idx(y01[0], y01[1])][col] = 1
        V_base[v_idx(z11[0], z11[1])][col] = 1
        W_base[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W_base[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10-y00)(z00+z01) -> X11
        U_base[u_idx(y10[0], y10[1])][col] = 1
        U_base[u_idx(y00[0], y00[1])][col] = -1
        V_base[v_idx(z00[0], z00[1])][col] = 1
        V_base[v_idx(z01[0], z01[1])][col] = 1
        W_base[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01-y11)(z10+z11) -> X00
        U_base[u_idx(y01[0], y01[1])][col] = 1
        U_base[u_idx(y11[0], y11[1])][col] = -1
        V_base[v_idx(z10[0], z10[1])][col] = 1
        V_base[v_idx(z11[0], z11[1])][col] = 1
        W_base[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L (cols 0,1) * B_11 (rows 0,1; cols 0,1)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R (cols 2,3) * B_21 (rows 2,3; cols 0,1)
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L (cols 0,1) * B_12 (rows 0,1; cols 2,3)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (cols 2,3) * B_22 (rows 2,3; cols 2,3)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (cols 0,1) * B_15 (rows 0,1; col 4) - schoolbook 2x2x1
    U_base[u_idx(0,0)][col] = 1
    V_base[v_idx(0,4)][col] = 1
    W_base[w_idx(4,0)][col] = 1
    col += 1
    U_base[u_idx(0,1)][col] = 1
    V_base[v_idx(1,4)][col] = 1
    W_base[w_idx(4,0)][col] = 1
    col += 1
    U_base[u_idx(1,0)][col] = 1
    V_base[v_idx(0,4)][col] = 1
    W_base[w_idx(4,1)][col] = 1
    col += 1
    U_base[u_idx(1,1)][col] = 1
    V_base[v_idx(1,4)][col] = 1
    W_base[w_idx(4,1)][col] = 1
    col += 1
    
    # Block 6: A_R (cols 2,3) * B_25 (rows 2,3; col 4) - schoolbook 2x2x1
    U_base[u_idx(0,2)][col] = 1
    V_base[v_idx(2,4)][col] = 1
    W_base[w_idx(4,0)][col] = 1
    col += 1
    U_base[u_idx(0,3)][col] = 1
    V_base[v_idx(3,4)][col] = 1
    W_base[w_idx(4,0)][col] = 1
    col += 1
    U_base[u_idx(1,2)][col] = 1
    V_base[v_idx(2,4)][col] = 1
    W_base[w_idx(4,1)][col] = 1
    col += 1
    U_base[u_idx(1,3)][col] = 1
    V_base[v_idx(3,4)][col] = 1
    W_base[w_idx(4,1)][col] = 1
    col += 1
    
    # Now search for linear dependencies to reduce rank
    # Convert to numpy arrays
    U_arr = np.array(U_base, dtype=float)
    V_arr = np.array(V_base, dtype=float)
    W_arr = np.array(W_base, dtype=float)
    
    # The tensor T has entries T[i,j,k] = sum_l U[i,l] * V[j,l] * W[k,l]
    # We want to find if some columns can be eliminated by linear dependencies
    
    # Check for dependencies in the U, V, W spaces
    # If column l can be expressed as linear combination of other columns in all three matrices,
    # we can remove it
    
    best_U, best_V, best_W = U_arr, V_arr, W_arr
    best_rank = rank_base
    
    # Try to find and remove dependent columns
    for target_rank in range(rank_base - 1, 25, -1):  # Try down to rank 26
        if time.time() - _start_time > TIME_LIMIT - 5:
            break
            
        # Check if we can achieve this rank
        # This is a complex optimization, so use a simple heuristic:
        # Try removing one column at a time if it's linearly dependent
        
        U_current = best_U.copy()
        V_current = best_V.copy()
        W_current = best_W.copy()
        current_rank = best_rank
        
        for col_to_remove in range(current_rank - 1, -1, -1):
            if time.time() - _start_time > TIME_LIMIT - 5:
                break
            if current_rank <= target_rank:
                break
            
            # Check if column col_to_remove can be removed
            # by seeing if it's in the span of other columns for U, V, W
            
            # For U: check if U[:, col_to_remove] is in span of U[:, other_cols]
            other_cols = [c for c in range(current_rank) if c != col_to_remove]
            
            try:
                # Check linear dependency
                U_other = U_current[:, other_cols]
                u_dep = U_current[:, col_to_remove]
                
                # Least squares to check dependency
                if U_other.shape[1] > 0:
                    U_sol, _, U_rank, _ = np.linalg.lstsq(U_other, u_dep, rcond=None)
                    U_resid = np.linalg.norm(u_dep - U_other @ U_sol)
                    U_scale = np.linalg.norm(u_dep) + 1e-10
                    U_dep_score = U_resid / U_scale
                else:
                    U_dep_score = 1.0 if np.linalg.norm(u_dep) > 1e-10 else 0.0
                
                # Same for V
                V_other = V_current[:, other_cols]
                v_dep = V_current[:, col_to_remove]
                if V_other.shape[1] > 0:
                    V_sol, _, V_rank, _ = np.linalg.lstsq(V_other, v_dep, rcond=None)
                    V_resid = np.linalg.norm(v_dep - V_other @ V_sol)
                    V_scale = np.linalg.norm(v_dep) + 1e-10
                    V_dep_score = V_resid / V_scale
                else:
                    V_dep_score = 1.0 if np.linalg.norm(v_dep) > 1e-10 else 0.0
                
                # Same for W
                W_other = W_current[:, other_cols]
                w_dep = W_current[:, col_to_remove]
                if W_other.shape[1] > 0:
                    W_sol, _, W_rank, _ = np.linalg.lstsq(W_other, w_dep, rcond=None)
                    W_resid = np.linalg.norm(w_dep - W_other @ W_sol)
                    W_scale = np.linalg.norm(w_dep) + 1e-10
                    W_dep_score = W_resid / W_scale
                else:
                    W_dep_score = 1.0 if np.linalg.norm(w_dep) > 1e-10 else 0.0
                
                # If all three are approximately dependent, remove this column
                if U_dep_score < 0.1 and V_dep_score < 0.1 and W_dep_score < 0.1:
                    U_current = np.delete(U_current, col_to_remove, axis=1)
                    V_current = np.delete(V_current, col_to_remove, axis=1)
                    W_current = np.delete(W_current, col_to_remove, axis=1)
                    current_rank -= 1
                    best_U = U_current.copy()
                    best_V = V_current.copy()
                    best_W = W_current.copy()
                    best_rank = current_rank
            except:
                pass
        
        if best_rank == target_rank:
            break
    
    # Convert back to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Round to nearest integer/half-integer
    def round_vals(lst):
        return [[round(v * 2) / 2 for v in row] for row in lst]
    
    U_list = round_vals(U_list)
    V_list = round_vals(V_list)
    W_list = round_vals(W_list)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
