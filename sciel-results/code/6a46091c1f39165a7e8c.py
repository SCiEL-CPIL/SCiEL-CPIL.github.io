import numpy as np
import time
import random
from fractions import Fraction
from itertools import combinations

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    # x in 0..7, y in 0..15, z in 0..7
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    # Current known solution
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    # Replicate the logic from the prompt to generate U_known, V_known, W_known
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('r,s,t->rst', U_np[:, r], V_np[:, r], W_np[:, r])
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    def try_reduce_rank(U_list, V_list, W_list, current_rank, target_rank, deadline):
        """Try to reduce rank from current_rank to target_rank by finding linear dependencies."""
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        
        # For each pair of columns, check if one can be expressed as a linear combination
        # of others, allowing us to eliminate it
        # More practically: try removing each column and see if the remaining can still
        # span the same tensor space
        
        # Build the tensor contribution matrix for each rank term
        # T_r = u_r ⊗ v_r ⊗ w_r, flattened to a vector of length 8*16*8
        # Check if any term is a linear combination of others
        
        # Flatten each rank-1 tensor
        rank1_tensors = []
        for r in range(current_rank):
            t = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r]).flatten()
            rank1_tensors.append(t)
        
        rank1_tensors = np.array(rank1_tensors)  # shape (current_rank, 8*16*8)
        
        # Try removing one column at a time
        # For each column c, check if rank1_tensors[c] is in the span of the others
        for c in range(current_rank):
            if time.time() > deadline:
                break
            
            # Get all other tensors
            others = np.delete(rank1_tensors, c, axis=0)
            
            # Check if rank1_tensors[c] is in the column space of others
            # Use SVD to check
            try:
                # Compute the projection
                # If rank1_tensors[c] is orthogonal to the null space of others, it's in the span
                # Simpler: compute least squares and check residual
                # Solve: others @ x ≈ rank1_tensors[c]
                # Use lstsq
                x, residuals, rank_others, sv = np.linalg.lstsq(others, rank1_tensors[c], rcond=None)
                residual = np.linalg.norm(rank1_tensors[c] - others @ x)
                
                if residual < 1e-6:
                    # Column c is redundant! Remove it
                    new_U = np.delete(U_np, c, axis=1)
                    new_V = np.delete(V_np, c, axis=1)
                    new_W = np.delete(W_np, c, axis=1)
                    
                    # Verify the solution still works
                    T_est = np.zeros((8, 16, 8), dtype=float)
                    for r in range(current_rank - 1):
                        T_est += np.einsum('r,s,t->rst', new_U[:, r], new_V[:, r], new_W[:, r])
                    
                    if np.allclose(T_est, T, atol=1e-9):
                        # Convert back to lists of integers/halves
                        # Round to nearest 0.5
                        new_U_int = np.round(new_U * 2) / 2
                        new_V_int = np.round(new_V * 2) / 2
                        new_W_int = np.round(new_W * 2) / 2
                        
                        # Verify with exact values
                        T_est2 = np.zeros((8, 16, 8), dtype=float)
                        for r in range(current_rank - 1):
                            T_est2 += np.einsum('r,s,t->rst', new_U_int[:, r], new_V_int[:, r], new_W_int[:, r])
                        
                        if np.allclose(T_est2, T, atol=1e-9):
                            return new_U_int.tolist(), new_V_int.tolist(), new_W_int.tolist(), current_rank - 1
            except:
                pass
        
        # Try merging: replace two columns with one that's their sum
        # This doesn't reduce rank, but might enable further reductions
        # For now, just return the original
        return None

    # Start search for lower rank
    best_rank = rank_known
    best_U, best_V, best_W = U_known, V_known, W_known
    
    start_time = time.time()
    deadline = start_time + 130  # 20s margin from 150s limit
    
    # Iteratively try to reduce rank by finding and removing redundant columns
    U_current = np.array(U_known, dtype=float)
    V_current = np.array(V_known, dtype=float)
    W_current = np.array(W_known, dtype=float)
    current_rank = rank_known
    
    # Repeatedly try to find and remove one redundant column
    while current_rank > 1:
        if time.time() > deadline:
            break
        
        reduced = try_reduce_rank(U_current.tolist(), V_current.tolist(), W_current.tolist(), 
                                   current_rank, current_rank - 1, deadline)
        
        if reduced is not None:
            U_current = np.array(reduced[0], dtype=float)
            V_current = np.array(reduced[1], dtype=float)
            W_current = np.array(reduced[2], dtype=float)
            current_rank = reduced[3]
            best_rank = current_rank
            best_U = U_current.tolist()
            best_V = V_current.tolist()
            best_W = W_current.tolist()
        else:
            # Try some random perturbations to escape local optima
            improved = False
            for attempt in range(5):
                if time.time() > deadline:
                    break
                
                # Randomly perturb some entries slightly
                U_pert = U_current.copy()
                V_pert = V_current.copy()
                W_pert = W_current.copy()
                
                # Perturb a few random entries
                for _ in range(3):
                    u_row = random.randint(0, U_pert.shape[0] - 1)
                    u_col = random.randint(0, U_pert.shape[1] - 1)
                    U_pert[u_row, u_col] += random.choice([-0.5, 0.5])
                    
                    v_row = random.randint(0, V_pert.shape[0] - 1)
                    v_col = random.randint(0, V_pert.shape[1] - 1)
                    V_pert[v_row, v_col] += random.choice([-0.5, 0.5])
                    
                    w_row = random.randint(0, W_pert.shape[0] - 1)
                    w_col = random.randint(0, W_pert.shape[1] - 1)
                    W_pert[w_row, w_col] += random.choice([-0.5, 0.5])
                
                # Check if still valid
                T_est = np.zeros((8, 16, 8), dtype=float)
                for r in range(current_rank):
                    T_est += np.einsum('r,s,t->rst', U_pert[:, r], V_pert[:, r], W_pert[:, r])
                
                if np.allclose(T_est, T, atol=1e-9):
                    # Try to reduce from this perturbed solution
                    reduced = try_reduce_rank(U_pert.tolist(), V_pert.tolist(), W_pert.tolist(),
                                               current_rank, current_rank - 1, deadline)
                    if reduced is not None:
                        U_current = np.array(reduced[0], dtype=float)
                        V_current = np.array(reduced[1], dtype=float)
                        W_current = np.array(reduced[2], dtype=float)
                        current_rank = reduced[3]
                        best_rank = current_rank
                        best_U = U_current.tolist()
                        best_V = V_current.tolist()
                        best_W = W_current.tolist()
                        improved = True
                        break
            
            if not improved:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
