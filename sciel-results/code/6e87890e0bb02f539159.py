"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    rank_init = 36
    
    U = [[0] * rank_init for _ in range(n * m)]
    V = [[0] * rank_init for _ in range(m * p)]
    W = [[0] * rank_init for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00+y11)(z00+z11) -> X00, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10+y11)z00 -> X10, -X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00(z01-z11) -> X01, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11(z10-z00) -> X00, X10
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00+y01)z11 -> -X00, X01
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10-y00)(z00+z01) -> X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01-y11)(z10+z11) -> X00
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
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
    
    # Block 5: A_L (cols 0,1) * B_15 (rows 0,1; col 4)
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Block 6: A_R (cols 2,3) * B_25 (rows 2,3; col 4)
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Now try to reduce rank by finding linearly dependent columns
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomposition(U_arr, V_arr, W_arr):
        """Check if the decomposition is valid by testing on random inputs."""
        np.random.seed(42)
        for _ in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(W_arr.shape[1]):
                C_computed += U_arr @ V_arr[:, r] * W_arr[:, r].reshape(p, n).T
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    def try_remove_columns(U_arr, V_arr, W_arr, max_removals=5):
        """Try to remove linearly dependent columns."""
        current_U = U_arr.copy()
        current_V = V_arr.copy()
        current_W = W_arr.copy()
        current_rank = current_U.shape[1]
        
        removed = 0
        for attempt in range(max_removals):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break
            
            found_removal = False
            for col_idx in range(current_rank):
                if time.time() > deadline:
                    break
                
                # Check if column col_idx is in the span of other columns
                # Build the tensor contribution of each column
                # A column c contributes: outer(U[:,c], V[:,c]) * W[:,c]
                # We check if column c's tensor is in the span of others
                
                # Build matrix of all column tensors except col_idx
                other_cols = [c for c in range(current_rank) if c != col_idx]
                
                # The tensor for column c is: T_c = (U_c ⊗ V_c) * W_c (element-wise scaling)
                # Actually, the contribution is: for each (i,j,k), T_c[i,j,k] = U[i,c]*V[j,c]*W[k,c]
                
                # Let's build the full tensor for each column
                target_tensor = np.zeros((n*m, m*p, p*n))
                for i in range(n*m):
                    for j in range(m*p):
                        for k in range(p*n):
                            target_tensor[i,j,k] = current_U[i,col_idx] * current_V[j,col_idx] * current_W[k,col_idx]
                
                # Build basis from other columns
                if len(other_cols) == 0:
                    continue
                    
                # Flatten target tensor
                target_flat = target_tensor.flatten()
                
                # Build matrix of other column tensors
                basis = np.zeros((n*m * m*p * p*n, len(other_cols)))
                for idx, oc in enumerate(other_cols):
                    for i in range(n*m):
                        for j in range(m*p):
                            for k in range(p*n):
                                basis[i*(m*p)*(p*n) + j*(p*n) + k, idx] = current_U[i,oc] * current_V[j,oc] * current_W[k,oc]
                
                # Check if target is in span of basis
                # Use least squares
                try:
                    coeffs, residuals, rank_basis, sv = np.linalg.lstsq(basis, target_flat, rcond=None)
                    reconstruction = basis @ coeffs
                    if np.allclose(reconstruction, target_flat, atol=1e-10):
                        # Column is redundant, remove it
                        new_U = np.delete(current_U, col_idx, axis=1)
                        new_V = np.delete(current_V, col_idx, axis=1)
                        new_W = np.delete(current_W, col_idx, axis=1)
                        
                        if verify_decomposition(new_U, new_V, new_W):
                            current_U = new_U
                            current_V = new_V
                            current_W = new_W
                            current_rank -= 1
                            removed += 1
                            found_removal = True
                            break
                except:
                    continue
            
            if not found_removal:
                break
        
        return current_U, current_V, current_W, current_rank
    
    # Try to remove columns
    best_U, best_V, best_W, best_rank = try_remove_columns(U_np, V_np, W_np, max_removals=3)
    
    # Convert back to lists
    best_U_list = [[best_U[i,j] for j in range(best_U.shape[1])] for i in range(best_U.shape[0])]
    best_V_list = [[best_V[i,j] for j in range(best_V.shape[1])] for i in range(best_V.shape[0])]
    best_W_list = [[best_W[i,j] for j in range(best_W.shape[1])] for i in range(best_W.shape[0])]
    
    # Clean up floating point values to integers/half-integers if possible
    def clean_val(v):
        if abs(v - round(v)) < 1e-10:
            return int(round(v))
        elif abs(v*2 - round(v*2)) < 1e-10:
            return round(v*2) / 2.0
        else:
            return v
    
    best_U_list = [[clean_val(v) for v in row] for row in best_U_list]
    best_V_list = [[clean_val(v) for v in row] for row in best_V_list]
    best_W_list = [[clean_val(v) for v in row] for row in best_W_list]
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
