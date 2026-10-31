"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Recursive Strassen decomposition for 4x4 matrix multiplication, rank 49.
    Attempts rank reduction to 48 via non-commutative ring exploitation."""
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # Define the 7 outer Strassen terms
    # Each term: X (2x2 of A-entry linear combos), Y (2x2 of B-entry linear combos)
    # X[i][j] = list of (a_row, a_col, coeff)
    # Y[i][j] = list of (b_row, b_col, coeff)
    
    def w_idx(block_r, block_c, local_r, local_c):
        row = block_r * 2 + local_r
        col = block_c * 2 + local_c
        return col * n + row
    
    outer_terms = []
    
    # M1: X = A00+A11, Y = B00+B11
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (2+i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (2+i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M2: X = A10+A11, Y = B00
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (2+i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M3: X = A00, Y = B01-B11
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, 2+j, 1), (2+i, 2+j, -1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M4: X = A11, Y = B10-B00
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (i, j, -1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M5: X = A00+A01, Y = B11
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (i, 2+j, 1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M6: X = A10-A00, Y = B00+B01
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (i, j, -1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, j, 1), (i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # M7: X = A01-A11, Y = B10+B11
    X = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(i, 2+j, 1), (2+i, 2+j, -1)])
        X.append(row)
    Y = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append([(2+i, j, 1), (2+i, 2+j, 1)])
        Y.append(row)
    outer_terms.append((X, Y))
    
    # C targets for each outer term
    c_targets = [
        [(0,0,1), (1,1,1)],       # M1 -> C00, C11
        [(1,0,1), (1,1,-1)],      # M2 -> C10, C11
        [(0,1,1), (1,1,1)],       # M3 -> C01, C11
        [(0,0,1), (1,0,1)],       # M4 -> C00, C10
        [(0,0,-1), (0,1,1)],      # M5 -> C00, C01
        [(1,1,1)],                 # M6 -> C11
        [(0,0,1)],                 # M7 -> C00
    ]
    
    # For each outer term, apply inner Strassen to compute X*Y
    col = 0
    for t in range(7):
        X = outer_terms[t][0]
        Y = outer_terms[t][1]
        
        # Helper: add two linear combinations
        def add_combos(c1, c2):
            result = {}
            for (r, c, v) in c1:
                result[(r,c)] = result.get((r,c), 0) + v
            for (r, c, v) in c2:
                result[(r,c)] = result.get((r,c), 0) + v
            return [(r, c, v) for (r,c), v in result.items() if v != 0]
        
        # Helper: subtract
        def sub_combos(c1, c2):
            return add_combos(c1, [(r, c, -v) for (r,c,v) in c2])
        
        # Inner M1: (X00+X11)(Y00+Y11)
        X_in = add_combos(X[0][0], X[1][1])
        Y_in = add_combos(Y[0][0], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M2: (X10+X11)Y00
        X_in = add_combos(X[1][0], X[1][1])
        Y_in = Y[0][0]
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 1, 0)][col] += coeff
            W[w_idx(br, bc, 1, 1)][col] -= coeff
        col += 1
        
        # Inner M3: X00(Y01-Y11)
        X_in = X[0][0]
        Y_in = sub_combos(Y[0][1], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 1)][col] += coeff
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M4: X11(Y10-Y00)
        X_in = X[1][1]
        Y_in = sub_combos(Y[1][0], Y[0][0])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
            W[w_idx(br, bc, 1, 0)][col] += coeff
        col += 1
        
        # Inner M5: (X00+X01)Y11
        X_in = add_combos(X[0][0], X[0][1])
        Y_in = Y[1][1]
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] -= coeff
            W[w_idx(br, bc, 0, 1)][col] += coeff
        col += 1
        
        # Inner M6: (X10-X00)(Y00+Y01)
        X_in = sub_combos(X[1][0], X[0][0])
        Y_in = add_combos(Y[0][0], Y[0][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 1, 1)][col] += coeff
        col += 1
        
        # Inner M7: (X01-X11)(Y10+Y11)
        X_in = sub_combos(X[0][1], X[1][1])
        Y_in = add_combos(Y[1][0], Y[1][1])
        for (a_r, a_c, a_v) in X_in:
            U[a_r * m + a_c][col] += a_v
        for (b_r, b_c, b_v) in Y_in:
            V[b_r * p + b_c][col] += b_v
        for (br, bc, coeff) in c_targets[t]:
            W[w_idx(br, bc, 0, 0)][col] += coeff
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to numpy for fast verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def fast_verify(U_mat, V_mat, W_mat):
        """Fast verification using batched matrix ops.
        For each (ar, bc): M[ac, br] = sum_r U[ar*4+ac,r]*V[br*4+bc,r]*W[bc*4+ar,r]
        should equal I_4.
        """
        R = U_mat.shape[1]
        for ar in range(4):
            U_ar = U_mat[ar*4:(ar+1)*4, :]  # 4 x R
            for bc in range(4):
                W_vec = W_mat[bc*4+ar, :]  # R
                V_bc = V_mat[bc::4, :]  # 4 x R
                M = (U_ar * W_vec[None, :]) @ V_bc.T  # 4 x 4
                if not np.allclose(M, np.eye(4), atol=1e-9):
                    return False
        return True
    
    # Verify the initial decomposition
    if not fast_verify(U_np, V_np, W_np):
        return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline - 30:
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank reduction by trying to remove columns one at a time
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    
    # Greedy removal: try multiple passes
    improved = True
    while improved:
        improved = False
        for col_to_try in range(current_rank - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            
            # Try removing this column
            keep_cols = [c for c in range(current_rank) if c != col_to_try]
            U_new = current_U[:, keep_cols]
            V_new = current_V[:, keep_cols]
            W_new = current_W[:, keep_cols]
            
            if fast_verify(U_new, V_new, W_new):
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                break  # restart scan from top
    
    if current_rank < rank:
        U_final = current_U.tolist()
        V_final = current_V.tolist()
        W_final = current_W.tolist()
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # If greedy removal didn't work, try a more aggressive approach:
    # Try removing pairs of columns simultaneously (non-commutative exploitation)
    if time.time() < deadline - 20:
        # Try removing specific pairs that might be redundant due to ring structure
        # The key insight: in the 2x2 matrix ring, certain combinations are not independent
        # because AB != BA in general, but specific linear combinations may still be redundant
        
        # Try all pairs of columns to remove
        for i in range(current_rank - 1):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, current_rank):
                if time.time() > deadline - 10:
                    break
                
                keep_cols = [c for c in range(current_rank) if c != i and c != j]
                U_new = current_U[:, keep_cols]
                V_new = current_V[:, keep_cols]
                W_new = current_W[:, keep_cols]
                
                if fast_verify(U_new, V_new, W_new):
                    current_U = U_new
                    current_V = V_new
                    current_W = W_new
                    current_rank -= 2
                    # Continue trying to remove more
                    improved = True
                    break
            
            if improved:
                break
    
    if current_rank < rank:
        U_final = current_U.tolist()
        V_final = current_V.tolist()
        W_final = current_W.tolist()
        return {"U": U_final, "V": V_final, "W": W_final}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
