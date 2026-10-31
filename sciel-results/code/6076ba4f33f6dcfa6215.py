"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Recursive Strassen decomposition for 4x4 matrix multiplication.
    Rank: 49 (7 outer multiplications * 7 inner multiplications).
    This is the optimal known rank for 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    _t0 = time.time()
    _DEADLINE = 150.0
    _MARGIN = 5.0
    
    rank = 49
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    # Note: The contract says W[k*n+i, r]. 
    # Standard matrix mult C = A B. C[i, k] = sum_j A[i, j] B[j, k].
    # U maps A to intermediate. V maps B to intermediate.
    # Usually U is (n*m, rank), V is (m*p, rank), W is (n*p, rank) or similar.
    # Let's stick to the provided structure in the seed which worked.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get W index based on trace convention W[k*n + i, r]
    # where C[i, k] is the element.
    def w_idx(i, k):
        return k * n + i

    # Define the 2x2 blocks of A and B
    # A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
    # A[i, j] global index is i * 4 + j.
    
    # We represent linear combinations of A and B elements.
    # An element is a tuple (row, col, coeff).
    # A matrix of such elements is a 2x2 list of lists of tuples.
    
    def make_A_block(br, bc):
        """Create 2x2 matrix of A elements for block at (br, bc)"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                # Global row = br*2 + i, Global col = bc*2 + j
                row.append([(br*2+i, bc*2+j, 1)])
            mat.append(row)
        return mat

    def make_B_block(br, bc):
        """Create 2x2 matrix of B elements for block at (br, bc)"""
        mat = []
        for i in range(2):
            row = []
            for j in range(2):
                # Global row = br*2 + i, Global col = bc*2 + j
                row.append([(br*2+i, bc*2+j, 1)])
            mat.append(row)
        return mat

    def add_mat(X, Y):
        """Add two 2x2 matrices of linear combos"""
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                d = {}
                for (r, c, v) in X[i][j]:
                    d[(r,c)] = d.get((r,c), 0) + v
                for (r, c, v) in Y[i][j]:
                    d[(r,c)] = d.get((r,c), 0) + v
                row.append([(r, c, v) for (r,c), v in d.items() if v != 0])
            result.append(row)
        return result

    def sub_mat(X, Y):
        """Subtract two 2x2 matrices of linear combos"""
        neg_Y = [[[(r, c, -v) for (r,c,v) in Y[i][j]] for j in range(2)] for i in range(2)]
        return add_mat(X, neg_Y)

    # Define blocks
    A00 = make_A_block(0, 0)
    A01 = make_A_block(0, 1)
    A10 = make_A_block(1, 0)
    A11 = make_A_block(1, 1)
    
    B00 = make_B_block(0, 0)
    B01 = make_B_block(0, 1)
    B10 = make_B_block(1, 0)
    B11 = make_B_block(1, 1)

    # Outer Strassen Factors
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    outer_X = [
        add_mat(A00, A11),       # M1
        add_mat(A10, A11),       # M2
        A00,                     # M3
        A11,                     # M4
        add_mat(A00, A01),       # M5
        sub_mat(A10, A00),       # M6
        sub_mat(A01, A11),       # M7
    ]
    outer_Y = [
        add_mat(B00, B11),       # M1
        B00,                     # M2
        sub_mat(B01, B11),       # M3
        sub_mat(B10, B00),       # M4
        B11,                     # M5
        add_mat(B00, B01),       # M6
        add_mat(B10, B11),       # M7
    ]
    
    # Outer Contributions to C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # c_targets[t] = list of (block_row, block_col, coeff)
    c_targets = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,-1)],     # M2
        [(0,1,1), (1,1,1)],      # M3
        [(0,0,1), (1,0,1)],      # M4
        [(0,0,-1), (0,1,1)],     # M5
        [(1,1,1)],               # M6
        [(0,0,1)],               # M7
    ]
    
    # Inner Strassen Factors for 2x2 matrices X, Y
    # X and Y are 2x2 matrices of linear combos.
    # We need to decompose the product of two 2x2 matrices into 7 scalar multiplications.
    # Let X = [[X00, X01], [X10, X11]] and Y = [[Y00, Y01], [Y10, Y11]]
    # Z = X Y
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Inner Targets: how each inner multiplication M_k contributes to Z elements
    # inner_targets[k] = list of (local_row, local_col, coeff)
    inner_targets = [
        [(0,0,1), (1,1,1)],      # M1 -> Z00(+1), Z11(+1)
        [(1,0,1), (1,1,-1)],     # M2 -> Z10(+1), Z11(-1)
        [(0,1,1), (1,1,1)],      # M3 -> Z01(+1), Z11(+1)
        [(0,0,1), (1,0,1)],      # M4 -> Z00(+1), Z10(+1)
        [(0,0,-1), (0,1,1)],     # M5 -> Z00(-1), Z01(+1)
        [(1,1,1)],               # M6 -> Z11(+1)
        [(0,0,1)],               # M7 -> Z00(+1)
    ]
    
    def inner_factors(X, Y):
        """
        Returns 7 pairs (X_f, Y_f) where X_f and Y_f are 2x2 matrices of linear combos
        such that the scalar multiplication of the 'flattened' vectors corresponds to the rank-1 term.
        However, since we are building U, V, W directly, we just need the linear combos for U and V.
        For a specific rank-1 term, U column is formed by X_f, V column by Y_f.
        X_f and Y_f are 2x2 matrices.
        """
        
        def single_entry(i, j, mat):
            """Create 2x2 matrix with only entry (i,j) from mat"""
            result = [[[] for _ in range(2)] for _ in range(2)]
            result[i][j] = list(mat[i][j])
            return result
        
        def add_entries(i1, j1, i2, j2, mat):
            """Create 2x2 matrix with entries (i1,j1) and (i2,j2) from mat"""
            result = [[[] for _ in range(2)] for _ in range(2)]
            d = {}
            for (r, c, v) in mat[i1][j1]:
                d[(r,c)] = d.get((r,c), 0) + v
            for (r, c, v) in mat[i2][j2]:
                d[(r,c)] = d.get((r,c), 0) + v
            result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
            return result
        
        def sub_entries(i1, j1, i2, j2, mat):
            """Create 2x2 matrix with entry (i1,j1) = mat[i1][j1] - mat[i2][j2]"""
            result = [[[] for _ in range(2)] for _ in range(2)]
            d = {}
            for (r, c, v) in mat[i1][j1]:
                d[(r,c)] = d.get((r,c), 0) + v
            for (r, c, v) in mat[i2][j2]:
                d[(r,c)] = d.get((r,c), 0) - v
            result[i1][j1] = [(r, c, v) for (r,c), v in d.items() if v != 0]
            return result
        
        # M1: (X00+X11)(Y00+Y11)
        X1 = add_entries(0, 0, 1, 1, X)
        Y1 = add_entries(0, 0, 1, 1, Y)
        
        # M2: (X10+X11)Y00
        X2 = add_entries(1, 0, 1, 1, X)
        Y2 = single_entry(0, 0, Y)
        
        # M3: X00(Y01-Y11)
        X3 = single_entry(0, 0, X)
        Y3 = sub_entries(0, 1, 1, 1, Y)
        
        # M4: X11(Y10-Y00)
        X4 = single_entry(1, 1, X)
        Y4 = sub_entries(1, 0, 0, 0, Y)
        
        # M5: (X00+X01)Y11
        X5 = add_entries(0, 0, 0, 1, X)
        Y5 = single_entry(1, 1, Y)
        
        # M6: (X10-X00)(Y00+Y01)
        X6 = sub_entries(1, 0, 0, 0, X)
        Y6 = add_entries(0, 0, 0, 1, Y)
        
        # M7: (X01-X11)(Y10+Y11)
        X7 = sub_entries(0, 1, 1, 1, X)
        Y7 = add_entries(1, 0, 1, 1, Y)
        
        return [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7)]
    
    col = 0
    for t in range(7):
        X = outer_X[t]
        Y = outer_Y[t]
        factors = inner_factors(X, Y)
        
        for k in range(7):
            Xf, Yf = factors[k]
            
            # Fill U from Xf
            # Xf is a 2x2 matrix of linear combos of A.
            # Each entry Xf[i][j] is a list of (a_r, a_c, a_v).
            # We need to map these to U indices.
            # U row index corresponds to A element index: a_r * m + a_c.
            for i in range(2):
                for j in range(2):
                    for (a_r, a_c, a_v) in Xf[i][j]:
                        U[a_r * m + a_c][col] += a_v
            
            # Fill V from Yf
            # Yf is a 2x2 matrix of linear combos of B.
            # V row index corresponds to B element index: b_r * p + b_c.
            for i in range(2):
                for j in range(2):
                    for (b_r, b_c, b_v) in Yf[i][j]:
                        V[b_r * p + b_c][col] += b_v
            
            # Fill W from c_targets and inner_targets
            # The product of the k-th inner term of the t-th outer term contributes to C.
            # Outer term t contributes to C blocks (br, bc) with coeff c_coeff.
            # Inner term k contributes to Z elements (li, lj) with coeff z_coeff.
            # Z is the 2x2 block result.
            # Z[li][lj] corresponds to C element at global row (br*2 + li) and global col (bc*2 + lj).
            # So global_i = br*2 + li, global_k = bc*2 + lj.
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k)][col] += c_coeff * z_coeff
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify correctness with random test cases
    if time.time() < _DEADLINE - _MARGIN:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for _ in range(3):
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            # Compute C from decomposition
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_r = U_arr[:, r]
                v_r = V_arr[:, r]
                w_r = W_arr[:, r]
                
                # u_r is length n*m, represents linear functional on A
                # f_r(A) = sum_{i,j} u_r[i*m+j] * A[i,j]
                f_r = np.sum(u_r.reshape(n, m) * A_test)
                
                # v_r is length m*p, represents linear functional on B
                # g_r(B) = sum_{j,k} v_r[j*p+k] * B[j,k]
                g_r = np.sum(v_r.reshape(m, p) * B_test)
                
                # w_r is length p*n, w_r[k*n+i] gives contribution to C[i,k]
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += w_r[k * n + i] * f_r * g_r
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Fallback: return naive decomposition
                U_naive = np.zeros((n*m, n*m*p))
                V_naive = np.zeros((m*p, n*m*p))
                W_naive = np.zeros((p*n, n*m*p))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_naive[i*m+j, idx] = 1
                            V_naive[j*p+k, idx] = 1
                            W_naive[k*n+i, idx] = 1
                            idx += 1
                return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
