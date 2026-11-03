import time
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 135.0

    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T (up to tolerance)"""
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)

    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Strategy: Block Strassen
    # Split m=4 into 2+2, p=6 into 2+2+2
    # This gives 6 blocks of 2x2x2 multiplication.
    # Each 2x2x2 uses Strassen (rank 7).
    # Total rank = 6 * 7 = 42.
    
    def make_rank42():
        U = np.zeros((n*m, 42))
        V = np.zeros((m*p, 42))
        W = np.zeros((p*n, 42))
        
        # Define blocks: (a_rows, a_cols, b_rows, b_cols)
        # a_rows/cols are indices for A (2x4)
        # b_rows/cols are indices for B (4x6)
        # We split A into two 2x2 blocks: A1 (rows 0-1, cols 0-1), A2 (rows 0-1, cols 2-3)
        # We split B into three 2x2 blocks: B1 (rows 0-1, cols 0-1), B2 (rows 0-1, cols 2-3), B3 (rows 0-1, cols 4-5)
        # Wait, B is 4x6.
        # A is 2x4.
        # A = [A1 A2] where A1, A2 are 2x2.
        # B = [B1; B2] where B1, B2 are 2x6.
        # Then A*B = A1*B1 + A2*B2.
        # B1 is 2x6. Split B1 into B1a, B1b, B1c (2x2 each).
        # B2 is 2x6. Split B2 into B2a, B2b, B2c (2x2 each).
        # So we have 6 blocks of 2x2x2.
        
        blocks = [
            # (a_row_indices, a_col_indices, b_row_indices, b_col_indices)
            # A1 is rows 0,1; cols 0,1. B1a is rows 0,1; cols 0,1.
            ([0, 1], [0, 1], [0, 1], [0, 1]),
            # A1, B1b (cols 2,3)
            ([0, 1], [0, 1], [0, 1], [2, 3]),
            # A1, B1c (cols 4,5)
            ([0, 1], [0, 1], [0, 1], [4, 5]),
            # A2 is rows 0,1; cols 2,3. B2a is rows 2,3; cols 0,1.
            ([0, 1], [2, 3], [2, 3], [0, 1]),
            # A2, B2b (cols 2,3)
            ([0, 1], [2, 3], [2, 3], [2, 3]),
            # A2, B2c (cols 4,5)
            ([0, 1], [2, 3], [2, 3], [4, 5]),
        ]
        
        # Strassen 7 multiplications for 2x2x2
        # A = [[a, b], [c, d]], B = [[e, f], [h, i]]
        # m1 = (a+d)(e+i)
        # m2 = (c+d)e
        # m3 = a(f-i)
        # m4 = d(h-e)
        # m5 = (a+b)i
        # m6 = (c-a)(e+h)
        # m7 = (b-d)(f+i)
        
        # Indices in local 2x2 block:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # Each coeff is list of (local_index, coefficient)
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # m1
            ([(2, 1), (3, 1)], [(0, 1)],       [(2, 1), (3, -1)]), # m2
            ([(0, 1)],        [(1, 1), (3, -1)],[(1, 1), (3, 1)]), # m3
            ([(3, 1)],        [(2, 1), (0, -1)],[(0, 1), (2, 1)]), # m4
            ([(0, 1), (1, 1)],  [(3, 1)],       [(0, -1), (1, 1)]), # m5
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]), # m6
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]), # m7
        ]
        
        col = 0
        for a_rows, a_cols, b_rows, b_cols in blocks:
            # Map local indices to global indices
            # A local (r, c) -> global (a_rows[r], a_cols[c]) -> index a_rows[r]*m + a_cols[c]
            a_map = {
                0: a_rows[0]*m + a_cols[0],
                1: a_rows[0]*m + a_cols[1],
                2: a_rows[1]*m + a_cols[0],
                3: a_rows[1]*m + a_cols[1]
            }
            
            # B local (r, c) -> global (b_rows[r], b_cols[c]) -> index b_rows[r]*p + b_cols[c]
            b_map = {
                0: b_rows[0]*p + b_cols[0],
                1: b_rows[0]*p + b_cols[1],
                2: b_rows[1]*p + b_cols[0],
                3: b_rows[1]*p + b_cols[1]
            }
            
            # W local (c, r) -> global (b_cols[c], a_rows[r]) -> index b_cols[c]*n + a_rows[r]
            # Note: W[k*n+i, r] where k is col of B, i is row of A.
            # In local 2x2, col c corresponds to b_cols[c], row r corresponds to a_rows[r].
            w_map = {
                0: b_cols[0]*n + a_rows[0],
                1: b_cols[1]*n + a_rows[0],
                2: b_cols[0]*n + a_rows[1],
                3: b_cols[1]*n + a_rows[1]
            }
            
            for u_c, v_c, w_c in strassen_terms:
                for li, cf in u_c:
                    U[a_map[li], col] += cf
                for li, cf in v_c:
                    V[b_map[li], col] += cf
                for li, cf in w_c:
                    W[w_map[li], col] += cf
                col += 1
        
        return U, V, W
    
    U, V, W = make_rank42()
    
    # Verify
    if not check_decomposition(U, V, W):
        # Fallback to schoolbook if Strassen fails (should not happen)
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        
    return to_lists(U, V, W)
