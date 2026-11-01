import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    # Original dimensions: A (n x m), B (m x p), C (n x p)
    # Here: A (2x2), B (2x7), C (2x7)
    # Indices:
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (14, rank)
    # W: (p*n, rank) = (14, rank)
    
    # Build target tensor T of shape (4, 14, 14)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    # Schoolbook decomposition (rank 28)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    # --- Strategy: Exploit commutativity to map to 7x7x2 problem ---
    # The multiplication of A (2x2) and B (2x7) results in C (2x7).
    # This is equivalent to multiplying B^T (7x2) and A^T (2x2) to get C^T (7x2)? No.
    # (AB)^T = B^T A^T.
    # Let X = B^T (7x2), Y = A^T (2x2). Then Z = X Y is (7x2).
    # The tensor for 7x2 x 2x2 -> 7x2 multiplication has dimensions:
    # Input 1: 7*2 = 14
    # Input 2: 2*2 = 4
    # Output: 7*2 = 14
    # This is the same shape (14, 4, 14) but with different index mappings.
    
    # However, there is a known result for 7x7 x 2 multiplication (rank 11).
    # Can we embed our problem into that?
    # Our problem: 2x2 * 2x7 -> 2x7.
    # This is a block of a larger multiplication.
    
    # Let's try to construct a rank-11 solution based on the known 7x7x2 decomposition.
    # The known rank-11 decomposition for multiplying a 7x7 matrix by a 7x2 matrix (or 2x7?)
    # Actually, the "rank 11" usually refers to 7x7 x 7x2? No, 7x7 x 2x7 is not standard.
    # Standard small matrix multiplication ranks:
    # 2x2 x 2x2: rank 7 (Strassen)
    # 7x7 x 7x7: rank 11 (Alon-Boppana?) No, 7x7x7 rank is 11? 
    # Wait, the rank of 7x7x7 multiplication is 11.
    # If we can map our 2x2 x 2x7 problem to a sub-problem of 7x7 x 7x7, 
    # we might use the 11-term decomposition.
    
    # However, a direct mapping is non-trivial. 
    # Let's look at the structure of the tensor.
    # T is 4x14x14.
    # We know rank <= 28.
    # Is rank 11 possible?
    # Lower bound for 2x2 x 2x7:
    # The tensor is the Kronecker product of (2x2 x 2x2) and (2x1 x 1x7)? No.
    # It's a block diagonal structure if we view it differently?
    
    # Let's try a different structural approach: 
    # Use the Strassen decomposition for the 2x2 part and extend it to the 7 columns.
    # Strassen for 2x2 x 2x2 uses 7 terms.
    # For 2x2 x 2x7, we can view B as [B1 B2] where B1, B2 are 2x3 and 2x4? No, 7 is odd.
    # B is 2x7. 
    # We can split the 7 columns into groups.
    # If we use Strassen on the 2x2 block of A and the 2x2 blocks of B, we get 7 terms per 2 columns of B.
    # 7 columns = 3 pairs + 1 single.
    # 3 * 7 + 4 = 25? Or 3 * 7 + 2 (for the single column 2x1 x 1x2?) 
    # 2x2 * 2x1 requires rank 4 (schoolbook) or maybe less?
    # 2x1 * 1x2 is rank 2? No, 2x1 matrix A, 1x2 matrix B. Product is 2x2.
    # A = [a, b]^T, B = [c, d]. Product = [[ac, ad], [bc, bd]].
    # This is rank 4 (outer product of A and B).
    # So 3 * 7 (Strassen for 2x2 blocks) + 4 (rank for 2x1*1x2) = 25.
    # This is a concrete structural improvement from 28 to 25.
    
    # Let's implement the rank-25 decomposition.
    # B (2x7) = [B_1 | B_2 | B_3 | b_4]
    # B_1, B_2, B_3 are 2x2. b_4 is 2x1.
    # A (2x2).
    # C = A * B = [A*B_1 | A*B_2 | A*B_3 | A*b_4]
    
    # For each 2x2 block multiplication, use Strassen (rank 7).
    # For the 2x1 part, use rank 4.
    # Total rank = 7 + 7 + 7 + 4 = 25.
    
    def strassen_2x2(a, b):
        # Returns 7 scalar terms
        # a = [[a00, a01], [a10, a11]]
        # b = [[b00, b01], [b10, b11]]
        m1 = (a[0,0] + a[1,1]) * (b[0,0] + b[1,1])
        m2 = (a[1,0] + a[1,1]) * b[0,0]
        m3 = a[0,0] * (b[0,1] - b[1,1])
        m4 = a[1,1] * (b[1,0] - b[0,0])
        m5 = (a[0,0] + a[0,1]) * b[1,1]
        m6 = (a[1,0] - a[0,0]) * (b[0,0] + b[0,1])
        m7 = (a[0,1] - a[1,1]) * (b[1,0] + b[1,1])
        return [m1, m2, m3, m4, m5, m6, m7]

    # We need to construct U, V, W for the whole tensor.
    # The tensor T is defined by indices a, b, c.
    # a in [0,3] (2x2 A)
    # b in [0,13] (2x7 B)
    # c in [0,13] (2x7 C)
    
    # We will build the decomposition term by term.
    # Term 1-7: Strassen for block 0 (columns 0,1 of B)
    # Term 8-14: Strassen for block 1 (columns 2,3 of B)
    # Term 15-21: Strassen for block 2 (columns 4,5 of B)
    # Term 22-25: Rank 4 for column 6 of B
    
    U = np.zeros((4, 25))
    V = np.zeros((14, 25))
    W = np.zeros((14, 25))
    
    # Helper to add a term
    # term_index: which column in U,V,W
    # u_vals: dict {a_index: val}
    # v_vals: dict {b_index: val}
    # w_vals: dict {c_index: val}
    
    def add_term(t_idx, u_dict, v_dict, w_dict):
        for a_idx, val in u_dict.items():
            U[a_idx, t_idx] += val
        for b_idx, val in v_dict.items():
            V[b_idx, t_idx] += val
        for c_idx, val in w_dict.items():
            W[c_idx, t_idx] += val
            
    # Indices for A (2x2):
    # a00: 0, a01: 1, a10: 2, a11: 3
    # Indices for B (2x7):
    # b00: 0, b01: 1, b02: 2, b03: 3, b04: 4, b05: 5, b06: 6
    # b10: 7, b11: 8, b12: 9, b13: 10, b14: 11, b15: 12, b16: 13
    # Indices for C (2x7):
    # c00: 0, c01: 1, c02: 2, c03: 3, c04: 4, c05: 5, c06: 6
    # c10: 7, c11: 8, c12: 9, c13: 10, c14: 11, c15: 12, c16: 13
    
    # Strassen terms for 2x2 * 2x2:
    # Inputs: A (2x2), B_block (2x2)
    # Outputs: C_block (2x2)
    
    # Let's define the Strassen coefficients for U, V, W.
    # For a specific block of B (cols j, j+1) and corresponding block of C (cols j, j+1).
    
    def add_strassen_block(b_col_start, c_col_start, t_start):
        # b_col_start is 0, 2, 4
        # b indices:
        # b0_j: b_col_start
        # b0_j+1: b_col_start + 1
        # b1_j: b_col_start + 7
        # b1_j+1: b_col_start + 8
        
        # c indices:
        # c0_j: c_col_start
        # c0_j+1: c_col_start + 1
        # c1_j: c_col_start + 7
        # c1_j+1: c_col_start + 8
        
        # m1 = (a00+a11)(b00+b11)
        # U: a00:1, a11:1
        # V: b00:1, b11:1
        # W: c00:1, c11:1, c01:-1, c10:-1 (Wait, C00 = m1 - m4 - m5 + m7, C01 = m3 + m5, C10 = m2 + m4, C11 = m1 - m2 + m6 - m7)
        # Let's use the standard Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 - M7
        
        # Term 1 (M1):
        # U: {0:1, 3:1}
        # V: {b00:1, b11:1}
        # W: {c00:1, c11:1}
        add_term(t_start, 
                 {0:1, 3:1}, 
                 {b_col_start:1, b_col_start+8:1}, 
                 {c_col_start:1, c_col_start+8:1})
                 
        # Term 2 (M2):
        # U: {2:1, 3:1}
        # V: {b00:1}
        # W: {c10:1}
        add_term(t_start+1, 
                 {2:1, 3:1}, 
                 {b_col_start:1}, 
                 {c_col_start+7:1})
                 
        # Term 3 (M3):
        # U: {0:1}
        # V: {b01:1, b11:-1}
        # W: {c01:1}
        add_term(t_start+2, 
                 {0:1}, 
                 {b_col_start+1:1, b_col_start+8:-1}, 
                 {c_col_start+1:1})
                 
        # Term 4 (M4):
        # U: {3:1}
        # V: {b10:1, b00:-1}
        # W: {c00:-1, c10:1}  # C00 has -M4, C10 has +M4
        add_term(t_start+3, 
                 {3:1}, 
                 {b_col_start+7:1, b_col_start:-1}, 
                 {c_col_start:-1, c_col_start+7:1})
                 
        # Term 5 (M5):
        # U: {0:1, 1:1}
        # V: {b11:1}
        # W: {c00:-1, c01:1} # C00 has -M5, C01 has +M5
        add_term(t_start+4, 
                 {0:1, 1:1}, 
                 {b_col_start+8:1}, 
                 {c_col_start:-1, c_col_start+1:1})
                 
        # Term 6 (M6):
        # U: {2:-1, 0:1} # (A10-A00) -> A10:1, A00:-1? No, A10 is idx 2, A00 is idx 0.
        # U: {2:1, 0:-1}
        # V: {b00:1, b01:1}
        # W: {c11:1}
        add_term(t_start+5, 
                 {2:1, 0:-1}, 
                 {b_col_start:1, b_col_start+1:1}, 
                 {c_col_start+8:1})
                 
        # Term 7 (M7):
        # U: {1:1, 3:-1} # (A01-A11) -> A01:1, A11:-1
        # V: {b10:1, b11:1}
        # W: {c00:1, c11:-1} # C00 has +M7, C11 has -M7
        add_term(t_start+6, 
                 {1:1, 3:-1}, 
                 {b_col_start+7:1, b_col_start+8:1}, 
                 {c_col_start:1, c_col_start+8:-1})

    # Add 3 Strassen blocks
    add_strassen_block(0, 0, 0)   # Cols 0,1
    add_strassen_block(2, 2, 7)   # Cols 2,3
    add_strassen_block(4, 4, 14)  # Cols 4,5
    
    # Add rank-4 block for col 6
    # B is 2x1: [b06, b16]^T. Indices: b06=6, b16=13.
    # A is 2x2.
    # C is 2x1: [c06, c16]^T. Indices: c06=6, c16=13.
    # C06 = A00*B06 + A01*B16
    # C16 = A10*B06 + A11*B16
    
    # Term 8 (t=21): A00 * B06 -> C06
    add_term(21, {0:1}, {6:1}, {6:1})
    
    # Term 9 (t=22): A01 * B16 -> C06
    add_term(22, {1:1}, {13:1}, {6:1})
    
    # Term 10 (t=23): A10 * B06 -> C16
    add_term(23, {2:1}, {6:1}, {13:1})
    
    # Term 11 (t=24): A11 * B16 -> C16
    add_term(24, {3:1}, {13:1}, {13:1})
    
    # Check validity
    if check_decomposition(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = 25
        
    # If we have time, we could try to optimize this further, but rank 25 is a solid structural improvement.
    # We can also try to see if we can reduce the rank of the 2x1 part? 
    # 2x1 * 1x2 is rank 4. Can it be lower?
    # The tensor for 2x1 x 1x2 is 2x1x2. 
    # T[0,0,0]=1, T[0,0,1]=1, T[1,0,0]=1, T[1,0,1]=1? 
    # No, indices:
    # A: 2x1 (idx 0,1). B: 1x2 (idx 0,1). C: 2x2 (idx 0,1,2,3).
    # C00 = A0 B0, C01 = A0 B1, C10 = A1 B0, C11 = A1 B1.
    # This is the outer product of A and B. Rank 4.
    # So 25 is likely the minimum for this block decomposition.
    
    # Can we do better than 25?
    # Maybe a different splitting?
    # 7 = 1 + 1 + 1 + 1 + 1 + 1 + 1? Rank 7*4 = 28.
    # 7 = 2 + 2 + 2 + 1? Rank 7+7+7+4 = 25.
    # 7 = 3 + 2 + 2? 
    # 2x2 * 2x3 multiplication.
    # Rank of 2x2 x 2x3?
    # Lower bound: max(rank(2x2), rank(2x3))?
    # Known rank for 2x2 x 2x3 is 8? Or 9?
    # If we can find a rank 8 decomposition for 2x2 x 2x3, then:
    # 7 = 3 + 2 + 2 -> 8 + 7 + 7 = 22.
    # 7 = 3 + 3 + 1 -> 8 + 8 + 4 = 20.
    
    # Let's try to search for a rank 8 decomposition for 2x2 x 2x3.
    # If we find one, we can implement it.
    
    def find_2x2_2x3_decomposition(max_time=10):
        # T shape: (4, 6, 6)
        # A: 2x2, B: 2x3, C: 2x2? No, C is 2x3.
        # A (2x2) * B (2x3) -> C (2x3).
        # T[a, b, c].
        # a in 0..3, b in 0..5, c in 0..5.
        T_sub = np.zeros((4, 6, 6), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a = i*2 + j
                    b = j*3 + k
                    c = k*2 + i  # Wait, C is 2x3. c index is k*2 + i?
                    # C row i, col k. Index i*3 + k?
                    # The task says "output factor follows the official trace convention W[k*n+i,r]"
                    # For 2x2 x 2x7, C is 2x7. n=2, p=7.
                    # W index is k*n + i. Here n=2.
                    # So for 2x2 x 2x3, n=2, p=3.
                    # W index is k*2 + i.
                    T_sub[a, b, c] = 1.0
                    
        # Try rank 8
        U_s = np.zeros((4, 8))
        V_s = np.zeros((6, 8))
        W_s = np.zeros((6, 8))
        
        # Random init and ALS
        for _ in range(5):
            U_s = np.random.randn(4, 8) * 0.5
            V_s = np.random.randn(6, 8) * 0.5
            W_s = np.random.randn(6, 8) * 0.5
            for _ in range(50):
                # Update U
                for a in range(4):
                    M = np.zeros((36, 8))
                    for r in range(8):
                        M[:, r] = np.kron(V_s[:, r], W_s[:, r])
                    U_s[a, :] = np.linalg.lstsq(M, T_sub[a].reshape(-1), rcond=None)[0]
                # Update V
                for b in range(6):
                    M = np.zeros((24, 8))
                    for r in range(8):
                        M[:, r] = np.kron(U_s[:, r], W_s[:, r])
                    V_s[b, :] = np.linalg.lstsq(M, T_sub[:, b, :].reshape(-1), rcond=None)[0]
                # Update W
                for c in range(6):
                    M = np.zeros((24, 8))
                    for r in range(8):
                        M[:, r] = np.kron(U_s[:, r], V_s[:, r])
                    W_s[c, :] = np.linalg.lstsq(M, T_sub[:, :, c].reshape(-1), rcond=None)[0]
                    
                T_rec = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
                if np.allclose(T_rec, T_sub, atol=1e-9):
                    return U_s, V_s, W_s
        return None

    # Try to find 2x2 x 2x3 decomposition
    # Note: This is a sub-problem. If we find it, we can embed it.
    # But embedding requires careful index mapping.
    # Given time constraints, let's stick with the rank 25 solution which is guaranteed.
    # We can try to run the ALS for the full 2x2 x 2x7 problem with rank 24, 23...
    
    def try_als_full(r, max_iters=100, n_restarts=5):
        nonlocal best_U, best_V, best_W, best_rank
        if r >= best_rank:
            return False
            
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
                
            U = np.random.randn(4, r) * 0.5
            V = np.random.randn(14, r) * 0.5
            W = np.random.randn(14, r) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                    
                # Update U
                for a in range(4):
                    M = np.zeros((196, r))
                    for rr in range(r):
                        M[:, rr] = np.kron(V[:, rr], W[:, rr])
                    U[a, :] = np.linalg.lstsq(M, T[a].reshape(-1), rcond=None)[0]
                # Update V
                for b in range(14):
                    M = np.zeros((56, r))
                    for rr in range(r):
                        M[:, rr] = np.kron(U[:, rr], W[:, rr])
                    V[b, :] = np.linalg.lstsq(M, T[:, b, :].reshape(-1), rcond=None)[0]
                # Update W
                for c in range(14):
                    M = np.zeros((56, r))
                    for rr in range(r):
                        M[:, rr] = np.kron(U[:, rr], V[:, rr])
                    W[c, :] = np.linalg.lstsq(M, T[:, :, c].reshape(-1), rcond=None)[0]
                    
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_rec, T, atol=1e-9):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = r
                    return True
        return False

    # Try ranks below 25
    for r in range(14, 25):
        if time.time() > deadline:
            break
        if try_als_full(r):
            break
            
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
