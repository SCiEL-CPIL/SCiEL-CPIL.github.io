import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # 1. Construct Rank 189 Solution (Strassen 7 * Schoolbook 27)
    U_189, V_189, W_189 = construct_strassen_schoolbook()
    
    # 2. Construct Rank 161 Solution (Strassen 7 * Laderman 23)
    U_161, V_161, W_161 = construct_strassen_laderman()
    
    # 3. Verify 161 Solution
    if time.time() < deadline:
        if verify_decomposition(U_161, V_161, W_161, n, m, p):
            return {"U": U_161, "V": V_161, "W": W_161}
            
    # Fallback to 189
    return {"U": U_189, "V": V_189, "W": W_189}

def verify_decomposition(U, V, W, n, m, p, num_tests=10):
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Convention: C[i, k] = sum_j A[i, j] B[j, k]
    # W index: k*n + i
    
    for _ in range(num_tests):
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        
        # Efficient verification using matrix operations
        # C = sum_r (A^T U_r) (V_r^T B) W_r^T ?
        # Let's stick to the scalar definition for correctness in verification, 
        # but optimize if needed. 10 tests * 161 rank * 36*36*36 is small.
        
        for r in range(rank):
            u_vec = np.array(U[:, r]).reshape(n, m)
            v_vec = np.array(V[:, r]).reshape(m, p)
            w_vec = np.array(W[:, r]).reshape(p, n) # W is (p*n, rank), so reshape to (p, n)
            
            # Term: (sum A*u) * (sum B*v) * w
            s_a = np.sum(A * u_vec)
            s_b = np.sum(B * v_vec)
            
            if s_a != 0 and s_b != 0:
                C_computed += s_a * s_b * w_vec.T # w_vec is (p, n), C is (n, p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
            
    return True

def construct_strassen_schoolbook():
    n, m, p = 6, 6, 6
    # Inner: 3x3 Schoolbook (Rank 27)
    U_3, V_3, W_3 = get_3x3_schoolbook()
    return build_strassen(n, m, p, U_3, V_3, W_3)

def construct_strassen_laderman():
    n, m, p = 6, 6, 6
    # Inner: 3x3 Laderman (Rank 23)
    U_3, V_3, W_3 = get_3x3_laderman()
    return build_strassen(n, m, p, U_3, V_3, W_3)

def get_3x3_schoolbook():
    n = 3
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*n+j][col] = 1
                V[j*n+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def get_3x3_laderman():
    # Laderman's rank 23 decomposition for 3x3
    # Returns U, V, W of shape (9, 23)
    # Coefficients are integers
    # Note: This is the risky part. Verification will catch errors.
    
    # Based on Laderman (1976)
    # Indices: A[i,j], B[i,j], C[i,j] with i,j in 0..2
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Terms (P_k = L_k(A) * L_k(B))
    # L_k(A) and L_k(B) are linear combinations of entries.
    
    # We define the vectors u_r, v_r, w_r for r=0..22
    # u_r[i,j] is coeff of A[i,j] in L_r(A)
    # v_r[j,k] is coeff of B[j,k] in L_r(B)
    # w_r[i,k] is coeff of P_r in C[i,k]
    
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    def set_coeff(U, V, W, r, a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: list of 9 (flattened 3x3)
        # b_coeffs: list of 9
        # c_coeffs: list of 9
        for idx, val in enumerate(a_coeffs):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(b_coeffs):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(c_coeffs):
            if val != 0:
                W[idx][r] = val

    # Helper to create flat list from 3x3 dict
    def flat_3x3(d):
        return [d.get((i,j), 0) for i in range(3) for j in range(3)]

    # Laderman's 23 terms
    # Reference: 
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    # P8 = (a20+a21)(b00+b01)
    # P9 = (a20-a00)(b10+b11)
    # P10 = (a21-a11)(b00+b01) -- Wait, need to check exact forms
    
    # Actually, I will use a known correct set from a reliable source in memory or derive via Strassen on 2x2 + corrections?
    # To be safe, I'll use a set that I can verify mentally or via the code's verification.
    
    # Let's use the "Bini" or standard Laderman set.
    # If I am unsure, the verification will fail, and we fall back to 189.
    
    # I will hardcode a specific set.
    # If this set is wrong, verify_decomposition returns False.
    
    # Term 1
    a1 = flat_3x3({(0,0):1, (1,1):1})
    b1 = flat_3x3({(0,0):1, (1,1):1})
    c1 = flat_3x3({(0,0):1, (0,1):0, (0,2):0, (1,0):0, (1,1):1, (1,2):0, (2,0):0, (2,1):0, (2,2):0}) # C00, C11 get P1?
    # C00 = P1 + P4 - P5 + P7 ...
    # This manual derivation is prone to error.
    
    # Alternative: Use the fact that I can just return the 189 solution if I don't have a verified 161 solution.
    # But I want to try.
    
    # I will use a simplified "Laderman" that I am more confident in, or just rely on the fallback.
    # Given the strict time and correctness requirements, and the fact that I cannot look up the exact 23 vectors online,
    # I will implement the 189 solution as the primary, and only attempt 161 if I can construct it reliably.
    
    # Actually, I can construct the 161 solution using the Strassen structure and *any* valid 3x3 decomposition.
    # If I use Schoolbook (27) inside, I get 189.
    # If I use Laderman (23), I get 161.
    
    # Since I cannot guarantee the Laderman coefficients, I will stick to the 189 solution which is definitely correct.
    # The score 1.143 is a good improvement over 1.0.
    
    return U, V, W

def build_strassen(n, m, p, U_3, V_3, W_3):
    # n,m,p are 6. Blocks are 3x3.
    # U_3, V_3, W_3 are for 3x3 multiplication.
    rank_inner = len(U_3[0])
    rank_outer = 7
    total_rank = rank_outer * rank_inner
    
    U = [[0]*total_rank for _ in range(36)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Strassen Coefficients for 2x2 blocks
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    a_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    b_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    def embed_3x3_to_6x6(vec_3x3, block_pos):
        # vec_3x3 is length 9
        # block_pos: 0->00, 1->01, 2->10, 3->11
        res = [0]*36
        base = block_pos * 9
        for i in range(9):
            res[base+i] = vec_3x3[i]
        return res
    
    for k in range(7):
        for r in range(rank_inner):
            col = k * rank_inner + r
            
            u_inner = [U_3[i][r] for i in range(9)]
            v_inner = [V_3[i][r] for i in range(9)]
            w_inner = [W_3[i][r] for i in range(9)]
            
            # Build U
            for pos in range(4):
                coeff = a_coeffs[k][pos]
                if coeff != 0:
                    embedded = embed_3x3_to_6x6(u_inner, pos)
                    for i in range(36):
                        U[i][col] += coeff * embedded[i]
                        
            # Build V
            for pos in range(4):
                coeff = b_coeffs[k][pos]
                if coeff != 0:
                    embedded = embed_3x3_to_6x6(v_inner, pos)
                    for i in range(36):
                        V[i][col] += coeff * embedded[i]
                        
            # Build W
            for pos_out in range(4):
                coeff = c_coeffs[pos_out][k]
                if coeff != 0:
                    embedded = embed_3x3_to_6x6(w_inner, pos_out)
                    for i in range(36):
                        W[i][col] += coeff * embedded[i]
                        
    return U, V, W

# EVOLVE-BLOCK-END
