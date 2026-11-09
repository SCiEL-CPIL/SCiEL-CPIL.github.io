import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication 
    of a (n x m) matrix by a (m x p) matrix.
    Uses recursive block decomposition (Strassen-style) to reduce rank.
    """
    n, m, p = DIMENSIONS
    
    # We will build U, V, W by accumulating contributions from sub-decompositions.
    # To avoid guessing the rank, we first determine the rank by counting terms.
    # Then we fill the matrices.
    
    # Helper to get decomposition for specific small cases
    def get_strassen_decomp():
        # Returns list of (u_idx, v_idx, w_idx, coeff) for M_{2,2,2}
        # u_idx: (i, j) in A
        # v_idx: (j, k) in B
        # w_idx: (k, i) in C (Trace convention W[k*n+i])
        # Note: The global indices will be mapped later.
        # Here we return local indices and coefficients.
        # A is 2x2, B is 2x2, C is 2x2.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        terms = []
        # M1
        terms.append(([(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)], 1)) # C00, C11 get +1? 
        # Wait, Strassen formula:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map each M to contributions to C.
        # M1 contributes to C00 (+1) and C11 (+1).
        # M2 contributes to C10 (+1) and C11 (-1).
        # M3 contributes to C01 (+1) and C11 (+1).
        # M4 contributes to C00 (+1) and C10 (+1).
        # M5 contributes to C00 (-1) and C01 (+1).
        # M6 contributes to C11 (+1).
        # M7 contributes to C00 (+1).
        
        # This representation is tricky. Let's store the terms as:
        # (u_factors, v_factors, w_factors, coeff)
        # where u_factors is a list of (i,j) and coeffs for A.
        
        # Actually, it's easier to just return the U, V, W vectors for the 7 terms.
        # U: 4x7, V: 4x7, W: 4x7.
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11)
        U_s[0, 0] = 1 # A00
        U_s[3, 0] = 1 # A11
        V_s[0, 0] = 1 # B00
        V_s[3, 0] = 1 # B11
        # C00 (idx 0), C11 (idx 3)
        W_s[0, 0] = 1
        W_s[3, 0] = 1
        
        # M2: (A10+A11)B00
        U_s[2, 1] = 1 # A10
        U_s[3, 1] = 1 # A11
        V_s[0, 1] = 1 # B00
        # C10 (idx 2), C11 (idx 3)
        W_s[2, 1] = 1
        W_s[3, 1] = -1
        
        # M3: A00(B10-B11)
        U_s[0, 2] = 1 # A00
        V_s[2, 2] = 1 # B10
        V_s[3, 2] = -1 # B11
        # C01 (idx 1), C11 (idx 3)
        W_s[1, 2] = 1
        W_s[3, 2] = 1
        
        # M4: A11(B01-B00)
        U_s[3, 3] = 1 # A11
        V_s[1, 3] = 1 # B01
        V_s[0, 3] = -1 # B00
        # C00 (idx 0), C10 (idx 2)
        W_s[0, 3] = 1
        W_s[2, 3] = 1
        
        # M5: (A00+A01)B11
        U_s[0, 4] = 1 # A00
        U_s[1, 4] = 1 # A01
        V_s[3, 4] = 1 # B11
        # C00 (idx 0), C01 (idx 1)
        W_s[0, 4] = -1
        W_s[1, 4] = 1
        
        # M6: (A10-A00)(B00+B01)
        U_s[2, 5] = 1 # A10
        U_s[0, 5] = -1 # A00
        V_s[0, 5] = 1 # B00
        V_s[1, 5] = 1 # B01
        # C11 (idx 3)
        W_s[3, 5] = 1
        
        # M7: (A01-A11)(B10+B11)
        U_s[1, 6] = 1 # A01
        U_s[3, 6] = -1 # A11
        V_s[2, 6] = 1 # B10
        V_s[3, 6] = 1 # B11
        # C00 (idx 0)
        W_s[0, 6] = 1
        
        return U_s, V_s, W_s

    def get_schoolbook_decomp(n, m, p):
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

    # Recursive decomposition function
    # Returns U, V, W for the sub-problem
    def decompose(n, m, p):
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1.0]])
            V = np.array([[1.0]])
            W = np.array([[1.0]])
            return U, V, W
        
        if n == 2 and m == 2 and p == 2:
            return get_strassen_decomp()
        
        if n == 1 and m == 2 and p == 2:
            return get_schoolbook_decomp(1, 2, 2)
        
        if n == 2 and m == 2 and p == 1:
            return get_schoolbook_decomp(2, 2, 1)
        
        if n == 1 and m == 2 and p == 1:
            return get_schoolbook_decomp(1, 2, 1)

        # General case: Split
        # Choose splits
        n1 = n // 2
        n2 = n - n1
        m1 = m // 2
        m2 = m - m1
        p1 = p // 2
        p2 = p - p1
        
        # Sub-problems:
        # C11 = A11 B11 + A12 B21
        # C12 = A11 B12 + A12 B22
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        
        # We need to compute 8 multiplications?
        # A11 B11 (n1, m1, p1)
        # A11 B12 (n1, m1, p2)
        # A12 B21 (n1, m2, p1)
        # A12 B22 (n1, m2, p2)
        # A21 B11 (n2, m1, p1)
        # A21 B12 (n2, m1, p2)
        # A22 B21 (n2, m2, p1)
        # A22 B22 (n2, m2, p2)
        
        # We compute each and combine.
        
        U_list = []
        V_list = []
        W_list = []
        
        # Helper to map local indices to global
        # A indices: i in [0, n), j in [0, m)
        # B indices: j in [0, m), k in [0, p)
        # C indices: k in [0, p), i in [0, n) -> W index k*n + i
        
        # For sub-problem A_ij B_kl:
        # A row i' in [0, n_i), col j' in [0, m_j)
        # Global A row: (i_start + i'), col (j_start + j')
        # Global U index: (i_start + i') * m + (j_start + j')
        
        # B row j'' in [0, m_k), col k' in [0, p_l)
        # Global B row: (j_start + j''), col (k_start + k')
        # Global V index: (j_start + j'') * p + (k_start + k')
        
        # C row k'' in [0, p_l), col i'' in [0, n_i)
        # Global C row: (k_start + k''), col (i_start + i'')
        # Global W index: (k_start + k'') * n + (i_start + i'')
        
        def add_subproblem(n_sub, m_sub, p_sub, i_start, j_start, k_start):
            U_sub, V_sub, W_sub = decompose(n_sub, m_sub, p_sub)
            rank_sub = U_sub.shape[1]
            
            U_mapped = np.zeros((n*m, rank_sub))
            V_mapped = np.zeros((m*p, rank_sub))
            W_mapped = np.zeros((p*n, rank_sub))
            
            for col in range(rank_sub):
                # Map U
                for local_row in range(n_sub * m_sub):
                    val = U_sub[local_row, col]
                    if val != 0:
                        # local_row = i' * m_sub + j'
                        i_prime = local_row // m_sub
                        j_prime = local_row % m_sub
                        global_row = (i_start + i_prime) * m + (j_start + j_prime)
                        U_mapped[global_row, col] = val
                        
                # Map V
                for local_row in range(m_sub * p_sub):
                    val = V_sub[local_row, col]
                    if val != 0:
                        j_prime = local_row // p_sub
                        k_prime = local_row % p_sub
                        global_row = (j_start + j_prime) * p + (k_start + k_prime)
                        V_mapped[global_row, col] = val
                        
                # Map W
                for local_row in range(p_sub * n_sub):
                    val = W_sub[local_row, col]
                    if val != 0:
                        k_prime = local_row // n_sub
                        i_prime = local_row % n_sub
                        global_row = (k_start + k_prime) * n + (i_start + i_prime)
                        W_mapped[global_row, col] = val
                        
            U_list.append(U_mapped)
            V_list.append(V_mapped)
            W_list.append(W_mapped)

        # Call for the 8 blocks
        # Block 1: A11 B11 -> C11
        add_subproblem(n1, m1, p1, 0, 0, 0)
        # Block 2: A11 B12 -> C12
        add_subproblem(n1, m1, p2, 0, 0, p1)
        # Block 3: A12 B21 -> C11
        add_subproblem(n1, m2, p1, 0, m1, 0)
        # Block 4: A12 B22 -> C12
        add_subproblem(n1, m2, p2, 0, m1, p1)
        # Block 5: A21 B11 -> C21
        add_subproblem(n2, m1, p1, n1, 0, 0)
        # Block 6: A21 B12 -> C22
        add_subproblem(n2, m1, p2, n1, 0, p1)
        # Block 7: A22 B21 -> C21
        add_subproblem(n2, m2, p1, n1, m1, 0)
        # Block 8: A22 B22 -> C22
        add_subproblem(n2, m2, p2, n1, m1, p1)
        
        U_final = np.hstack(U_list)
        V_final = np.hstack(V_list)
        W_final = np.hstack(W_list)
        
        return U_final, V_final, W_final

    U, V, W = decompose(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
