import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 78
    # Structure:
    # A (3x4) split into A1 (2x4) and A2 (1x4)
    # B (4x7) is used as is.
    
    # We need to build U, V, W matrices of size (n*m, R), (m*p, R), (p*n, R)
    # U corresponds to A indices (i, j) where i in [0, n), j in [0, m)
    # V corresponds to B indices (j, k) where j in [0, m), k in [0, p)
    # W corresponds to C indices (k, i) where k in [0, p), i in [0, n)
    # Note: The task says W[k*n+i, r]. This is C[k, i] in standard math notation C = A @ B?
    # Standard: C[i, k] = sum_j A[i, j] B[j, k].
    # Task convention: W[k*n+i, r]. So W index is k*n + i. This matches C[k, i] if we view C as p x n.
    # Usually C is n x p. If C is n x p, index is i*p + k.
    # If W index is k*n + i, then the first index is k (0..p-1) and second is i (0..n-1).
    # This implies the output tensor is indexed by (k, i).
    # Let's check the tensor definition in the prompt:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # And the decomposition T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r] is NOT explicitly stated in the prompt text 
    # but implied by the shapes and standard tensor decomposition.
    # However, the prompt says: "T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]" in the comments of the seed program.
    # Wait, the seed program comment says:
    # "U: nm x R, V: mp x R, W: pn x R"
    # "T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]"
    # Here i is in [0, n*m), j in [0, m*p), k in [0, p*n).
    # But standard matrix multiplication tensor is T_{i, j, k} where i=row of A, j=col of A/row of B, k=col of B.
    # The seed program uses:
    # U index: i*m + j (flattened A)
    # V index: j*p + k (flattened B)
    # W index: k*n + i (flattened C? No, k is col of B, i is row of A. C[i, k]. Index k*n + i means C is stored column-major? Or C[k, i]?)
    # If C is n x p, standard flatten is i*p + k.
    # Here it is k*n + i. This suggests the output is treated as a p x n matrix, or the indices are swapped.
    # Let's assume the seed program's index mapping is correct for the evaluator.
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # We will build the solution by concatenating the sub-decompositions.
    
    R_total = 78
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    col_offset = 0
    
    # Helper to add a decomposition for a sub-problem
    # sub_n, sub_m, sub_p
    # offset_i, offset_j, offset_k (start indices in the global flattened arrays)
    # U_sub, V_sub, W_sub (local factor matrices)
    def add_subproblem(sub_n, sub_m, sub_p, offset_i, offset_j, offset_k, U_sub, V_sub, W_sub):
        nonlocal col_offset
        # Copy U_sub to U
        # U_sub is (sub_n * sub_m, sub_rank)
        # Global U indices: (offset_i + i_local) * m + (offset_j + j_local) ?
        # No, the global U index is i*m + j.
        # If we split rows of A (n), the columns j are the same.
        # If we split columns of A (m), the rows i are the same.
        # If we split columns of B (p), the rows j of A are the same.
        
        # Let's handle the specific splits we are doing.
        # Split 1: n = 2 + 1.
        # Block 1: n=2, m=4, p=7. i in [0, 2), j in [0, 4), k in [0, 7).
        # Block 2: n=1, m=4, p=7. i in [2, 3), j in [0, 4), k in [0, 7).
        
        # Inside Block 1 (n=2, m=4, p=7):
        # Split m = 2 + 2.
        # Block 1a: n=2, m=2, p=7. j in [0, 2).
        # Block 1b: n=2, m=2, p=7. j in [2, 4).
        
        # Inside Block 1a (n=2, m=2, p=7):
        # Split p = 2 + 2 + 2 + 1.
        # Block 1a1: n=2, m=2, p=2. k in [0, 2).
        # Block 1a2: n=2, m=2, p=2. k in [2, 4).
        # Block 1a3: n=2, m=2, p=2. k in [4, 6).
        # Block 1a4: n=2, m=2, p=1. k in [6, 7).
        
        # We will generate the factors for the atomic blocks (2x2x2 Strassen, 2x2x1 Schoolbook, 1x2x2 Schoolbook, 1x2x1 Schoolbook)
        # and then map them to the global indices.
        
        # For simplicity, we will construct the atomic blocks and then place them.
        
        # Atomic Block Types:
        # Type S: 2x2x2 Strassen (Rank 7)
        # Type A: 2x2x1 Schoolbook (Rank 4)
        # Type B: 1x2x2 Schoolbook (Rank 4)
        # Type C: 1x2x1 Schoolbook (Rank 2)
        
        # We need a function to get the factors for these atomic blocks.
        
        def get_strassen_222():
            # U (4x7), V (4x7), W (4x7)
            # Standard Strassen:
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B12 - B22)
            # M4 = A22(B21 - B11)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            
            # U factors (for A):
            # M1: A11, A22
            # M2: A21, A22
            # M3: A11
            # M4: A22
            # M5: A11, A12
            # M6: A21, A11
            # M7: A12, A22
            
            # Indices in 2x2 A:
            # 0: 11, 1: 12, 2: 21, 3: 22
            
            U_s = np.zeros((4, 7))
            V_s = np.zeros((4, 7))
            W_s = np.zeros((4, 7))
            
            # M1
            U_s[0, 0] = 1; U_s[3, 0] = 1
            V_s[0, 0] = 1; V_s[3, 0] = 1
            W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1 # C11 = M1-M4+M5-M7+M3? No.
            
            # Let's use the standard formulas for C entries in terms of M's.
            # C11 = M1 - M4 + M5 - M7 + M3
            # C12 = M4 + M7
            # C21 = M2 + M5
            # C22 = M1 - M2 + M3 + M6
            
            # W matrix columns correspond to M1..M7.
            # W[0, r] is coeff for C11.
            # W[1, r] is coeff for C12.
            # W[2, r] is coeff for C21.
            # W[3, r] is coeff for C22.
            
            # M1
            W_s[0, 0] = 1
            # M2
            W_s[2, 1] = 1
            # M3
            W_s[0, 2] = 1
            # M4
            W_s[0, 3] = -1
            W_s[1, 3] = 1
            # M5
            W_s[0, 4] = 1
            W_s[2, 4] = 1
            # M6
            W_s[3, 5] = 1
            # M7
            W_s[0, 6] = -1
            W_s[1, 6] = 1
            
            # Now U and V
            # M1 = (A11+A22)(B11+B22)
            U_s[0, 0] = 1; U_s[3, 0] = 1
            V_s[0, 0] = 1; V_s[3, 0] = 1
            
            # M2 = (A21+A22)B11
            U_s[2, 1] = 1; U_s[3, 1] = 1
            V_s[0, 1] = 1
            
            # M3 = A11(B12-B22)
            U_s[0, 2] = 1
            V_s[1, 2] = 1; V_s[3, 2] = -1
            
            # M4 = A22(B21-B11)
            U_s[3, 3] = 1
            V_s[2, 3] = 1; V_s[0, 3] = -1
            
            # M5 = (A11+A12)B22
            U_s[0, 4] = 1; U_s[1, 4] = 1
            V_s[3, 4] = 1
            
            # M6 = (A21-A11)(B11+B12)
            U_s[2, 5] = 1; U_s[0, 5] = -1
            V_s[0, 5] = 1; V_s[1, 5] = 1
            
            # M7 = (A12-A22)(B21+B22)
            U_s[1, 6] = 1; U_s[3, 6] = -1
            V_s[2, 6] = 1; V_s[3, 6] = 1
            
            return U_s, V_s, W_s

        def get_schoolbook_221():
            # 2x2x1. Rank 4.
            # C11 = A11 B11
            # C21 = A21 B11
            # C12 = A12 B21 ? No.
            # A is 2x2, B is 2x1. C is 2x1.
            # C1 = A11 B1 + A12 B2
            # C2 = A21 B1 + A22 B2
            # Indices:
            # A: 0:11, 1:12, 2:21, 3:22
            # B: 0:1, 1:2 (since p=1, B is 2x1? No, B is m x p = 2x1. So B has 2 rows, 1 col. Index 0 and 1.)
            # C: 0:1, 1:2 (C is 2x1. Index 0 and 1.)
            
            U_a = np.zeros((4, 4))
            V_a = np.zeros((2, 4))
            W_a = np.zeros((2, 4))
            
            # Term 1: A11 B1 -> C1
            U_a[0, 0] = 1
            V_a[0, 0] = 1
            W_a[0, 0] = 1
            
            # Term 2: A12 B2 -> C1
            U_a[1, 1] = 1
            V_a[1, 1] = 1
            W_a[0, 1] = 1
            
            # Term 3: A21 B1 -> C2
            U_a[2, 2] = 1
            V_a[0, 2] = 1
            W_a[1, 2] = 1
            
            # Term 4: A22 B2 -> C2
            U_a[3, 3] = 1
            V_a[1, 3] = 1
            W_a[1, 3] = 1
            
            return U_a, V_a, W_a

        def get_schoolbook_122():
            # 1x2x2. Rank 4.
            # A is 1x2. B is 2x2. C is 1x2.
            # C1 = A1 B11 + A2 B21
            # C2 = A1 B12 + A2 B22
            # A indices: 0:1, 1:2
            # B indices: 0:11, 1:12, 2:21, 3:22
            # C indices: 0:1, 1:2
            
            U_b = np.zeros((2, 4))
            V_b = np.zeros((4, 4))
            W_b = np.zeros((2, 4))
            
            # Term 1: A1 B11 -> C1
            U_b[0, 0] = 1
            V_b[0, 0] = 1
            W_b[0, 0] = 1
            
            # Term 2: A2 B21 -> C1
            U_b[1, 1] = 1
            V_b[2, 1] = 1
            W_b[0, 1] = 1
            
            # Term 3: A1 B12 -> C2
            U_b[0, 2] = 1
            V_b[1, 2] = 1
            W_b[1, 2] = 1
            
            # Term 4: A2 B22 -> C2
            U_b[1, 3] = 1
            V_b[3, 3] = 1
            W_b[1, 3] = 1
            
            return U_b, V_b, W_b

        def get_schoolbook_121():
            # 1x2x1. Rank 2.
            # A is 1x2. B is 2x1. C is 1x1.
            # C1 = A1 B1 + A2 B2
            U_c = np.zeros((2, 2))
            V_c = np.zeros((2, 2))
            W_c = np.zeros((1, 2))
            
            U_c[0, 0] = 1
            V_c[0, 0] = 1
            W_c[0, 0] = 1
            
            U_c[1, 1] = 1
            V_c[1, 1] = 1
            W_c[0, 1] = 1
            
            return U_c, V_c, W_c

        # Now we map these to the global indices.
        # We need to define the mapping for each block.
        
        # Block 1a1: n=2, m=2, p=2.
        # Global offsets:
        # i offset: 0 (rows 0,1 of A)
        # j offset: 0 (cols 0,1 of A)
        # k offset: 0 (cols 0,1 of B)
        
        def place_block(U_loc, V_loc, W_loc, i_off, j_off, k_off):
            nonlocal col_offset
            R_loc = U_loc.shape[1]
            
            for r in range(R_loc):
                # U
                for i_l in range(U_loc.shape[0]):
                    if U_loc[i_l, r] != 0:
                        # Local index i_l corresponds to (i_l // m_sub, i_l % m_sub)?
                        # No, U_loc is flattened A_sub.
                        # We need to know the shape of A_sub to map i_l to global i, j.
                        # This is getting complicated with generic offsets.
                        # Let's hardcode the placements.
                        pass

        # Hardcoded Placement Strategy:
        
        # 1. Block 1a1: Strassen 2x2x2. i:[0,2), j:[0,2), k:[0,2)
        U_s, V_s, W_s = get_strassen_222()
        for r in range(7):
            for i_l in range(4):
                if U_s[i_l, r] != 0:
                    i_g = i_l // 2
                    j_g = i_l % 2
                    U[i_g * 4 + j_g, col_offset] = U_s[i_l, r]
            for j_l in range(4):
                if V_s[j_l, r] != 0:
                    j_g = j_l // 2
                    k_g = j_l % 2
                    V[j_g * 7 + k_g, col_offset] = V_s[j_l, r]
            for k_l in range(4):
                if W_s[k_l, r] != 0:
                    k_g = k_l // 2
                    i_g = k_l % 2
                    W[k_g * 3 + i_g, col_offset] = W_s[k_l, r]
            col_offset += 1

        # 2. Block 1a2: Strassen 2x2x2. i:[0,2), j:[0,2), k:[2,4)
        U_s, V_s, W_s = get_strassen_222()
        for r in range(7):
            for i_l in range(4):
                if U_s[i_l, r] != 0:
                    i_g = i_l // 2
                    j_g = i_l % 2
                    U[i_g * 4 + j_g, col_offset] = U_s[i_l, r]
            for j_l in range(4):
                if V_s[j_l, r] != 0:
                    j_g = j_l // 2
                    k_g = 2 + (j_l % 2)
                    V[j_g * 7 + k_g, col_offset] = V_s[j_l, r]
            for k_l in range(4):
                if W_s[k_l, r] != 0:
                    k_g = 2 + (k_l // 2)
                    i_g = k_l % 2
                    W[k_g * 3 + i_g, col_offset] = W_s[k_l, r]
            col_offset += 1

        # 3. Block 1a3: Strassen 2x2x2. i:[0,2), j:[0,2), k:[4,6)
        U_s, V_s, W_s = get_strassen_222()
        for r in range(7):
            for i_l in range(4):
                if U_s[i_l, r] != 0:
                    i_g = i_l // 2
                    j_g = i_l % 2
                    U[i_g * 4 + j_g, col_offset] = U_s[i_l, r]
            for j_l in range(4):
                if V_s[j_l, r] != 0:
                    j_g = j_l // 2
                    k_g = 4 + (j_l % 2)
                    V[j_g * 7 + k_g, col_offset] = V_s[j_l, r]
            for k_l in range(4):
                if W_s[k_l, r] != 0:
                    k_g = 4 + (k_l // 2)
                    i_g = k_l % 2
                    W[k_g * 3 + i_g, col_offset] = W_s[k_l, r]
            col_offset += 1

        # 4. Block 1a4: Schoolbook 2x2x1. i:[0,2), j:[0,2), k:[6,7)
        U_a, V_a, W_a = get_schoolbook_221()
        for r in range(4):
            for i_l in range(4):
                if U_a[i_l, r] != 0:
                    i_g = i_l // 2
                    j_g = i_l % 2
                    U[i_g * 4 + j_g, col_offset] = U_a[i_l, r]
            for j_l in range(2):
                if V_a[j_l, r] != 0:
                    j_g = j_l
                    k_g = 6
                    V[j_g * 7 + k_g, col_offset] = V_a[j_l, r]
            for k_l in range(2):
                if W_a[k_l, r] != 0:
                    k_g = 6
                    i_g = k_l
                    W[k_g * 3 + i_g, col_offset] = W_a[k_l, r]
            col_offset += 1

        # 5. Block 1b1: Strassen 2x2x2. i:[0,2), j:[2,4), k:[0,2)
        U_s, V_s, W_s = get_strassen_222()
        for r in range(7):
            for i_l in range(4):
                if U_s[i_l, r] != 0:
                    i_g = i_l // 2
                    j_g = 2 + (i_l % 2)
                    U[i_g * 4 + j_g, col_offset] = U_s[i_l, r]
            for j_l in range(4):
                if V_s[j_l, r] != 0:
                    j_g = 2 + (j_l // 2)
                    k_g = j_l % 2
                    V[j_g * 7 + k_g, col_offset] = V_s[j_l, r]
            for k_l in range(4):
                if W_s[k_l, r] != 0:
                    k_g = k_l // 2
                    i_g = k_l % 2
                    W[k_g * 3 + i_g, col_offset] = W_s[k_l, r]
            col_offset += 1

        # 6. Block 1b2: Strassen 2x2x2. i:[0,2), j:[2,4), k:[2,4)
        U_s, V_s, W_s = get_strassen_222()
        for r in range(7):
            for i_l in range(4):
                if U_s[i_l, r] != 0:
                    i_g = i_l // 2
                    j_g = 2 + (i_l % 2)
                    U[i_g * 4 + j_g, col_offset] = U_s[i_l, r]
            for j_l in range(4):
                if V_s[j_l, r] != 0:
                    j_g = 2 + (j_l // 2)
                    k_g = 2 + (j_l % 2)
                    V[j_g * 7 + k_g, col_offset] = V_s[j_l, r]
            for k_l in range(4):
                if W_s[k_l, r] != 0:
                    k_g = 2 + (k_l // 2)
                    i_g = k_l % 2
                    W[k_g * 3 + i_g, col_offset] = W_s[k_l, r]
            col_offset += 1

        # 7. Block 1b3: Strassen 2x2x2. i:[0,2), j:[2,4), k:[4,6)
        U_s, V_s, W_s = get_strassen_222()
        for r in range(7):
            for i_l in range(4):
                if U_s[i_l, r] != 0:
                    i_g = i_l // 2
                    j_g = 2 + (i_l % 2)
                    U[i_g * 4 + j_g, col_offset] = U_s[i_l, r]
            for j_l in range(4):
                if V_s[j_l, r] != 0:
                    j_g = 2 + (j_l // 2)
                    k_g = 4 + (j_l % 2)
                    V[j_g * 7 + k_g, col_offset] = V_s[j_l, r]
            for k_l in range(4):
                if W_s[k_l, r] != 0:
                    k_g = 4 + (k_l // 2)
                    i_g = k_l % 2
                    W[k_g * 3 + i_g, col_offset] = W_s[k_l, r]
            col_offset += 1

        # 8. Block 1b4: Schoolbook 2x2x1. i:[0,2), j:[2,4), k:[6,7)
        U_a, V_a, W_a = get_schoolbook_221()
        for r in range(4):
            for i_l in range(4):
                if U_a[i_l, r] != 0:
                    i_g = i_l // 2
                    j_g = 2 + (i_l % 2)
                    U[i_g * 4 + j_g, col_offset] = U_a[i_l, r]
            for j_l in range(2):
                if V_a[j_l, r] != 0:
                    j_g = 2 + j_l
                    k_g = 6
                    V[j_g * 7 + k_g, col_offset] = V_a[j_l, r]
            for k_l in range(2):
                if W_a[k_l, r] != 0:
                    k_g = 6
                    i_g = k_l
                    W[k_g * 3 + i_g, col_offset] = W_a[k_l, r]
            col_offset += 1

        # Now Block 2: n=1, m=4, p=7. i:[2,3).
        # Split m=2+2.
        # Block 2a: n=1, m=2, p=7. j:[0,2).
        # Block 2b: n=1, m=2, p=7. j:[2,4).
        
        # Block 2a Split p=2+2+2+1.
        # 2a1: 1x2x2 Schoolbook. i:[2,3), j:[0,2), k:[0,2)
        U_b, V_b, W_b = get_schoolbook_122()
        for r in range(4):
            for i_l in range(2):
                if U_b[i_l, r] != 0:
                    i_g = 2
                    j_g = i_l
                    U[i_g * 4 + j_g, col_offset] = U_b[i_l, r]
            for j_l in range(4):
                if V_b[j_l, r] != 0:
                    j_g = j_l // 2
                    k_g = j_l % 2
                    V[j_g * 7 + k_g, col_offset] = V_b[j_l, r]
            for k_l in range(2):
                if W_b[k_l, r] != 0:
                    k_g = k_l
                    i_g = 2
                    W[k_g * 3 + i_g, col_offset] = W_b[k_l, r]
            col_offset += 1

        # 2a2: 1x2x2 Schoolbook. i:[2,3), j:[0,2), k:[2,4)
        U_b, V_b, W_b = get_schoolbook_122()
        for r in range(4):
            for i_l in range(2):
                if U_b[i_l, r] != 0:
                    i_g = 2
                    j_g = i_l
                    U[i_g * 4 + j_g, col_offset] = U_b[i_l, r]
            for j_l in range(4):
                if V_b[j_l, r] != 0:
                    j_g = j_l // 2
                    k_g = 2 + (j_l % 2)
                    V[j_g * 7 + k_g, col_offset] = V_b[j_l, r]
            for k_l in range(2):
                if W_b[k_l, r] != 0:
                    k_g = 2 + k_l
                    i_g = 2
                    W[k_g * 3 + i_g, col_offset] = W_b[k_l, r]
            col_offset += 1

        # 2a3: 1x2x2 Schoolbook. i:[2,3), j:[0,2), k:[4,6)
        U_b, V_b, W_b = get_schoolbook_122()
        for r in range(4):
            for i_l in range(2):
                if U_b[i_l, r] != 0:
                    i_g = 2
                    j_g = i_l
                    U[i_g * 4 + j_g, col_offset] = U_b[i_l, r]
            for j_l in range(4):
                if V_b[j_l, r] != 0:
                    j_g = j_l // 2
                    k_g = 4 + (j_l % 2)
                    V[j_g * 7 + k_g, col_offset] = V_b[j_l, r]
            for k_l in range(2):
                if W_b[k_l, r] != 0:
                    k_g = 4 + k_l
                    i_g = 2
                    W[k_g * 3 + i_g, col_offset] = W_b[k_l, r]
            col_offset += 1

        # 2a4: 1x2x1 Schoolbook. i:[2,3), j:[0,2), k:[6,7)
        U_c, V_c, W_c = get_schoolbook_121()
        for r in range(2):
            for i_l in range(2):
                if U_c[i_l, r] != 0:
                    i_g = 2
                    j_g = i_l
                    U[i_g * 4 + j_g, col_offset] = U_c[i_l, r]
            for j_l in range(2):
                if V_c[j_l, r] != 0:
                    j_g = j_l
                    k_g = 6
                    V[j_g * 7 + k_g, col_offset] = V_c[j_l, r]
            for k_l in range(1):
                if W_c[k_l, r] != 0:
                    k_g = 6
                    i_g = 2
                    W[k_g * 3 + i_g, col_offset] = W_c[k_l, r]
            col_offset += 1

        # Block 2b: n=1, m=2, p=7. j:[2,4).
        # 2b1: 1x2x2 Schoolbook. i:[2,3), j:[2,4), k:[0,2)
        U_b, V_b, W_b = get_schoolbook_122()
        for r in range(4):
            for i_l in range(2):
                if U_b[i_l, r] != 0:
                    i_g = 2
                    j_g = 2 + i_l
                    U[i_g * 4 + j_g, col_offset] = U_b[i_l, r]
            for j_l in range(4):
                if V_b[j_l, r] != 0:
                    j_g = 2 + (j_l // 2)
                    k_g = j_l % 2
                    V[j_g * 7 + k_g, col_offset] = V_b[j_l, r]
            for k_l in range(2):
                if W_b[k_l, r] != 0:
                    k_g = k_l
                    i_g = 2
                    W[k_g * 3 + i_g, col_offset] = W_b[k_l, r]
            col_offset += 1

        # 2b2: 1x2x2 Schoolbook. i:[2,3), j:[2,4), k:[2,4)
        U_b, V_b, W_b = get_schoolbook_122()
        for r in range(4):
            for i_l in range(2):
                if U_b[i_l, r] != 0:
                    i_g = 2
                    j_g = 2 + i_l
                    U[i_g * 4 + j_g, col_offset] = U_b[i_l, r]
            for j_l in range(4):
                if V_b[j_l, r] != 0:
                    j_g = 2 + (j_l // 2)
                    k_g = 2 + (j_l % 2)
                    V[j_g * 7 + k_g, col_offset] = V_b[j_l, r]
            for k_l in range(2):
                if W_b[k_l, r] != 0:
                    k_g = 2 + k_l
                    i_g = 2
                    W[k_g * 3 + i_g, col_offset] = W_b[k_l, r]
            col_offset += 1

        # 2b3: 1x2x2 Schoolbook. i:[2,3), j:[2,4), k:[4,6)
        U_b, V_b, W_b = get_schoolbook_122()
        for r in range(4):
            for i_l in range(2):
                if U_b[i_l, r] != 0:
                    i_g = 2
                    j_g = 2 + i_l
                    U[i_g * 4 + j_g, col_offset] = U_b[i_l, r]
            for j_l in range(4):
                if V_b[j_l, r] != 0:
                    j_g = 2 + (j_l // 2)
                    k_g = 4 + (j_l % 2)
                    V[j_g * 7 + k_g, col_offset] = V_b[j_l, r]
            for k_l in range(2):
                if W_b[k_l, r] != 0:
                    k_g = 4 + k_l
                    i_g = 2
                    W[k_g * 3 + i_g, col_offset] = W_b[k_l, r]
            col_offset += 1

        # 2b4: 1x2x1 Schoolbook. i:[2,3), j:[2,4), k:[6,7)
        U_c, V_c, W_c = get_schoolbook_121()
        for r in range(2):
            for i_l in range(2):
                if U_c[i_l, r] != 0:
                    i_g = 2
                    j_g = 2 + i_l
                    U[i_g * 4 + j_g, col_offset] = U_c[i_l, r]
            for j_l in range(2):
                if V_c[j_l, r] != 0:
                    j_g = 2 + j_l
                    k_g = 6
                    V[j_g * 7 + k_g, col_offset] = V_c[j_l, r]
            for k_l in range(1):
                if W_c[k_l, r] != 0:
                    k_g = 6
                    i_g = 2
                    W[k_g * 3 + i_g, col_offset] = W_c[k_l, r]
            col_offset += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

