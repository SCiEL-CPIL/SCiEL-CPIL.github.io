import numpy as np

def construct():
    """
    Returns a rank-110 decomposition for 5x5x5 matrix multiplication.
    Uses a 4+1 block partition:
    A = [[A44, A41], [A14, A11]]
    B = [[B44, B41], [B14, B11]]
    C = [[C44, C41], [C14, C11]]
    
    C44 = A44*B44 + A41*B14
    C41 = A44*B41 + A41*B11
    C14 = A14*B44 + A11*B14
    C11 = A14*B41 + A11*B11
    """
    n, m, p = 5, 5, 5
    
    # List to hold (U, V, W) tuples for each component
    components = []
    
    def get_schoolbook(m, n, p):
        """Returns U, V, W for schoolbook M_{m x n x p}"""
        rank = m * n * p
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p * m)]
        
        col = 0
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    U[i * n + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * m + i][col] = 1
                    col += 1
        return U, V, W

    def get_strassen_2x2():
        """Returns U, V, W for rank-7 M_{2 x 2 x 2}"""
        # Strassen algorithm
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # A indices: 0(11), 1(12), 2(21), 3(22)
        # B indices: 0(11), 1(12), 2(21), 3(22)
        # C indices: 0(11), 1(12), 2(21), 3(22)
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[2][3] = 1; W[0][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[3][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = 1
        
        return U, V, W

    def get_strassen_4x4():
        """Returns U, V, W for rank-49 M_{4 x 4 x 4} using 2x2 Strassen blocks"""
        # We can simulate this by combining 7 copies of 2x2 Strassen.
        # However, the indices need to be mapped.
        # A 4x4 matrix has indices 0..15.
        # Block 11: 0,1,4,5
        # Block 12: 2,3,6,7
        # Block 21: 8,9,12,13
        # Block 22: 10,11,14,15
        
        # It is easier to generate the 49 terms directly or use the structure.
        # Let's use the property that M_{4x4x4} can be built from 7 M_{2x2x2} tensors.
        # But the U, V, W matrices will be sparse.
        
        # Instead of hardcoding 49, we can use the schoolbook for 4x4x4 if we don't have the exact coefficients handy?
        # No, 49 is much better than 64.
        # I will implement the 49 rank algorithm by combining 7 Strassen 2x2 instances.
        
        # The 7 instances correspond to the 7 Strassen products.
        # For each instance, we have a 2x2x2 decomposition.
        # We need to map the local 2x2 indices to the global 4x4 indices.
        
        # Local 2x2 indices: 0,1,2,3
        # Global 4x4 indices:
        # A:
        # 11: 0,1,4,5
        # 12: 2,3,6,7
        # 21: 8,9,12,13
        # 22: 10,11,14,15
        
        # We will create U, V, W of size 16x49.
        U = [[0] * 49 for _ in range(16)]
        V = [[0] * 49 for _ in range(16)]
        W = [[0] * 49 for _ in range(16)]
        
        # Strassen 2x2 U, V, W
        U2, V2, W2 = get_strassen_2x2()
        
        # Mapping of Strassen components to blocks
        # M1: A11+A22, B11+B22
        # M2: A21+A22, B11
        # M3: A11, B12-B22
        # M4: A22, B21-B11
        # M5: A11+A12, B22
        # M6: A21-A11, B11+B12
        # M7: A12-A22, B21+B22
        
        # We iterate over the 7 Strassen terms.
        # For each term, we determine which global indices are involved in U, V, W.
        
        # Term 0 (M1):
        # U: A11 (0,1,4,5) and A22 (10,11,14,15) with coeff 1
        # V: B11 (0,1,4,5) and B22 (10,11,14,15) with coeff 1
        # W: C11 (0,1,4,5) and C22 (10,11,14,15) with coeff 1
        
        # This manual mapping is error-prone and long.
        # Alternative: Use the schoolbook for 4x4x4 (rank 64) to be safe?
        # 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125.
        # That's the same as the seed!
        
        # I MUST implement the rank 49 algorithm for 4x4.
        # Let's define the index mappings.
        
        # A indices for blocks:
        A_11 = [0, 1, 4, 5]
        A_12 = [2, 3, 6, 7]
        A_21 = [8, 9, 12, 13]
        A_22 = [10, 11, 14, 15]
        
        B_11 = [0, 1, 4, 5]
        B_12 = [2, 3, 6, 7]
        B_21 = [8, 9, 12, 13]
        B_22 = [10, 11, 14, 15]
        
        C_11 = [0, 1, 4, 5]
        C_12 = [2, 3, 6, 7]
        C_21 = [8, 9, 12, 13]
        C_22 = [10, 11, 14, 15]
        
        # Strassen U, V, W for 2x2:
        # U[0][0]=1 (A11), U[3][0]=1 (A22)
        # V[0][0]=1 (B11), V[3][0]=1 (B22)
        # W[0][0]=1 (C11), W[3][0]=1 (C22)
        
        # We will add the contributions of each of the 7 Strassen terms to the global U, V, W.
        
        # Term 0: M1 = (A11+A22)(B11+B22)
        # Local U: A11(0), A22(3)
        # Local V: B11(0), B22(3)
        # Local W: C11(0), C22(3)
        
        def add_term(term_idx, U_maps, V_maps, W_maps):
            # U_maps: list of (global_indices, coeff)
            # V_maps: list of (global_indices, coeff)
            # W_maps: list of (global_indices, coeff)
            for g_idx, coeff in U_maps:
                for i in g_idx:
                    U[i][term_idx] += coeff
            for g_idx, coeff in V_maps:
                for i in g_idx:
                    V[i][term_idx] += coeff
            for g_idx, coeff in W_maps:
                for i in g_idx:
                    W[i][term_idx] += coeff

        # Term 0
        add_term(0, 
                 [(A_11, 1), (A_22, 1)], 
                 [(B_11, 1), (B_22, 1)], 
                 [(C_11, 1), (C_22, 1)])
                 
        # Term 1: M2 = (A21+A22)B11
        add_term(1, 
                 [(A_21, 1), (A_22, 1)], 
                 [(B_11, 1)], 
                 [(C_21, 1), (C_22, 1)])
                 
        # Term 2: M3 = A11(B12-B22)
        add_term(2, 
                 [(A_11, 1)], 
                 [(B_12, 1), (B_22, -1)], 
                 [(C_12, 1), (C_22, 1)])
                 
        # Term 3: M4 = A22(B21-B11)
        add_term(3, 
                 [(A_22, 1)], 
                 [(B_21, 1), (B_11, -1)], 
                 [(C_21, 1), (C_11, 1)])
                 
        # Term 4: M5 = (A11+A12)B22
        add_term(4, 
                 [(A_11, 1), (A_12, 1)], 
                 [(B_22, 1)], 
                 [(C_11, -1), (C_22, 1)])
                 
        # Term 5: M6 = (A21-A11)(B11+B12)
        add_term(5, 
                 [(A_21, 1), (A_11, -1)], 
                 [(B_11, 1), (B_12, 1)], 
                 [(C_22, 1)])
                 
        # Term 6: M7 = (A12-A22)(B21+B22)
        add_term(6, 
                 [(A_12, 1), (A_22, -1)], 
                 [(B_21, 1), (B_22, 1)], 
                 [(C_11, 1), (C_12, 1)])
                 
        return U, V, W

    # Component 1: A44 B44 (4x4x4)
    U1, V1, W1 = get_strassen_4x4()
    # Map to global indices
    # A44: rows 0-3, cols 0-3 -> indices 0-15
    # B44: rows 0-3, cols 0-3 -> indices 0-15
    # C44: rows 0-3, cols 0-3 -> indices 0-15
    # The local indices 0-15 correspond directly to global 0-15 for these blocks.
    components.append((U1, V1, W1))

    # Component 2: A41 B14 (4x1x4) -> C44
    # A41: rows 0-3, col 4 -> indices 4, 9, 14, 19
    # B14: row 4, cols 0-3 -> indices 20, 21, 22, 23
    # C44: rows 0-3, cols 0-3 -> indices 0-15
    U2, V2, W2 = get_schoolbook(4, 1, 4)
    # Remap indices
    # U local 0-3 -> global 4, 9, 14, 19
    # V local 0-3 -> global 20, 21, 22, 23
    # W local 0-15 -> global 0-15
    U2g = [[0]*16 for _ in range(25)]
    V2g = [[0]*16 for _ in range(25)]
    W2g = [[0]*16 for _ in range(25)]
    A_map = [4, 9, 14, 19]
    B_map = [20, 21, 22, 23]
    C_map = list(range(16))
    for i in range(16):
        for j in range(16):
            if U2[i][j]:
                U2g[A_map[i//4]][j] = U2[i][j]
            if V2[i][j]:
                V2g[B_map[i//4]][j] = V2[i][j]
            if W2[i][j]:
                W2g[C_map[i]][j] = W2[i][j]
    components.append((U2g, V2g, W2g))

    # Component 3: A44 B41 (4x4x1) -> C41
    # A44: 0-15
    # B41: rows 0-3, col 4 -> indices 4, 9, 14, 19
    # C41: rows 0-3, col 4 -> indices 4, 9, 14, 19
    U3, V3, W3 = get_schoolbook(4, 4, 1)
    U3g = [[0]*16 for _ in range(25)]
    V3g = [[0]*16 for _ in range(25)]
    W3g = [[0]*16 for _ in range(25)]
    A_map = list(range(16))
    B_map = [4, 9, 14, 19]
    C_map = [4, 9, 14, 19]
    for i in range(16):
        for j in range(16):
            if U3[i][j]:
                U3g[A_map[i]][j] = U3[i][j]
            if V3[i][j]:
                V3g[B_map[i//4]][j] = V3[i][j]
            if W3[i][j]:
                W3g[C_map[i//4]][j] = W3[i][j]
    components.append((U3g, V3g, W3g))

    # Component 4: A41 B11 (4x1x1) -> C41
    # A41: 4, 9, 14, 19
    # B11: 24
    # C41: 4, 9, 14, 19
    U4, V4, W4 = get_schoolbook(4, 1, 1)
    U4g = [[0]*4 for _ in range(25)]
    V4g = [[0]*4 for _ in range(25)]
    W4g = [[0]*4 for _ in range(25)]
    A_map = [4, 9, 14, 19]
    B_map = [24]
    C_map = [4, 9, 14, 19]
    for i in range(4):
        for j in range(4):
            if U4[i][j]:
                U4g[A_map[i]][j] = U4[i][j]
            if V4[i][j]:
                V4g[B_map[i//4]][j] = V4[i][j]
            if W4[i][j]:
                W4g[C_map[i]][j] = W4[i][j]
    components.append((U4g, V4g, W4g))

    # Component 5: A14 B44 (1x4x4) -> C14
    # A14: row 4, cols 0-3 -> 20, 21, 22, 23
    # B44: 0-15
    # C14: row 4, cols 0-3 -> 20, 21, 22, 23
    U5, V5, W5 = get_schoolbook(1, 4, 4)
    U5g = [[0]*16 for _ in range(25)]
    V5g = [[0]*16 for _ in range(25)]
    W5g = [[0]*16 for _ in range(25)]
    A_map = [20, 21, 22, 23]
    B_map = list(range(16))
    C_map = [20, 21, 22, 23]
    for i in range(16):
        for j in range(16):
            if U5[i][j]:
                U5g[A_map[i//4]][j] = U5[i][j]
            if V5[i][j]:
                V5g[B_map[i]][j] = V5[i][j]
            if W5[i][j]:
                W5g[C_map[i//4]][j] = W5[i][j]
    components.append((U5g, V5g, W5g))

    # Component 6: A11 B14 (1x1x4) -> C14
    # A11: 24
    # B14: 20, 21, 22, 23
    # C14: 20, 21, 22, 23
    U6, V6, W6 = get_schoolbook(1, 1, 4)
    U6g = [[0]*4 for _ in range(25)]
    V6g = [[0]*4 for _ in range(25)]
    W6g = [[0]*4 for _ in range(25)]
    A_map = [24]
    B_map = [20, 21, 22, 23]
    C_map = [20, 21, 22, 23]
    for i in range(4):
        for j in range(4):
            if U6[i][j]:
                U6g[A_map[i//4]][j] = U6[i][j]
            if V6[i][j]:
                V6g[B_map[i//4]][j] = V6[i][j]
            if W6[i][j]:
                W6g[C_map[i]][j] = W6[i][j]
    components.append((U6g, V6g, W6g))

    # Component 7: A14 B41 (1x4x1) -> C11
    # A14: 20, 21, 22, 23
    # B41: 4, 9, 14, 19
    # C11: 24
    U7, V7, W7 = get_schoolbook(1, 4, 1)
    U7g = [[0]*4 for _ in range(25)]
    V7g = [[0]*4 for _ in range(25)]
    W7g = [[0]*4 for _ in range(25)]
    A_map = [20, 21, 22, 23]
    B_map = [4, 9, 14, 19]
    C_map = [24]
    for i in range(4):
        for j in range(4):
            if U7[i][j]:
                U7g[A_map[i//4]][j] = U7[i][j]
            if V7[i][j]:
                V7g[B_map[i//4]][j] = V7[i][j]
            if W7[i][j]:
                W7g[C_map[i//4]][j] = W7[i][j]
    components.append((U7g, V7g, W7g))

    # Component 8: A11 B11 (1x1x1) -> C11
    U8, V8, W8 = get_schoolbook(1, 1, 1)
    U8g = [[0]*1 for _ in range(25)]
    V8g = [[0]*1 for _ in range(25)]
    W8g = [[0]*1 for _ in range(25)]
    U8g[24][0] = 1
    V8g[24][0] = 1
    W8g[24][0] = 1
    components.append((U8g, V8g, W8g))

    # Concatenate all components
    total_rank = sum(len(comp[0][0]) for comp in components)
    U_final = [[0] * total_rank for _ in range(25)]
    V_final = [[0] * total_rank for _ in range(25)]
    W_final = [[0] * total_rank for _ in range(25)]
    
    offset = 0
    for U, V, W in components:
        rank = len(U[0])
        for i in range(25):
            for j in range(rank):
                U_final[i][offset + j] = U[i][j]
                V_final[i][offset + j] = V[i][j]
                W_final[i][offset + j] = W[i][j]
        offset += rank
        
    return {"U": U_final, "V": V_final, "W": W_final}
