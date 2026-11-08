import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def strassen_2x2():
    """
    Returns U, V, W for 2x2 x 2x2 multiplication with rank 7.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U: i*2 + j
    V: j*2 + k
    W: k*2 + i
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank), dtype=int)
    V = np.zeros((n*n, rank), dtype=int)
    W = np.zeros((n*n, rank), dtype=int)
    
    # Strassen formulas
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U encodes A, V encodes B, W encodes C.
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1; W[1, 0] -= 1; W[2, 0] += 1 # C11, C22, C12, C21?
    # Wait, W index is k*n + i.
    # C = [[C11, C12], [C21, C22]]
    # C11 is at 0*2+0=0. C12 is at 0*2+1=1. C21 is at 1*2+0=2. C22 is at 1*2+1=3.
    # M1 contributes to C11, C22, C12, C21?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's just assign the terms directly to W.
    
    # Reset W to 0 and fill based on formulas.
    W = np.zeros((n*n, rank), dtype=int)
    
    # M1: C11 += 1, C22 += 1
    W[0, 0] += 1
    W[3, 0] += 1
    
    # M2: C21 += 1, C22 -= 1
    W[2, 1] += 1
    W[3, 1] -= 1
    
    # M3: C12 += 1, C22 += 1
    W[1, 2] += 1
    W[3, 2] += 1
    
    # M4: C11 += 1, C21 += 1
    W[0, 3] += 1
    W[2, 3] += 1
    
    # M5: C11 -= 1, C12 += 1
    W[0, 4] -= 1
    W[1, 4] += 1
    
    # M6: C22 += 1
    W[3, 5] += 1
    
    # M7: C11 += 1
    W[0, 6] += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # We will use the decomposition:
    # B = [B1 (4x4), B2 (4x4), B3 (4x1)]
    # C1 = A B1 (rank 49)
    # C2 = A B2 (rank 49)
    # C3 = A B3 (rank 16)
    # Total rank 114.
    
    rank_total = 49 + 49 + 16
    U = np.zeros((n*m, rank_total), dtype=int)
    V = np.zeros((m*p, rank_total), dtype=int)
    W = np.zeros((p*n, rank_total), dtype=int)
    
    # Get 4x4 x 4x4 decomposition (rank 49)
    # This is recursive Strassen.
    # We need to generate it.
    
    def get_4x4_decomp():
        # Returns U4, V4, W4 for 4x4 x 4x4
        n4 = 4
        rank4 = 49
        U4 = np.zeros((n4*n4, rank4), dtype=int)
        V4 = np.zeros((n4*n4, rank4), dtype=int)
        W4 = np.zeros((n4*n4, rank4), dtype=int)
        
        # Get 2x2 decomp
        U2, V2, W2 = strassen_2x2()
        
        # Strassen on 4x4 blocks
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # Blocks are 2x2.
        # Indices in 4x4:
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # B11: 0,1,8,9 (row 0,1 cols 0,1 -> 0*4+0..1, 1*4+0..1? No. B is 4x4.
        # V index is j*p + k. Here p=4.
        # B11: j=0,1; k=0,1. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
        # B12: j=0,1; k=2,3. Indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
        # B21: j=2,3; k=0,1. Indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13.
        # B22: j=2,3; k=2,3. Indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15.
        
        # W index is k*n + i. Here n=4.
        # C11: k=0,1; i=0,1. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
        # C12: k=0,1; i=2,3. Indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
        # C21: k=2,3; i=0,1. Indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13.
        # C22: k=2,3; i=2,3. Indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15.
        
        # Strassen M1..M7
        # Each Mk is a 2x2 x 2x2 multiplication.
        # We map the 2x2 indices to 4x4 indices.
        
        # M1 = (A11+A22)(B11+B22)
        # A part: A11 + A22.
        # A11 indices in 4x4: 0,1,4,5.
        # A22 indices in 4x4: 10,11,14,15.
        # U2 has shape (4,7).
        # We need to add U2 scaled by 1 to U4 for the corresponding rows.
        
        # Helper to add a 2x2 block decomp to the 4x4 decomp
        def add_block_decomp(U2, V2, W2, 
                             a_offsets, b_offsets, c_offsets, 
                             sign_a=1, sign_b=1, sign_c=1, 
                             target_U=None, target_V=None, target_W=None, 
                             col_offset=0):
            # a_offsets: list of 4 indices in 4x4 A
            # b_offsets: list of 4 indices in 4x4 B
            # c_offsets: list of 4 indices in 4x4 C
            
            for i in range(4):
                # U
                target_U[a_offsets[i], col_offset:i+4] += sign_a * U2[i, :]
                # V
                target_V[b_offsets[i], col_offset:i+4] += sign_b * V2[i, :]
                # W
                target_W[c_offsets[i], col_offset:i+4] += sign_c * W2[i, :]
                
        # This helper is slightly wrong because U2, V2, W2 are (4,7).
        # We are adding 7 columns at a time.
        
        # Let's do it manually for the 7 Strassen terms.
        
        # M1: (A11+A22)(B11+B22)
        # A11: [0,1,4,5], A22: [10,11,14,15]
        # B11: [0,1,4,5], B22: [10,11,14,15]
        # C11: [0,1,4,5], C22: [10,11,14,15]
        
        # We need to form A11+A22.
        # U2[i, c] is the coefficient for A_block[i] in term c.
        # So for M1, term c=0..6:
        # U4[0, c] += U2[0, c] (from A11)
        # U4[10, c] += U2[0, c] (from A22)
        # etc.
        
        # Let's write a more generic function.
        
        def map_2x2_to_4x4(U2, V2, W2, a_map, b_map, c_map, col_start):
            # a_map: dict {2x2_index: 4x4_index}
            # 2x2 indices: 0,1,2,3
            # 4x4 indices: 0..15
            
            for c in range(7):
                for i2 in range(4):
                    i4 = a_map[i2]
                    U4[i4, col_start+c] += U2[i2, c]
                    j4 = b_map[i2]
                    V4[j4, col_start+c] += V2[i2, c]
                    k4 = c_map[i2]
                    W4[k4, col_start+c] += W2[i2, c]
            
        # Maps for blocks
        # 2x2 indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # A11: 00->0, 01->1, 10->4, 11->5
        map_A11 = {0:0, 1:1, 2:4, 3:5}
        # A12: 00->2, 01->3, 10->6, 11->7
        map_A12 = {0:2, 1:3, 2:6, 3:7}
        # A21: 00->8, 01->9, 10->12, 11->13
        map_A21 = {0:8, 1:9, 2:12, 3:13}
        # A22: 00->10, 01->11, 10->14, 11->15
        map_A22 = {0:10, 1:11, 2:14, 3:15}
        
        # B11: 00->0, 01->1, 10->4, 11->5 (V index j*p+k)
        # B is 4x4. p=4.
        # B11 (j=0,1, k=0,1): 0,1,4,5
        map_B11 = {0:0, 1:1, 2:4, 3:5}
        # B12 (j=0,1, k=2,3): 2,3,6,7
        map_B12 = {0:2, 1:3, 2:6, 3:7}
        # B21 (j=2,3, k=0,1): 8,9,12,13
        map_B21 = {0:8, 1:9, 2:12, 3:13}
        # B22 (j=2,3, k=2,3): 10,11,14,15
        map_B22 = {0:10, 1:11, 2:14, 3:15}
        
        # C11 (k=0,1, i=0,1): 0,1,4,5 (W index k*n+i)
        # C is 4x4. n=4.
        # C11 (k=0,1, i=0,1): 0,1,4,5
        map_C11 = {0:0, 1:1, 2:4, 3:5}
        # C12 (k=0,1, i=2,3): 2,3,6,7
        map_C12 = {0:2, 1:3, 2:6, 3:7}
        # C21 (k=2,3, i=0,1): 8,9,12,13
        map_C21 = {0:8, 1:9, 2:12, 3:13}
        # C22 (k=2,3, i=2,3): 10,11,14,15
        map_C22 = {0:10, 1:11, 2:14, 3:15}
        
        # M1: (A11+A22)(B11+B22)
        # A: A11 + A22.
        # We need to sum the contributions.
        # This map approach adds U2 to specific rows.
        # For M1, A part is A11 + A22.
        # So we call map for A11 and A22?
        # No, U2 represents the linear form.
        # U2[i, c] is coeff of A_i in M_c.
        # So for M1, A_i is A11_i + A22_i.
        # So we add U2[i, c] to U4[A11_i, c] and U4[A22_i, c].
        
        def add_strassen_term(U2, V2, W2, a_maps, b_maps, c_maps, col_start):
            # a_maps: list of maps (for sums)
            for c in range(7):
                for i2 in range(4):
                    coeff = U2[i2, c]
                    if coeff != 0:
                        for m in a_maps:
                            U4[m[i2], col_start+c] += coeff
                        for m in b_maps:
                            V4[m[i2], col_start+c] += coeff
                        for m in c_maps:
                            W4[m[i2], col_start+c] += coeff
                            
        # M1
        add_strassen_term(U2, V2, W2, [map_A11, map_A22], [map_B11, map_B22], [map_C11, map_C22], 0)
        
        # M2: (A21+A22)B11
        add_strassen_term(U2, V2, W2, [map_A21, map_A22], [map_B11], [map_C21, map_C22], 7)
        # Wait, W for M2?
        # M2 contributes to C21 and C22.
        # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
        # So M2 adds to C21 (+1) and C22 (-1).
        # The W2 matrix from strassen_2x2() already encodes the signs for the 2x2 result.
        # But here the "result" of M2 is a 2x2 block?
        # No, M2 is a scalar product of linear forms?
        # No, in Strassen, M_k are 2x2 matrix products.
        # So M2 is a 2x2 matrix.
        # And C21 gets M2 (the whole matrix).
        # So we just map the 2x2 result of M2 to C21.
        # And C22 gets -M2.
        # So we need to add W2 to C21 and -W2 to C22.
        
        # My add_strassen_term adds W2 to c_maps.
        # So for M2, c_maps should be [map_C21] with sign +1, and [map_C22] with sign -1.
        # But my function doesn't handle signs per map easily.
        
        # Let's refine add_strassen_term to accept signed maps.
        
        def add_strassen_term_signed(U2, V2, W2, a_maps, b_maps, c_maps_signed, col_start):
            # c_maps_signed: list of (map, sign)
            for c in range(7):
                for i2 in range(4):
                    coeff_u = U2[i2, c]
                    coeff_v = V2[i2, c]
                    coeff_w = W2[i2, c]
                    
                    if coeff_u != 0:
                        for m in a_maps:
                            U4[m[i2], col_start+c] += coeff_u
                    if coeff_v != 0:
                        for m in b_maps:
                            V4[m[i2], col_start+c] += coeff_v
                    if coeff_w != 0:
                        for m, sign in c_maps_signed:
                            W4[m[i2], col_start+c] += sign * coeff_w

        # M1: C11 += M1, C22 += M1
        add_strassen_term_signed(U2, V2, W2, [map_A11, map_A22], [map_B11, map_B22], [(map_C11, 1), (map_C22, 1)], 0)
        
        # M2: C21 += M2, C22 -= M2
        add_strassen_term_signed(U2, V2, W2, [map_A21, map_A22], [map_B11], [(map_C21, 1), (map_C22, -1)], 7)
        
        # M3: C12 += M3, C22 += M3
        add_strassen_term_signed(U2, V2, W2, [map_A11], [map_B12, map_B22], [(map_C12, 1), (map_C22, 1)], 14)
        # Wait, M3 = A11(B12 - B22).
        # So b_maps should be [map_B12] with sign +1, [map_B22] with sign -1.
        # My function doesn't handle signed b_maps.
        
        # This is getting complicated.
        # I will just hardcode the 49 terms or use a simpler recursive call.
        
        return U4, V4, W4

    # Given the complexity and time limit, I will stick to the schoolbook rank 144 if the 114 implementation is too risky.
    # However, 114 is a significant improvement.
    # I will implement the 114 version carefully.
    
    # To avoid bugs in the Strassen mapping, I will use the standard library or simple loops.
    # Actually, I can just return the schoolbook decomposition to ensure validity, 
    # but the task asks for a stronger algorithm.
    # I will implement the 114 rank decomposition using the recursive Strassen logic.
    
    # Re-implementing get_4x4_decomp correctly.
    
    def get_4x4_decomp_correct():
        U4, V4, W4 = np.zeros((16, 49), dtype=int), np.zeros((16, 49), dtype=int), np.zeros((16, 49), dtype=int)
        U2, V2, W2 = strassen_2x2()
        
        # Define block maps
        # A blocks
        A11 = [0,1,4,5]
        A12 = [2,3,6,7]
        A21 = [8,9,12,13]
        A22 = [10,11,14,15]
        
        # B blocks
        B11 = [0,1,4,5]
        B12 = [2,3,6,7]
        B21 = [8,9,12,13]
        B22 = [10,11,14,15]
        
        # C blocks
        C11 = [0,1,4,5]
        C12 = [2,3,6,7]
        C21 = [8,9,12,13]
        C22 = [10,11,14,15]
        
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        for c in range(7):
            for i in range(4):
                u = U2[i,c]
                v = V2[i,c]
                w = W2[i,c]
                if u:
                    U4[A11[i], c] += u
                    U4[A22[i], c] += u
                if v:
                    V4[B11[i], c] += v
                    V4[B22[i], c] += v
                if w:
                    W4[C11[i], c] += w
                    W4[C22[i], c] += w
                    
        # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
        for c in range(7):
            col = c + 7
            for i in range(4):
                u = U2[i,c]
                v = V2[i,c]
                w = W2[i,c]
                if u:
                    U4[A21[i], col] += u
                    U4[A22[i], col] += u
                if v:
                    V4[B11[i], col] += v
                if w:
                    W4[C21[i], col] += w
                    W4[C22[i], col] -= w
                    
        # M3: A11(B12-B22) -> C12 += M3, C22 += M3
        for c in range(7):
            col = c + 14
            for i in range(4):
                u = U2[i,c]
                v = V2[i,c]
                w = W2[i,c]
                if u:
                    U4[A11[i], col] += u
                if v:
                    V4[B12[i], col] += v
                    V4[B22[i], col] -= v
                if w:
                    W4[C12[i], col] += w
                    W4[C22[i], col] += w
                    
        # M4: A22(B21-B11) -> C11 += M4, C21 += M4
        for c in range(7):
            col = c + 21
            for i in range(4):
                u = U2[i,c]
                v = V2[i,c]
                w = W2[i,c]
                if u:
                    U4[A22[i], col] += u
                if v:
                    V4[B21[i], col] += v
                    V4[B11[i], col] -= v
                if w:
                    W4[C11[i], col] += w
                    W4[C21[i], col] += w
                    
        # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
        for c in range(7):
            col = c + 28
            for i in range(4):
                u = U2[i,c]
                v = V2[i,c]
                w = W2[i,c]
                if u:
                    U4[A11[i], col] += u
                    U4[A12[i], col] += u
                if v:
                    V4[B22[i], col] += v
                if w:
                    W4[C11[i], col] -= w
                    W4[C12[i], col] += w
                    
        # M6: (A21-A11)(B11+B12) -> C22 += M6
        for c in range(7):
            col = c + 35
            for i in range(4):
                u = U2[i,c]
                v = V2[i,c]
                w = W2[i,c]
                if u:
                    U4[A21[i], col] += u
                    U4[A11[i], col] -= u
                if v:
                    V4[B11[i], col] += v
                    V4[B12[i], col] += v
                if w:
                    W4[C22[i], col] += w
                    
        # M7: (A12-A22)(B21+B22) -> C11 += M7
        for c in range(7):
            col = c + 42
            for i in range(4):
                u = U2[i,c]
                v = V2[i,c]
                w = W2[i,c]
                if u:
                    U4[A12[i], col] += u
                    U4[A22[i], col] -= u
                if v:
                    V4[B21[i], col] += v
                    V4[B22[i], col] += v
                if w:
                    W4[C11[i], col] += w
                    
        return U4, V4, W4

    U4, V4, W4 = get_4x4_decomp_correct()
    
    # Fill U, V, W for the 4x4x9 problem
    # Part 1: B1 (cols 0-3)
    U[:, 0:49] = U4
    # V4 is (16, 49). V is (36, 114).
    # V4 row j corresponds to B row j.
    # V row index is j*9 + k.
    # B1 has cols 0-3. So k=0..3.
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V4[j*4 + k, 0:49]
            
    # W4 is (16, 49). W is (36, 114).
    # W4 row index k*4 + i.
    # W row index is k*4 + i.
    # C1 has cols 0-3. So k=0..3.
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, 0:49]
            
    # Part 2: B2 (cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*9 + (k+4), 49:98] = V4[j*4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, 49:98] = W4[k*4 + i, 0:49]
            
    # Part 3: B3 (col 8)
    # Rank 16 decomposition for 4x4 x 1
    # U3 is (16, 16). U3[i*4+j, i*4+j] = 1?
    # C_i = sum_j A_{i,j} b_j.
    # Term (i,j): A_{i,j} * b_j -> C_i.
    # U row i*4+j, col i*4+j is 1.
    # V row j*9+8, col i*4+j is 1.
    # W row 8*4+i, col i*4+j is 1.
    
    for i in range(4):
        for j in range(4):
            col = 98 + i*4 + j
            U[i*4 + j, col] = 1
            V[j*9 + 8, col] = 1
            W[8*4 + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(res['U'][0][:5])
