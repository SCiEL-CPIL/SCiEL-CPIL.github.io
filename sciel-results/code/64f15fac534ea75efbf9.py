import numpy as np
import time
import sys
from fractions import Fraction as F

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target dimensions: A is n x m (3x3), B is m x p (3x4), C is n x p (3x4)
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (12, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    # We construct a rank 34 decomposition using Strassen on 2x2 blocks.
    
    # Define Strassen factors for 2x2 matrix multiplication.
    # A (2x2) indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # B (2x2) indices: 0,1,2,3
    # C (2x2) indices: 0,1,2,3
    # The tensor for 2x2x2 multiplication has rank 7.
    # Factors are lists of (u, v, w) where u, v, w are length 4 vectors with -1, 0, 1.
    
    strassen_7 = [
        # M1 = (A11+A22)(B11+B22)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # C11 += M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1]), # C22 += M1
        # M2 = (A21+A22)B11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # C21 += M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # C12? No, C22? 
        # Let's use the standard Strassen formulas to be sure.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices for 2x2:
        # A: a00, a01, a10, a11 -> idx 0,1,2,3
        # B: b00, b01, b10, b11 -> idx 0,1,2,3
        # C: c00, c01, c10, c11 -> idx 0,1,2,3
    ]
    
    # Let's explicitly define the 7 terms with signs for C.
    # term: (U_vec, V_vec, W_vec)
    # U_vec has non-zero at A indices.
    # V_vec has non-zero at B indices.
    # W_vec has non-zero at C indices with sign.
    
    s1_u = [1, 0, 0, 1] # A11+A22
    s1_v = [1, 0, 0, 1] # B11+B22
    # M1 contributes to C11 (+1) and C22 (+1)
    
    s2_u = [0, 0, 1, 1] # A21+A22
    s2_v = [1, 0, 0, 0] # B11
    # M2 contributes to C21 (+1) and C22 (-1)
    
    s3_u = [1, 0, 0, 0] # A11
    s3_v = [0, 0, 1, -1] # B21-B11
    # M3 contributes to C12 (+1) and C22 (+1)
    
    s4_u = [0, 0, 0, 1] # A22
    s4_v = [0, 1, 0, -1] # B12-B22
    # M4 contributes to C11 (+1) and C21 (+1)
    
    s5_u = [1, 1, 0, 0] # A11+A12
    s5_v = [0, 0, 0, 1] # B22
    # M5 contributes to C11 (-1) and C12 (+1)
    
    s6_u = [0, 0, 1, -1] # A21-A11
    s6_v = [1, 1, 0, 0] # B11+B12
    # M6 contributes to C22 (+1)
    
    s7_u = [0, 1, 0, -1] # A12-A22
    s7_v = [0, 0, 1, 1] # B21+B22
    # M7 contributes to C11 (+1)
    
    # Collect all (u, v, w) triples for the 2x2x2 block.
    # Each term is a rank 1 tensor.
    # We need to map these to the global indices.
    
    # Block 1: A22 * B22 -> C22 (top-left 2x2 of C)
    # A indices: 0,1,2,3 (rows 0,1; cols 0,1 of A)
    # B indices: 0,1,2,3 (rows 0,1; cols 0,1 of B)
    # C indices: 0,1,2,3 (rows 0,1; cols 0,1 of C)
    
    # Block 2: A22 * B22p -> C22p (top-right 2x2 of C)
    # A indices: 0,1,2,3
    # B indices: 4,5,6,7 (rows 0,1; cols 2,3 of B)
    # C indices: 4,5,6,7 (rows 0,1; cols 2,3 of C)
    
    # Block 3: A21 * B12 -> C21 (bottom-left 2x1 of C? No, A21 is 2x1, B12 is 1x2 -> 2x2)
    # Wait, my partitioning was:
    # A = [A22 (2x2) A21 (2x1); A12 (1x2) A11 (1x1)]
    # B = [B22 (2x2) B22p (2x2); B12 (1x2) B12p (1x2)]
    # C = [A22 B22 + A21 B12, A22 B22p + A21 B12p; A12 B22 + A11 B12, A12 B22p + A11 B12p]
    
    # Let's re-verify the block shapes.
    # A (3x3):
    # Rows 0,1 (top 2), Row 2 (bottom 1)
    # Cols 0,1 (left 2), Col 2 (right 1)
    # A22: rows 0,1; cols 0,1 (2x2)
    # A21: rows 0,1; col 2 (2x1)
    # A12: row 2; cols 0,1 (1x2)
    # A11: row 2; col 2 (1x1)
    
    # B (3x4):
    # Rows 0,1 (top 2), Row 2 (bottom 1)
    # Cols 0,1 (left 2), Cols 2,3 (right 2)
    # B22: rows 0,1; cols 0,1 (2x2)
    # B22p: rows 0,1; cols 2,3 (2x2)
    # B12: row 2; cols 0,1 (1x2)
    # B12p: row 2; cols 2,3 (1x2)
    
    # C (3x4):
    # C22 (top-left 2x2): A22*B22 + A21*B12
    # C22p (top-right 2x2): A22*B22p + A21*B12p
    # C12 (bottom-left 1x2): A12*B22 + A11*B12
    # C12p (bottom-right 1x2): A12*B22p + A11*B12p
    
    # Multiplications:
    # 1. A22 (2x2) * B22 (2x2) -> 2x2. Rank 7.
    # 2. A22 (2x2) * B22p (2x2) -> 2x2. Rank 7.
    # 3. A21 (2x1) * B12 (1x2) -> 2x2. Rank 4 (Schoolbook: 2*1*2=4).
    # 4. A21 (2x1) * B12p (1x2) -> 2x2. Rank 4.
    # 5. A12 (1x2) * B22 (2x2) -> 1x2. Rank 4 (Schoolbook: 1*2*2=4).
    # 6. A12 (1x2) * B22p (2x2) -> 1x2. Rank 4.
    # 7. A11 (1x1) * B12 (1x2) -> 1x2. Rank 2 (Schoolbook: 1*1*2=2).
    # 8. A11 (1x1) * B12p (1x2) -> 1x2. Rank 2.
    
    # Total Rank: 7+7+4+4+4+4+2+2 = 34.
    
    # Helper to add a term to U, V, W
    # u_indices: list of (global_a_idx, coeff)
    # v_indices: list of (global_b_idx, coeff)
    # w_indices: list of (global_c_idx, coeff)
    
    U = [[0] * 34 for _ in range(9)]
    V = [[0] * 34 for _ in range(12)]
    W = [[0] * 34 for _ in range(12)]
    
    col = 0
    
    # Map local indices to global indices
    # A (3x3) -> U index: i*3 + j
    # B (3x4) -> V index: i*4 + j
    # C (3x4) -> W index: i*4 + j (Wait, task says W[k*n+i, r]? 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Standard matrix multiplication C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T[a,b,c] where a is A index, b is B index, c is C index.
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # Here n=3, m=3, p=4.
    # A index: i*m+j = 3i+j.
    # B index: j*p+k = 4j+k.
    # C index: k*n+i = 4k+i.
    # So W index is 4k+i.
    
    def add_strassen_block(a_start, b_start, c_start, a_step, b_step, c_step, a_dims, b_dims, c_dims):
        # This is getting complicated to generalize. I will hardcode the index mappings.
        pass

    # Block 1: A22 * B22 -> C22
    # A22 indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B22 indices: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # C22 indices: (0,0)->0, (0,1)->4, (1,0)->1, (1,1)->5  (Wait, C index is 4k+i)
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 4
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 5
    
    # Local 2x2 indices:
    # A: 0,1,2,3 -> Global: 0, 1, 3, 4
    # B: 0,1,2,3 -> Global: 0, 1, 4, 5
    # C: 0,1,2,3 -> Global: 0, 4, 1, 5
    
    a_map = [0, 1, 3, 4]
    b_map = [0, 1, 4, 5]
    c_map = [0, 4, 1, 5]
    
    strassen_terms = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]), # M1 -> C11
        ([1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1]), # M1 -> C22
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2 -> C21
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2 -> C12? No, C22 is -M2.
        # Let's stick to the 7 M's and add them to C with correct signs.
    ]
    
    # I will use a list of (u_vec, v_vec, w_vec) where w_vec has the sign.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We can represent each M as a rank 1 term, and then add it to the C components.
    # But we need to add them to the global U, V, W.
    # For each M, we have a U_vec, V_vec, W_vec (all 1s).
    # We add M to C11: U_vec, V_vec, W_vec with 1 at C11 pos.
    # We add M to C22: U_vec, V_vec, W_vec with 1 at C22 pos.
    # etc.
    
    def add_term(u_local, v_local, w_local, a_map, b_map, c_map, col_idx):
        for i in range(4):
            if u_local[i] != 0:
                U[a_map[i]][col_idx] += u_local[i]
            if v_local[i] != 0:
                V[b_map[i]][col_idx] += v_local[i]
            if w_local[i] != 0:
                W[c_map[i]][col_idx] += w_local[i]
        
    # Block 1: A22 * B22 -> C22
    # M1
    add_term([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0], a_map, b_map, c_map, col); col+=1
    add_term([1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1], a_map, b_map, c_map, col); col+=1
    # M2
    add_term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0], a_map, b_map, c_map, col); col+=1
    add_term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, -1], a_map, b_map, c_map, col); col+=1
    # M3
    add_term([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0], a_map, b_map, c_map, col); col+=1
    add_term([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1], a_map, b_map, c_map, col); col+=1
    # M4
    add_term([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 0, 0], a_map, b_map, c_map, col); col+=1
    add_term([0, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 0], a_map, b_map, c_map, col); col+=1
    # M5
    add_term([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0], a_map, b_map, c_map, col); col+=1
    add_term([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], a_map, b_map, c_map, col); col+=1
    # M6
    add_term([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1], a_map, b_map, c_map, col); col+=1
    # M7
    add_term([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0], a_map, b_map, c_map, col); col+=1
    
    # Block 2: A22 * B22p -> C22p
    # A22 indices same: 0,1,3,4
    # B22p indices: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # C22p indices: (0,2)->8, (0,3)->9, (1,2)->5? No.
    # C[0,2] -> k=2, i=0 -> 8
    # C[0,3] -> k=3, i=0 -> 9
    # C[1,2] -> k=2, i=1 -> 10
    # C[1,3] -> k=3, i=1 -> 11
    
    a_map2 = [0, 1, 3, 4]
    b_map2 = [2, 3, 6, 7]
    c_map2 = [8, 9, 10, 11]
    
    # Same Strassen terms
    add_term([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0], a_map2, b_map2, c_map2, col); col+=1
    add_term([1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1], a_map2, b_map2, c_map2, col); col+=1
    add_term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0], a_map2, b_map2, c_map2, col); col+=1
    add_term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, -1], a_map2, b_map2, c_map2, col); col+=1
    add_term([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0], a_map2, b_map2, c_map2, col); col+=1
    add_term([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1], a_map2, b_map2, c_map2, col); col+=1
    add_term([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 0, 0], a_map2, b_map2, c_map2, col); col+=1
    add_term([0, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 0], a_map2, b_map2, c_map2, col); col+=1
    add_term([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0], a_map2, b_map2, c_map2, col); col+=1
    add_term([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], a_map2, b_map2, c_map2, col); col+=1
    add_term([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1], a_map2, b_map2, c_map2, col); col+=1
    add_term([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0], a_map2, b_map2, c_map2, col); col+=1
    
    # Block 3: A21 * B12 -> C21 (part of C22)
    # A21 (2x1): indices (0,2)->2, (1,2)->5
    # B12 (1x2): indices (2,0)->8, (2,1)->9
    # C21 (2x2): indices (0,0)->0, (0,1)->4, (1,0)->1, (1,1)->5
    # Wait, C21 is the top-left 2x2 block?
    # C = [A22 B22 + A21 B12, ...]
    # So A21 B12 adds to C[0:2, 0:2].
    # A21[i, 0] * B12[0, j] -> C[i, j]
    # i in 0,1; j in 0,1.
    # Terms:
    # i=0, j=0: A[0,2]*B[2,0] -> C[0,0]
    # i=0, j=1: A[0,2]*B[2,1] -> C[0,1]
    # i=1, j=0: A[1,2]*B[2,0] -> C[1,0]
    # i=1, j=1: A[1,2]*B[2,1] -> C[1,1]
    
    U[2][col] = 1; V[8][col] = 1; W[0][col] = 1; col+=1
    U[2][col] = 1; V[9][col] = 1; W[4][col] = 1; col+=1
    U[5][col] = 1; V[8][col] = 1; W[1][col] = 1; col+=1
    U[5][col] = 1; V[9][col] = 1; W[5][col] = 1; col+=1
    
    # Block 4: A21 * B12p -> C22p (part of C22p)
    # A21 (2x1): 2, 5
    # B12p (1x2): (2,2)->10, (2,3)->11
    # C22p (2x2): 8, 9, 10, 11
    # i=0, j=0: A[0,2]*B[2,2] -> C[0,2]
    # i=0, j=1: A[0,2]*B[2,3] -> C[0,3]
    # i=1, j=0: A[1,2]*B[2,2] -> C[1,2]
    # i=1, j=1: A[1,2]*B[2,3] -> C[1,3]
    
    U[2][col] = 1; V[10][col] = 1; W[8][col] = 1; col+=1
    U[2][col] = 1; V[11][col] = 1; W[9][col] = 1; col+=1
    U[5][col] = 1; V[10][col] = 1; W[10][col] = 1; col+=1
    U[5][col] = 1; V[11][col] = 1; W[11][col] = 1; col+=1
    
    # Block 5: A12 * B22 -> C12 (part of C12)
    # A12 (1x2): (2,0)->6, (2,1)->7
    # B22 (2x2): 0,1,4,5
    # C12 (1x2): (2,0)->2, (2,1)->3
    # i=0 (only row), j=0,1
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0]
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1]
    
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col+=1
    U[7][col] = 1; V[4][col] = 1; W[2][col] = 1; col+=1
    U[6][col] = 1; V[1][col] = 1; W[3][col] = 1; col+=1
    U[7][col] = 1; V[5][col] = 1; W[3][col] = 1; col+=1
    
    # Block 6: A12 * B22p -> C12p (part of C12p)
    # A12 (1x2): 6, 7
    # B22p (2x2): 2,3,6,7
    # C12p (1x2): (2,2)->6, (2,3)->7
    
    U[6][col] = 1; V[2][col] = 1; W[6][col] = 1; col+=1
    U[7][col] = 1; V[6][col] = 1; W[6][col] = 1; col+=1
    U[6][col] = 1; V[3][col] = 1; W[7][col] = 1; col+=1
    U[7][col] = 1; V[7][col] = 1; W[7][col] = 1; col+=1
    
    # Block 7: A11 * B12 -> C12 (part of C12)
    # A11 (1x1): (2,2)->8
    # B12 (1x2): 8, 9
    # C12 (1x2): 2, 3
    # C[2,0] += A[2,2]B[2,0]
    # C[2,1] += A[2,2]B[2,1]
    
    U[8][col] = 1; V[8][col] = 1; W[2][col] = 1; col+=1
    U[8][col] = 1; V[9][col] = 1; W[3][col] = 1; col+=1
    
    # Block 8: A11 * B12p -> C12p (part of C12p)
    # A11 (1x1): 8
    # B12p (1x2): 10, 11
    # C12p (1x2): 6, 7
    
    U[8][col] = 1; V[10][col] = 1; W[6][col] = 1; col+=1
    U[8][col] = 1; V[11][col] = 1; W[7][col] = 1; col+=1
    
    # Total rank should be 34.
    # Check col
    assert col == 34, f"Expected rank 34, got {col}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
