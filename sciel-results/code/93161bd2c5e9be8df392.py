import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]
DEADLINE = 120.0
START_TIME = time.time()

def construct():
    n, m, p = DIMENSIONS
    
    # The rank of the 2x3x3 matrix multiplication tensor is 17.
    # We construct a rank-17 decomposition.
    # Strategy: Split the multiplication into independent sub-problems.
    # C is 2x3. A is 2x3, B is 3x3.
    # Let's split A into A1 (2x2) and A2 (2x1) columns? Or rows?
    # Let's split by the last dimension of A and first of B.
    # A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (2x3); B_bottom (1x3)]  # Wait, B is 3x3.
    # C = A B = A_left B_top + A_right B_bottom
    # This is a sum of two matrix products:
    # 1. 2x2 * 2x3
    # 2. 2x1 * 1x3
    
    # Rank of 2x2 * 2x3?
    # This can be seen as two 2x2 * 2x2 multiplications plus a 2x2 * 2x1?
    # Actually, simpler decomposition:
    # The tensor M_{2,3,3} can be decomposed using the following known result:
    # R(2,3,3) <= R(2,2,2) + R(2,1,2) + R(2,3,1) ? No, dimensions must match.
    
    # Let's use the known optimal rank 17 construction.
    # One standard way is to use Strassen for a 2x2x2 part and schoolbook for the rest.
    
    # Partition A (2x3) into A_0 (2x2) and A_1 (2x1)
    # Partition B (3x3) into B_0 (2x3) and B_1 (1x3)
    # C = A_0 B_0 + A_1 B_1
    
    # Term 1: A_0 (2x2) * B_0 (2x3)
    # B_0 is 2x3. We can write B_0 = [B_00 (2x2), B_01 (2x1)]
    # A_0 B_0 = A_0 B_00 + A_0 B_01
    # A_0 B_00 is 2x2 * 2x2. Rank 7 (Strassen).
    # A_0 B_01 is 2x2 * 2x1. This is equivalent to two 2x1 * 2x1? No.
    # A_0 is 2x2, B_01 is 2x1. Result 2x1.
    # This is just 2x2 matrix-vector multiplication.
    # Rank of 2x2 * 2x1 tensor?
    # It's equivalent to multiplying a 2x2 matrix by a 2x1 vector.
    # This can be done with 4 multiplications (schoolbook) or maybe less?
    # Actually, 2x2 * 2x1 is just 4 scalar multiplications and additions.
    # Is rank 4 optimal? Yes, for rectangular m x n where min(m,n)=1, rank is m*n.
    # So A_0 B_01 has rank 4.
    
    # Term 2: A_1 (2x1) * B_1 (1x3)
    # This is outer product of 2x1 and 1x3. Result 2x3.
    # This is equivalent to 2x3 matrix multiplication where one side is rank 1.
    # Actually, A_1 is a vector u (2x1), B_1 is a row vector v (1x3).
    # C_part = u v^T.
    # This requires 2*3 = 6 multiplications?
    # u_i v_j. There are 6 entries.
    # Can we do better?
    # The tensor is the outer product tensor.
    # Rank of outer product of n and p is n*p.
    # So A_1 B_1 has rank 6.
    
    # Total rank = 7 (Strassen) + 4 (Schoolbook 2x2*2x1) + 6 (Outer 2x1*1x3) = 17.
    
    # This matches the known optimal rank 17.
    
    R = 17
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Indices:
    # U: 2x3 flattened -> 6 rows. (i, j) where i in 0..1, j in 0..2.
    # V: 3x3 flattened -> 9 rows. (j, k) where j in 0..2, k in 0..2.
    # W: 3x2 flattened -> 6 rows. (k, i) where k in 0..2, i in 0..1.
    # Note: W index is k*n + i.
    
    col = 0
    
    # Part 1: A_0 (2x2) * B_00 (2x2) using Strassen
    # A_0 indices: i in {0,1}, j in {0,1}
    # B_00 indices: j in {0,1}, k in {0,1}
    # Result C_00: i in {0,1}, k in {0,1}
    
    # Strassen formulas for C = A B (2x2):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Map to our indices:
    # A11 -> U[0, col], A22 -> U[3, col] (i=1, j=1 -> 1*2+1=3)
    # B11 -> V[0, col], B22 -> V[3, col] (j=1, k=1 -> 1*2+1=3)
    # etc.
    
    def add_strassen_term(coeffs_U, coeffs_V, coeffs_W):
        nonlocal col
        for idx, coeff in coeffs_U:
            U[idx, col] += coeff
        for idx, coeff in coeffs_V:
            V[idx, col] += coeff
        for idx, coeff in coeffs_W:
            W[idx, col] += coeff
        col += 1
        
    # M1
    # U: A11(0), A22(3)
    # V: B11(0), B22(3)
    # W: C11(0), C22(3) -- wait, W index is k*n+i.
    # C11: k=0, i=0 -> 0*2+0=0
    # C22: k=1, i=1 -> 1*2+1=3
    # Coeffs:
    # M1 contributes +1 to C11, +1 to C22.
    add_strassen_term([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    
    # M2
    # U: A21(2), A22(3)
    # V: B11(0)
    # W: C21(2), C22(3)
    # C21: k=0, i=1 -> 0*2+1=1? No.
    # W index: k*n + i.
    # C is 2x3? No, this sub-part is 2x2.
    # C_00 is top-left 2x2 of C.
    # C[i,k]. i in 0..1, k in 0..1.
    # W index for C[i,k] is k*n + i.
    # C11 (i=0,k=0) -> 0
    # C12 (i=0,k=1) -> 1*2+0=2
    # C21 (i=1,k=0) -> 0*2+1=1
    # C22 (i=1,k=1) -> 1*2+1=3
    
    # M2: (A21+A22)B11
    # U: (2,1), (3,1)
    # V: (0,1)
    # W: C21(1) coeff 1, C22(3) coeff 1
    add_strassen_term([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, 1)])
    
    # M3
    # U: A11(0)
    # V: B12(1), B22(3) -> B12(1, -1)? No.
    # M3 = A11 * (B12 - B22)
    # V: (1, 1), (3, -1)
    # W: C12(2) coeff 1
    add_strassen_term([(0, 1)], [(1, 1), (3, -1)], [(2, 1)])
    
    # M4
    # U: A22(3)
    # V: B21(2), B11(0) -> (B21 - B11)
    # V: (2, 1), (0, -1)
    # W: C11(0) coeff 1, C21(1) coeff 1
    add_strassen_term([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # M5
    # U: A11(0), A12(1)
    # V: B22(3)
    # W: C11(0) coeff -1, C12(2) coeff 1
    add_strassen_term([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)])
    
    # M6
    # U: A21(2), A11(0) -> (A21 - A11)
    # V: B11(0), B12(1)
    # W: C22(3) coeff 1
    add_strassen_term([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    
    # M7
    # U: A12(1), A22(3) -> (A12 - A22)
    # V: B21(2), B22(3)
    # W: C11(0) coeff 1
    add_strassen_term([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    
    # Part 2: A_0 (2x2) * B_01 (2x1)
    # A_0: i in 0..1, j in 0..1
    # B_01: j in 0..1, k in 2 (since B is 3x3, B_01 is cols 0..1 of B? No.)
    # Let's re-verify the split.
    # A = [A_0 (2x2), A_1 (2x1)] -> Columns 0,1 and 2.
    # B = [B_0 (2x3); B_1 (1x3)] -> Rows 0,1 and 2.
    # C = A_0 B_0 + A_1 B_1
    # B_0 is 2x3. B_01 is the 3rd column of B_0? No, B_0 is rows 0,1 of B.
    # B_0 is 2x3.
    # A_0 is 2x2.
    # A_0 B_0 is 2x3.
    # This involves j in 0..1 for A, and k in 0..2 for B.
    # Wait, my previous split was:
    # A_0 B_00 (2x2*2x2) + A_0 B_01 (2x2*2x1)
    # This implies B_0 was split into B_00 (2x2) and B_01 (2x1).
    # B_00 uses k=0,1. B_01 uses k=2.
    # So B_01 is the column k=2 of B_0.
    # B_01 indices: j in 0..1, k=2.
    # V indices for B_01: j*p + k.
    # j=0, k=2 -> 0*3+2=2
    # j=1, k=2 -> 1*3+2=5
    
    # A_0 indices: i in 0..1, j in 0..1.
    # U indices: i*m+j.
    # i=0,j=0 -> 0
    # i=0,j=1 -> 1
    # i=1,j=0 -> 2
    # i=1,j=1 -> 3
    
    # Result C_01: i in 0..1, k=2.
    # W indices: k*n+i.
    # k=2, i=0 -> 2*2+0=4
    # k=2, i=1 -> 2*2+1=5
    
    # Schoolbook for 2x2 * 2x1:
    # C_i2 = A_i0 B_02 + A_i1 B_12
    # 4 terms.
    
    # Term 1: A_00 B_02
    U[0, col] = 1; V[2, col] = 1; W[4, col] = 1; col += 1
    # Term 2: A_01 B_12
    U[1, col] = 1; V[5, col] = 1; W[4, col] = 1; col += 1
    # Term 3: A_10 B_02
    U[2, col] = 1; V[2, col] = 1; W[5, col] = 1; col += 1
    # Term 4: A_11 B_12
    U[3, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    
    # Part 3: A_1 (2x1) * B_1 (1x3)
    # A_1: i in 0..1, j=2.
    # U indices: i*3+2.
    # i=0 -> 2
    # i=1 -> 5
    
    # B_1: j=2, k in 0..2.
    # V indices: 2*3+k.
    # k=0 -> 6
    # k=1 -> 7
    # k=2 -> 8
    
    # Result C_1: i in 0..1, k in 0..2.
    # W indices: k*n+i.
    # i=0, k=0 -> 0
    # i=0, k=1 -> 2
    # i=0, k=2 -> 4
    # i=1, k=0 -> 1
    # i=1, k=1 -> 3
    # i=1, k=2 -> 5
    
    # Outer product: C_ik = A_i2 B_2k
    # 6 terms.
    
    # i=0, k=0
    U[2, col] = 1; V[6, col] = 1; W[0, col] = 1; col += 1
    # i=0, k=1
    U[2, col] = 1; V[7, col] = 1; W[2, col] = 1; col += 1
    # i=0, k=2
    U[2, col] = 1; V[8, col] = 1; W[4, col] = 1; col += 1
    # i=1, k=0
    U[5, col] = 1; V[6, col] = 1; W[1, col] = 1; col += 1
    # i=1, k=1
    U[5, col] = 1; V[7, col] = 1; W[3, col] = 1; col += 1
    # i=1, k=2
    U[5, col] = 1; V[8, col] = 1; W[5, col] = 1; col += 1
    
    # Verify col == 17
    assert col == 17, f"Expected 17 terms, got {col}"
    
    # Optional: Local search to ensure half-integer validity (though this is already integer)
    # The problem statement allows integer/half-integer.
    # Our construction uses integers.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print(res)
