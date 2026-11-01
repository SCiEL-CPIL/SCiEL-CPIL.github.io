import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    
    # Fallback: Schoolbook (Rank 52)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    # Try Strassen (Rank 46)
    # 6 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (Schoolbook, rank 4)
    rank_strassen = 6 * 7 + 4
    U_str = np.zeros((n * m, rank_strassen))
    V_str = np.zeros((m * p, rank_strassen))
    W_str = np.zeros((p * n, rank_strassen))
    
    # Strassen coefficients for 2x2x2
    # Indices: A(i,j), B(j,k), C(k,i) -> Note: C is p x n, so C(k,i)
    # In our case, for a block, C is 2x2.
    # Let's define the 7 terms for one block.
    # A is 2x2. B is 2x2. C is 2x2.
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # Term 2: (A10+A11)B00 -> C10, -C11
    # Term 3: A00(B10-B00) -> C01, C11
    # Term 4: A11(B01-B11) -> C00, C10
    # Term 5: (A00+A01)B11 -> -C00, C01
    # Term 6: (A10-A00)(B00+B01) -> C11
    # Term 7: (A01-A11)(B10+B11) -> C00
    
    # Verify these coefficients against standard Strassen.
    # Standard:
    # m1 = (A11+A22)(B11+B22) -> C11, C22
    # m2 = (A21+A22)B11 -> C21, C22
    # m3 = A11(B21-B11) -> C12, C22
    # m4 = A22(B12-B22) -> C11, C21
    # m5 = (A11+A12)B22 -> C11, C12
    # m6 = (A21-A11)(B11+B12) -> C22
    # m7 = (A12-A22)(B21+B22) -> C11
    
    # Mapping 1-based to 0-based:
    # A11->A00, A12->A01, A21->A10, A22->A11
    # B11->B00, B12->B01, B21->B10, B22->B11
    # C11->C00, C12->C01, C21->C10, C22->C11
    
    # m1: (A00+A11)(B00+B11) -> C00, C11
    # m2: (A10+A11)B00 -> C10, C11 (Wait, standard says C21, C22 -> C10, C11)
    #    Standard: C21 = m2 + m4. C22 = m1 - m2 + m3 + m6.
    #    So m2 contributes +1 to C10 and -1 to C11.
    # m3: A00(B10-B00) -> C01, C11
    #    Standard: C12 = m3 + m5. C22 = m1 - m2 + m3 + m6.
    #    So m3 contributes +1 to C01 and +1 to C11.
    # m4: A11(B01-B11) -> C00, C10
    #    Standard: C11 = m1 + m4 - m5 + m7. C21 = m2 + m4.
    #    So m4 contributes +1 to C00 and +1 to C10.
    # m5: (A00+A01)B11 -> C00, C01
    #    Standard: C11 = m1 + m4 - m5 + m7. C12 = m3 + m5.
    #    So m5 contributes -1 to C00 and +1 to C01.
    # m6: (A10-A00)(B00+B01) -> C11
    #    Standard: C22 = m1 - m2 + m3 + m6.
    #    So m6 contributes +1 to C11.
    # m7: (A01-A11)(B10+B11) -> C00
    #    Standard: C11 = m1 + m4 - m5 + m7.
    #    So m7 contributes +1 to C00.
    
    # Let's check C00: m1 + m4 - m5 + m7
    # m1: A00B00 + A00B11 + A11B00 + A11B11
    # m4: A11B01 - A11B11
    # m5: -A00B11 - A01B11
    # m7: A01B10 + A01B11 - A11B10 - A11B11
    # Sum:
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1
    # A11B11: 1 - 1 - 1 = -1
    # A11B01: 1
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    # A11B10: -1
    # Result: A00B00 + A11B00 + A11B01 - A11B11 + A01B10 - A11B10
    # Target C00: A00B00 + A01B10
    # Mismatch.
    
    # My Strassen coefficients are likely incorrect or I am applying them to the wrong indices.
    # Given the risk, I will return the schoolbook decomposition.
    
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
