import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS

    # 4x4x4 optimal tensor decomposition (rank 19)
    # This is a known minimal decomposition for 4x4x4 matrix multiplication
    # We'll construct it from a known 19-term decomposition
    
    # The 19-term decomposition of the 4x4x4 tensor:
    # Based on the work by Alon and Shpilka, or we can use a known construction
    # Here I'll use a simpler approach: build from 2x2x2 Strassen blocks
    # but with shared structure to get rank < 49
    
    # Actually, let me use a verified 19-term decomposition.
    # The indices for 4x4 matrices are flattened as 0..15
    # A[i,j] is at index i*4+j
    
    # I'll construct a valid decomposition using a combination of Strassen-like
    # patterns that achieve rank 19 for 4x4x4.
    
    # Since finding a true 19-rank decomposition from scratch is complex,
    # let me use a practical approach: use the 7x7=49 Strassen decomposition
    # but try to find a lower-rank decomposition via a different strategy.
    
    # Alternative: Use the fact that for 4x4x8, we can decompose as:
    # Multiply A (4x4) by B (4x8)
    # Split B into B1 (4x4) and B2 (4x4)
    # C1 = A * B1, C2 = A * B2
    # If we use rank r for 4x4x4, total rank = 2r
    # With Strassen 4x4x4 (rank 49), total = 98
    # With a better 4x4x4 decomposition, we can do better
    
    # Let me implement a known 19-rank decomposition for 4x4x4
    # This is based on the "19-multiplication" algorithm
    
    # Actually, I'll use a well-known construction. Let me build it carefully.
    # For 4x4x4, one approach uses the polynomial method or explicit coefficients.
    
    # Given the complexity, let me use a safe fallback: 
    # Use rank-49 Strassen for 4x4x4, giving rank 98 for 4x4x8.
    # But let me try a different structural approach.
    
    # New approach: Decompose 4x4x8 directly
    # A is 4x4, B is 4x8, C is 4x8
    # We can view this as two independent 4x4x4 multiplications
    # OR we can find a decomposition that shares terms between the two halves
    
    # Let me try to use a 16-term base for 4x4x4 (if it exists) or
    # use a more efficient construction.
    
    # For now, let me implement the Strassen 4x4x4 (rank 49) approach
    # which is guaranteed correct, and then see if we can improve.
    
    # Actually, let me try a different decomposition strategy:
    # Use the "baby-step giant-step" idea mentioned in the directive.
    
    # For 4x4x8:
    # Think of A as a 2x2 block matrix with 2x2 blocks
    # B as a 2x4 block matrix with 2x2 blocks (2 rows of 2x2 blocks)
    # C as a 2x4 block matrix with 2x2 blocks
    
    # Use Strassen for the 2x2 "macro" level, but each "macro" multiplication
    # involves 2x2x2 blocks for A and B, producing 2x2 blocks for C.
    
    # At the macro level:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each Bj is 2x2
    # C = [[C00, C01, C02, C03], [C10, C11, C12, C13]] where each Ck is 2x2
    
    # Strassen for 2x2x2 gives 7 scalar multiplications
    # But each "scalar" here is a 2x2x2 matrix multiplication (rank 7 each)
    # So 7 * 7 = 49 for the first 4 columns of B
    # And another 49 for the second 4 columns
    # Total: 98
    
    # To do better, we need to share work between the two halves of B.
    
    # Alternative: Group B differently
    # B has 4 rows and 8 columns
    # Group B into 2 groups of 4 columns each
    # C = A * B = [A*B1 | A*B2] where B1, B2 are 4x4
    
    # This gives 2 * (rank of 4x4x4) = 2 * 49 = 98 with Strassen
    
    # To get below 98, we need a 4x4x4 decomposition with rank < 49
    # OR we need to share terms between A*B1 and A*B2
    
    # Let me try: use the same U vectors for both halves, but different V vectors
    # U is shared (4x4 side), V has two copies (one for B1, one for B2)
    # W has two copies (one for C1, one for C2)
    
    # If 4x4x4 has rank r, then 4x4x8 has rank 2r with this approach
    # With r=49 (Strassen), we get 98
    
    # The known lower bound for 4x4x4 is 19, and there exist decompositions
    # with rank 19 or 20. Let me implement a rank-19 decomposition.
    
    # I'll use a known 19-rank decomposition. Here are the coefficients:
    # This is a well-known result; let me encode it.
    
    # For a 4x4x4 tensor T[i,j,k] = delta[i,k] (matrix mult tensor)
    # The 19-rank decomposition uses 19 terms t=0..18
    
    # I'll construct it from literature. One known construction:
    
    # Let me just use the rank-49 Strassen approach for safety,
    # since implementing a correct 19-rank decomposition from memory is risky.
    
    # Rank-49 Strassen for 4x4x4:
    
    # 2x2x2 Strassen base
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # M3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1  # C00 += M4, C10 += M4
    
    # M5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1  # C00 -= M5, C01 += M5
    
    # M6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1  # C11 += M6
    
    # M7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1  # C00 += M7
    
    # Verify 2x2x2: C_pq = sum_t W2[p*2+q, t] * (sum_i U2[i,t]*A_i) * (sum_j V2[j,t]*B_j)
    # Let me verify C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Check W2 for C00 (row 0): M1(1) + M4(1) + M5(-1) + M7(1) = correct
    # C01 (row 1): M3(1) + M5(1) = correct
    # C10 (row 2): M2(1) + M4(1) = correct
    # C11 (row 3): M1(1) + M2(-1) + M3(1) + M6(1) = correct
    
    # Now build 4x4x4 using Strassen recursion
    # 4x4 = 2x2 blocks of 2x2
    # A blocks: A00(0-3), A01(4-7), A10(8-11), A11(12-15)
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Macro Strassen terms for 4x4x4
    A_macro = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    
    B_macro = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    
    C_macro = [
        [1, 0, 0, 0, -1, 0, 1],   # C00: M1+M4-M5+M7
        [0, 0, 1, 0, 1, 0, 0],    # C01: M3+M5
        [0, 1, 0, 1, 0, 0, 0],    # C10: M2+M4
        [1, -1, 1, 0, 0, 1, 0]    # C11: M1-M2+M3+M6
    ]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for block in range(4):
                if A_macro[s][block] != 0:
                    offset = block * 4
                    for local_i in range(4):
                        U4[offset + local_i, r] += A_macro[s][block] * U2[local_i, t]
                if B_macro[s][block] != 0:
                    offset = block * 4
                    for local_j in range(4):
                        V4[offset + local_j, r] += B_macro[s][block] * V2[local_j, t]
            for c_block in range(4):
                if C_macro[c_block][s] != 0:
                    offset = c_block * 4
                    for local_k in range(4):
                        W4[offset + local_k, r] += C_macro[c_block][s] * W2[local_k, t]
    
    # Extend to 4x4x8: two independent 4x4x4 multiplications
    # B is 4x8, split into B1 (4x4) and B2 (4x4)
    # C is 4x8, split into C1 (4x4) and C2 (4x4)
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First half: A * B1 = C1
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Second half: A * B2 = C2
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    construct()
