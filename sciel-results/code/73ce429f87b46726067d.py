import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the rank factors by combining:
    # 1. Strassen decomposition for 4x4x4 (rank 49) applied to two blocks of 4 columns of B
    # 2. Schoolbook decomposition for 4x4x1 (rank 16) applied to the remaining 1 column of B
    
    # The tensor T_{i,j,k} corresponds to C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Factors U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that T = sum_r U[:,r] V[:,r] W[:,r] (in tensor form)
    # Specifically, U_{i*m+j, r}, V_{j*p+k, r}, W_{k*n+i, r}
    
    # We will collect the factors for each block and concatenate them.
    
    U_list = []
    V_list = []
    W_list = []
    
    # --- Strassen 4x4x4 ---
    # Strassen's algorithm for 4x4 matrix multiplication has rank 49.
    # It is derived from applying Strassen's 2x2x2 (rank 7) algorithm to 2x2 blocks.
    # Let A = [[A00, A01], [A10, A11]] and B = [[B00, B01], [B10, B11]] where blocks are 2x2.
    # The 7 products are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # Then:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # To get rank 49 for 4x4x4, we apply this 7-product scheme to each of the 7 "virtual" 2x2x2 sub-problems?
    # Actually, the standard way to get rank 49 is to apply the rank 7 decomposition to each of the 7 combinations of block indices?
    # No, the rank 49 decomposition is obtained by applying the 2x2x2 rank 7 decomposition to each of the 7 "tuples" of block positions? 
    # Let's be precise. The 4x4x4 tensor is equivalent to 16 2x2x2 tensors? No.
    # The 4x4x4 multiplication can be seen as 16 scalar multiplications of 2x2 matrices? No.
    # It is the Kronecker product of 2x2x2 with 2x2x2? No, that would be rank 49.
    # Yes! The 4x4x4 tensor is the tensor product of two 2x2x2 tensors.
    # Since T_{4x4x4} = T_{2x2x2} \otimes T_{2x2x2}, its rank is rank(T_{2x2x2})^2 = 7^2 = 49.
    
    # The factors for T_{2x2x2} (rank 7) are:
    # Let indices be i,j in {0,1} for A and B respectively, k in {0,1} for C.
    # U_{i,j,r}, V_{j,k,r}, W_{k,i,r} for r in 0..6.
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Let's define the 7 linear forms for A (2x2), B (2x2), C (2x2).
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Form r: coefficients for A, B, C
    # r=0 (M1): A: (1,0,0,1), B: (1,0,0,1), C: (1,0,0,-1) -> Wait, C00 gets M1, C11 gets M1.
    # Let's check the reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for C, the coefficients for each M are:
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 += -1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += 1, C10 += 1
    # M5: C00 += -1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1
    
    # Let's verify:
    # C00: 1(M1) + 1(M4) - 1(M5) + 1(M7) = M1 + M4 - M5 + M7. Correct.
    # C01: 1(M3) + 1(M5) = M3 + M5. Correct.
    # C10: 1(M2) + 1(M4) = M2 + M4. Correct.
    # C11: 1(M1) - 1(M2) + 1(M3) + 1(M6) = M1 - M2 + M3 + M6. Correct.
    
    # Now for A and B:
    # M1 = (A00 + A11) * (B00 + B11)
    # A coeffs: (1, 0, 0, 1) -> A00=1, A01=0, A10=0, A11=1
    # B coeffs: (1, 0, 0, 1) -> B00=1, B01=0, B10=0, B11=1
    
    # M2 = (A10 + A11) * B00
    # A coeffs: (0, 0, 1, 1)
    # B coeffs: (1, 0, 0, 0)
    
    # M3 = A00 * (B01 - B11)
    # A coeffs: (1, 0, 0, 0)
    # B coeffs: (0, 1, 0, -1)
    
    # M4 = A11 * (B10 - B00)
    # A coeffs: (0, 0, 0, 1)
    # B coeffs: (-1, 0, 1, 0)
    
    # M5 = (A00 + A01) * B11
    # A coeffs: (1, 1, 0, 0)
    # B coeffs: (0, 0, 0, 1)
    
    # M6 = (A10 - A00) * (B00 + B01)
    # A coeffs: (-1, 0, 1, 0)
    # B coeffs: (1, 1, 0, 0)
    
    # M7 = (A01 - A11) * (B10 + B11)
    # A coeffs: (0, 1, 0, -1)
    # B coeffs: (0, 0, 1, 1)
    
    # C coeffs (for C00, C01, C10, C11):
    # M1: (1, 0, 0, 1)
    # M2: (0, 0, 1, -1)
    # M3: (0, 1, 0, 1)
    # M4: (1, 0, 1, 0)
    # M5: (-1, 1, 0, 0)
    # M6: (0, 0, 0, 1)
    # M7: (1, 0, 0, 0)
    
    A_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    B_forms = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    C_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Now, for 4x4x4, we have 7x7 = 49 terms.
    # Each term is a tensor product of one 2x2x2 form with another.
    # The indices for 4x4 are (i1,i2), (j1,j2), (k1,k2) where each is in {0,1}.
    # A index: (i1,i2, j1,j2) -> flattened as i1*2 + i2, j1*2 + j2? 
    # In our convention, A is 4x4, so index is i*m+j = i*4+j.
    # i is in 0..3, j is in 0..3.
    # i = i1*2 + i2, j = j1*2 + j2.
    # So A index = (i1*2 + i2)*4 + (j1*2 + j2).
    # This is not a simple Kronecker product of indices, but the values are.
    # Actually, the tensor product of two 2x2x2 tensors T1 and T2 gives a 4x4x4 tensor T.
    # T_{(i1,i2),(j1,j2),(k1,k2)} = sum_{r1,r2} U1_{i1,j1,r1} V1_{j1,k1,r1} W1_{k1,i1,r1} * U2_{i2,j2,r2} V2_{j2,k2,r2} W2_{k2,i2,r2}
    # = sum_{r1,r2} (U1_{i1,j1,r1} U2_{i2,j2,r2}) * (V1_{j1,k1,r1} V2_{j2,k2,r2}) * (W1_{k1,i1,r1} W2_{k2,i2,r2})
    
    # So the factors for 4x4x4 are:
    # U_{(i1,i2),(j1,j2), (r1,r2)} = U1_{i1,j1,r1} * U2_{i2,j2,r2}
    # V_{(j1,j2),(k1,k2), (r1,r2)} = V1_{j1,k1,r1} * V2_{j2,k2,r2}
    # W_{(k1,k2),(i1,i2), (r1,r2)} = W1_{k1,i1,r1} * W2_{k2,i2,r2}
    
    # Here, U1, V1, W1 are the 2x2 factors for the "first" pair of indices, and U2, V2, W2 for the "second".
    # But wait, the 2x2 decomposition we defined above is for a single 2x2x2 problem.
    # The 4x4x4 problem is the tensor product of two 2x2x2 problems.
    # So we can use the same forms for both "slices".
    
    # Let's construct the 49 factors for 4x4x4.
    # For each r1 in 0..6, r2 in 0..6:
    # r = r1 * 7 + r2
    
    # U factor for (i,j) where i=i1*2+i2, j=j1*2+j2:
    # U_{i,j,r} = A_forms[r1][i1*2+j1] * A_forms[r2][i2*2+j2]
    # Wait, A_forms[r] is a vector of 4 coefficients for A_{00}, A_{01}, A_{10}, A_{11}.
    # A_{i1,j1} corresponds to A_forms[r][i1*2+j1].
    # So U_{i1*2+i2, j1*2+j2, r1*7+r2} = A_forms[r1][i1*2+j1] * A_forms[r2][i2*2+j2]
    
    # Similarly for V and W.
    
    def get_strassen_4x4_factors():
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for i2 in range(2):
                        for j1 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U[i*4+j, r] = A_forms[r1][i1*2+j1] * A_forms[r2][i2*2+j2]
                                
                                V[j*4+k, r] = 0 # Placeholder, need k
                                # V index is j*p+k. Here p=4, so V[j*4+k, r]
                                # V_{j1,j2,k1,k2} = B_forms[r1][j1*2+k1] * B_forms[r2][j2*2+k2]
                                # But we are iterating over i,j. We need to iterate over j,k for V.
                                # Let's restructure the loops or fill V and W separately.
        
        # Let's fill U, V, W properly.
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                # Fill U
                for i1 in range(2):
                    for i2 in range(2):
                        for j1 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U[i*4+j, r] = A_forms[r1][i1*2+j1] * A_forms[r2][i2*2+j2]
                
                # Fill V
                for j1 in range(2):
                    for j2 in range(2):
                        for k1 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                V[j*4+k, r] = B_forms[r1][j1*2+k1] * B_forms[r2][j2*2+k2]
                
                # Fill W
                for k1 in range(2):
                    for k2 in range(2):
                        for i1 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                W[k*4+i, r] = C_forms[r1][k1*2+i1] * C_forms[r2][k2*2+i2]
        
        return U, V, W
    
    U_str, V_str, W_str = get_strassen_4x4_factors()
    
    # Now we apply this to the first 8 columns of B (two blocks of 4).
    # B is 4x9. Let B1 = B[:, 0:4], B2 = B[:, 4:8].
    # For B1, the V factor is V_str. The rows of V_str correspond to B_{j,k} for k=0..3.
    # In the full problem, B is 4x9. V has shape (m*p, rank) = (4*9, rank) = (36, rank).
    # For the first block, the k index in V_str corresponds to the global k index 0..3.
    # So for the first block, we place V_str into V[:, 0:4] (rows j*9 + k for k=0..3).
    # Wait, V index is j*p+k. p=9.
    # So for the first block, we place V_str into rows j*9 + k for k=0..3.
    
    # For the second block, B2 = B[:, 4:8]. The k index in V_str corresponds to global k=4..7.
    # So we place V_str into rows j*9 + (k+4) for k=0..3.
    
    # The U and W factors are the same for both blocks?
    # U depends on A, which is the same for both blocks.
    # W depends on C. For the first block, C is C[:, 0:4]. For the second, C[:, 4:8].
    # So W for the first block is placed in columns 0..3 of W (global k=0..3).
    # W index is k*n+i.
    # For the first block, we place W_str into columns k*4+i for k=0..3.
    # For the second block, we place W_str into columns (k+4)*4+i for k=0..3.
    
    # So for each block, we have 49 rank terms.
    # Total rank so far: 49 + 49 = 98.
    
    # Now for the last column B3 = B[:, 8]. This is a 4x1 matrix.
    # We use schoolbook for 4x4x1. Rank = 4*4*1 = 16.
    # U, V, W for 4x4x1:
    # U_{i,j, r} = 1 if r = i*4+j, else 0.
    # V_{j,8, r} = 1 if r = j, else 0? No.
    # Schoolbook: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Here k=8 (only one column).
    # So C_{i,8} = sum_j A_{i,j} B_{j,8}.
    # This is 16 terms, one for each (i,j).
    # For each (i,j), we have a term with:
    # U_{i*4+j, r} = 1
    # V_{j*9+8, r} = 1
    # W_{8*4+i, r} = 1
    # where r is the index of the (i,j) pair.
    
    # Let's construct the full factors.
    rank = 49 + 49 + 16
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    # Block 1: columns 0-3
    offset = 0
    for r in range(49):
        U[:, offset+r] = U_str[:, r]
        # V: rows j*9+k for k=0..3
        for j in range(4):
            for k in range(4):
                V[j*9+k, offset+r] = V_str[j*4+k, r]
        # W: columns k*4+i for k=0..3
        for k in range(4):
            for i in range(4):
                W[k*4+i, offset+r] = W_str[k*4+i, r]
    
    # Block 2: columns 4-7
    offset = 49
    for r in range(49):
        U[:, offset+r] = U_str[:, r]
        # V: rows j*9+k for k=4..7
        for j in range(4):
            for k in range(4):
                V[j*9+(k+4), offset+r] = V_str[j*4+k, r]
        # W: columns k*4+i for k=4..7
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i, offset+r] = W_str[k*4+i, r]
    
    # Block 3: column 8
    offset = 98
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j, offset+col] = 1
            V[j*9+8, offset+col] = 1
            W[8*4+i, offset+col] = 1
            col += 1
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# Test the decomposition
if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Verify the decomposition
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # This should equal 1 if i,j,k are valid indices for the multiplication tensor, else 0.
    # Actually, the tensor is T_{i,j,k} = delta_{i,j,k} for the multiplication?
    # No, the multiplication tensor has T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}.
    # So T_{i,j,k} = 1 for all i,j,k in the valid ranges? No.
    # The tensor is defined such that <T, A \otimes B> = A B.
    # So T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # This is always true for valid i,j,k. So T_{i,j,k} = 1 for all i,j,k?
    # Wait, no. The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k where e_i are basis vectors.
    # Actually, the multiplication tensor T_{m,n,p} has components T_{i,j,k} = 1 if i in 0..m-1, j in 0..n-1, k in 0..p-1.
    # But that's not right. The tensor is such that (A \otimes B) T = C.
    # The components are T_{i,j,k} = 1 if the index triple (i,j,k) is part of the multiplication.
    # For matrix multiplication, T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T = sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{i,k}? No.
    # Let's check the schoolbook decomposition.
    # U_{i*m+j, r} = 1 if r = i*m*p + j*p + k? No.
    # In the seed program:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # for each (i,j,k).
    # So the tensor T has T_{i*m+j, j*p+k, k*n+i} = 1 for each (i,j,k).
    # This is a permutation of the indices.
    # The contraction is sum_r U_{a,r} V_{b,r} W_{c,r} where a=i*m+j, b=j*p+k, c=k*n+i.
    # This should equal 1 if (a,b,c) correspond to a valid (i,j,k), else 0.
    # But actually, the tensor is defined on the index sets {i*m+j}, {j*p+k}, {k*n+i}.
    # So we can verify by checking that for each (i,j,k), the sum is 1, and for other triples, it is 0.
    # But since the index sets are disjoint for different (i,j,k), we can just check that the sum is 1 for each (i,j,k).
    
    # Let's verify.
    n, m, p = 4, 4, 9
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if val != 1:
                    print(f"Error at ({i},{j},{k}): got {val}")
    
    # Also check that the sum is 0 for random other triples.
    import random
    for _ in range(1000):
        a = random.randint(0, 15)
        b = random.randint(0, 35)
        c = random.randint(0, 35)
        # Check if (a,b,c) corresponds to a valid (i,j,k)
        i = a // m
        j = a % m
        k1 = b // p
        k2 = b % p
        k3 = c // n
        k4 = c % n
        if not (j == k1 and i == k4 and k2 == k3):
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
            if val != 0:
                print(f"Error at ({a},{b},{c}): got {val}")
    
    print("Verification complete.")
