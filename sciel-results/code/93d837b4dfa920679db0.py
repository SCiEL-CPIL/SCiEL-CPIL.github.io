DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33: 24 (Strassen-mixed for 3x3x3) + 9 (Standard for 3x3x1)
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # --- Part 2: 3x3x1 (Column 3) ---
    # 9 terms
    for i in range(n):
        for j in range(m):
            col = 24 + i*m + j
            U[i*m + j, col] = 1
            V[j*p + 3, col] = 1
            W[3*n + i, col] = 1
            
    # --- Part 1: 3x3x3 (Columns 0,1,2) ---
    # Strassen for 2x2 block (indices 1,2)
    # A block: rows 1,2 cols 1,2. Indices: 4,5,7,8
    # B block: rows 1,2 cols 0,1,2? No, B is 3x3 for this part.
    # B indices: j*4 + k.
    # B[1,1]=5, B[1,2]=6, B[2,1]=9, B[2,2]=10.
    # C block: rows 1,2 cols 0,1,2.
    # C[1,1] idx 1*3+1=4. C[1,2] idx 2*3+1=7.
    # C[2,1] idx 1*3+2=5. C[2,2] idx 2*3+2=8.
    
    # Strassen M1: (A11+A22)(B11+B22)
    # U: 4,8. V: 5,10.
    # W: C11(4), C12(7), C21(5), C22(8) with coeffs from Strassen.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M1 - M2 + M6
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # We will add the 7 Strassen terms.
    # M1
    U[4, 0] += 1; U[8, 0] += 1
    V[5, 0] += 1; V[10, 0] += 1
    W[4, 0] += 1; W[7, 0] += 1; W[5, 0] += 1; W[8, 0] += 1
    
    # M2: (A21+A22)B11 -> (A[2,1]+A[2,2]) B[1,1]
    U[7, 1] += 1; U[8, 1] += 1
    V[5, 1] += 1
    W[5, 1] += 1; W[8, 1] += 1 # C21, C22
    
    # M3: A11(B21-B11) -> A[1,1] (B[2,1]-B[1,1])
    U[4, 2] += 1
    V[9, 2] += 1; V[5, 2] -= 1
    W[7, 2] += 1; W[8, 2] -= 1 # C12, C22
    
    # M4: A22(B11-B21) -> A[2,2] (B[1,1]-B[2,1])
    U[8, 3] += 1
    V[5, 3] += 1; V[9, 3] -= 1
    W[4, 3] += 1; W[5, 3] += 1 # C11, C21
    
    # M5: (A11+A12)B22 -> (A[1,1]+A[1,2]) B[2,2]
    U[4, 4] += 1; U[5, 4] += 1
    V[10, 4] += 1
    W[4, 4] -= 1; W[8, 4] += 1 # C11, C22
    
    # M6: (A21-A11)(B11+B12) -> (A[2,1]-A[1,1]) (B[1,1]+B[1,2])
    U[7, 5] += 1; U[4, 5] -= 1
    V[5, 5] += 1; V[6, 5] += 1
    W[7, 5] += 1 # C12
    
    # M7: (A12-A22)(B21+B22) -> (A[1,2]-A[2,2]) (B[2,1]+B[2,2])
    U[5, 6] += 1; U[8, 6] -= 1
    V[9, 6] += 1; V[10, 6] += 1
    W[4, 6] += 1; W[7, 6] += 1 # C11, C12? 
    # Strassen C11 = M1 + M4 - M5 + M7. C12 = M3 + M1 - M2 + M6.
    # My W assignments above are just adding the term. The linear combination is handled by the sum of terms.
    # Wait, the tensor decomposition is a sum of rank-1 tensors.
    # T = sum r U_r V_r W_r.
    # So for each term, I add the product of the vectors.
    
    # Now the remaining terms for 3x3x3:
    # C00 = A00 B00
    # C01 = A00 B01 + A01 B11 + A02 B21 ? No.
    # C0j = A00 B0j + A01 B1j + A02 B2j.
    # C10 = A10 B00 + A11 B10 + A12 B20.
    # C20 = A20 B00 + A21 B10 + A22 B20.
    
    # We already handled A11 B11, A12 B22, etc in Strassen?
    # Strassen handled:
    # A11 B11 (in M1, M2, M3, M6)
    # A11 B12 (in M6)
    # A12 B11 (none? M5 is A12 B22)
    # A12 B21 (in M7)
    # A12 B22 (in M5, M7)
    # A21 B11 (in M2)
    # A21 B12 (in M6)
    # A22 B11 (in M1, M4)
    # A22 B12 (none)
    # A22 B21 (in M4, M7)
    # A22 B22 (in M1, M5)
    
    # Missing terms for C00, C01, C02, C10, C20 involving A0*, A*0, B*0.
    # C00 = A00 B00
    # C01 = A00 B01 + A01 B11 + A02 B21
    # C02 = A00 B02 + A01 B12 + A02 B22
    # C10 = A10 B00 + A11 B10 + A12 B20
    # C20 = A20 B00 + A21 B10 + A22 B20
    
    # We need to add these.
    # A00 B00 (1)
    U[0, 7] = 1; V[0, 7] = 1; W[0, 7] = 1
    
    # A00 B01 (1)
    U[0, 8] = 1; V[1, 8] = 1; W[3, 8] = 1 # C01 idx 1*3+0=3
    
    # A01 B11 (1)
    U[1, 9] = 1; V[5, 9] = 1; W[3, 9] = 1
    
    # A02 B21 (1)
    U[2, 10] = 1; V[9, 10] = 1; W[3, 10] = 1
    
    # A00 B02 (1)
    U[0, 11] = 1; V[2, 11] = 1; W[6, 11] = 1 # C02 idx 2*3+0=6
    
    # A01 B12 (1)
    U[1, 12] = 1; V[6, 12] = 1; W[6, 12] = 1
    
    # A02 B22 (1)
    U[2, 13] = 1; V[10, 13] = 1; W[6, 13] = 1
    
    # A10 B00 (1)
    U[3, 14] = 1; V[0, 14] = 1; W[1, 14] = 1 # C10 idx 0*3+1=1
    
    # A11 B10 (1)
    U[4, 15] = 1; V[4, 15] = 1; W[1, 15] = 1
    
    # A12 B20 (1)
    U[5, 16] = 1; V[8, 16] = 1; W[1, 16] = 1
    
    # A20 B00 (1)
    U[6, 17] = 1; V[0, 17] = 1; W[2, 17] = 1 # C20 idx 0*3+2=2
    
    # A21 B10 (1)
    U[7, 18] = 1; V[4, 18] = 1; W[2, 18] = 1
    
    # A22 B20 (1)
    U[8, 19] = 1; V[8, 19] = 1; W[2, 19] = 1
    
    # Total terms: 7 (Strassen) + 13 (Standard) = 20?
    # Wait, 20 + 9 = 29?
    # Did I miss some?
    # C11, C12, C21, C22 are fully covered by Strassen?
    # C11 = A11 B11 + A12 B21. Strassen covers A11 B11 and A12 B21?
    # M7 has A12 B21. M1 has A11 B11.
    # Yes.
    
    # So Rank is 7 + 13 + 9 = 29?
    # This is better than 32!
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
