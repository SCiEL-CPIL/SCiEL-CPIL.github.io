def construct():
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms defined as (U_indices, V_indices, W_indices)
    # U index: k*n + i (row k, col i of A)
    # V index: k*m + i (row k, col i of B)
    # W index: k*n + i (row k, col i of C)
    # Note: The task description says W[k*n+i, r].
    # Standard Strassen:
    # M1 = (A11)(B11 - B22) -> C11, C22
    # M2 = (A11 + A12)(B22) -> C12, C22
    # M3 = (A21 - A11)(B11) -> C21, C22
    # M4 = (A22)(B11 + B21) -> C21, C22
    # M5 = (A11 + A22)(B11 + B22) -> C11, C12, C21, C22 (actually distributed)
    # M6 = (A12 - A22)(B21 + B22) -> C21, C11
    # M7 = (A11 - A21)(B11 + B12) -> C12, C22
    
    # Let's map the indices carefully to match the existing valid solution's logic 
    # or just generate a known valid Strassen decomposition.
    
    # Indices:
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:10, 3:11
    # C: 0:00, 1:01, 2:10, 3:11
    
    # Term 0: M1 = A11 * (B11 - B22)
    # U: A11 (0), V: B11(0), -B22(3)
    U[0][0] = 1
    V[0][0] = 1
    V[3][0] = -1
    # W: C11(0) += 1, C22(3) -= 1
    W[0][0] = 1
    W[3][0] = -1
    
    # Term 1: M2 = (A11 + A12) * B22
    # U: A11(0), A12(1), V: B22(3)
    U[0][1] = 1
    U[1][1] = 1
    V[3][1] = 1
    # W: C12(1) += 1, C22(3) += 1
    W[1][1] = 1
    W[3][1] = 1
    
    # Term 2: M3 = (A21 - A11) * B11
    # U: A21(2), -A11(0), V: B11(0)
    U[2][2] = 1
    U[0][2] = -1
    V[0][2] = 1
    # W: C21(2) += 1, C22(3) -= 1
    W[2][2] = 1
    W[3][2] = -1
    
    # Term 3: M4 = A22 * (B11 + B21)
    # U: A22(3), V: B11(0), B21(2)
    U[3][3] = 1
    V[0][3] = 1
    V[2][3] = 1
    # W: C21(2) += 1, C22(3) += 1
    W[2][3] = 1
    W[3][3] = 1
    
    # Term 4: M5 = (A11 + A22) * (B11 + B22)
    # U: A11(0), A22(3), V: B11(0), B22(3)
    U[0][4] = 1
    U[3][4] = 1
    V[0][4] = 1
    V[3][4] = 1
    # W: C11(0) += 1, C12(1) -= 1, C21(2) -= 1, C22(3) += 1
    # Wait, standard Strassen M5 contributes to C11, C12, C21, C22?
    # C11 = M1 + M5 - M2 - M4
    # C12 = M2 + M3
    # C21 = M6 + M7
    # C22 = M5 - M1 + M3 - M6
    # Let's check the standard formulas again.
    # M1 = A11(B11-B22)
    # M2 = (A11+A12)B22
    # M3 = (A21-A11)B11
    # M4 = A22(B11+B21)
    # M5 = (A11+A22)(B11+B22)
    # M6 = (A12-A22)(B21+B22)
    # M7 = (A11-A21)(B11+B12)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M2 + M3
    # C21 = M6 - M7
    # C22 = M1 - M3 + M4 - M5
    
    # Let's re-evaluate W for Term 4 (M5):
    # C11 += 1, C22 -= 1
    W[0][4] = 1
    W[3][4] = -1
    
    # Term 5: M6 = (A12 - A22) * (B21 + B22)
    # U: A12(1), -A22(3), V: B21(2), B22(3)
    U[1][5] = 1
    U[3][5] = -1
    V[2][5] = 1
    V[3][5] = 1
    # C21 += 1, C22 -= 1 (from C22 formula: -M5... wait. C22 = M1 - M3 + M4 - M5. M6 is not in C22?
    # Let's check C21 = M6 - M7. So C21 += 1.
    # What about C11? No. C12? No.
    # Wait, is M6 in C22? No.
    # Let's check C11 again. M1 - M4 + M5 + M7.
    # Let's check C12. M2 + M3.
    # Let's check C21. M6 - M7.
    # Let's check C22. M1 - M3 + M4 - M5.
    
    # So for M6:
    # W[2][5] = 1 (C21)
    # Any other? No.
    
    # Term 6: M7 = (A11 - A21) * (B11 + B12)
    # U: A11(0), -A21(2), V: B11(0), B12(1)
    U[0][6] = 1
    U[2][6] = -1
    V[0][6] = 1
    V[1][6] = 1
    # C11 += 1, C21 -= 1
    W[0][6] = 1
    W[2][6] = -1
    
    return {"U": U, "V": V, "W": W}
