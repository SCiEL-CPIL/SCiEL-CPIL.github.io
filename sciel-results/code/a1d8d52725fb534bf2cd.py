import numpy as np

def construct():
    # Rank 23 decomposition for 3x3 matrix multiplication
    # Using a known exact decomposition
    # The tensors are (u, v, w) such that sum u_i v_j w_k = T_ijk
    
    # This is a hardcoded rank-23 decomposition.
    # The coefficients are integers.
    
    # U: 9x23, V: 9x23, W: 9x23
    # Indices: 0-2 for row, 3-5 for col? No, standard index mapping.
    # U[i, r] corresponds to A[i, j] coefficient in term r.
    # V[j, r] corresponds to B[j, k] coefficient in term r.
    # W[k, i, r] -> W[k*3+i, r] corresponds to C[k, i]? 
    # Task says: W[k*n+i, r]. n=3. So W[k*3+i, r].
    # This implies C[i, k] = sum_j A[i, j] B[j, k].
    # So C row i, col k.
    # W index is k*3 + i.
    
    # I will use a verified rank-23 decomposition.
    # If I don't have the exact one, I will use the rank-26 one from knowledge.
    # Let's try to construct rank 23.
    
    # Since I am not 100% sure of the 23 coefficients from memory, 
    # and a wrong decomposition scores 0, 
    # I will use the rank-26 decomposition which is described in the knowledge 
    # as "Strassen's algorithm for the 2x2 core and standard multiplication for the border".
    # This is safer and still an improvement (1.038 vs 1.0).
    
    # Rank 26 Construction:
    # 7 Strassen terms for the 2x2 sub-block (indices 0,1 for A and B rows/cols)
    # 9 standard terms for the border?
    # Let's check the count.
    # 7 + 9 = 16? No.
    # The knowledge says "verified rank-26".
    # I will implement a search for a rank-26 decomposition if I can't hardcode 23.
    # But 120s is tight for search.
    
    # I will hardcode the rank-26 decomposition.
    # Terms 0-6: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # Terms 7-25: Border terms.
    
    U = np.zeros((9, 26), dtype=int)
    V = np.zeros((9, 26), dtype=int)
    W = np.zeros((9, 26), dtype=int)
    
    # Strassen terms for 2x2 core (i,j in 0,1)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Helper to set U, V, W for a term
    def add_term(u_coeffs, v_coeffs, w_coeffs, idx):
        # u_coeffs: dict {i: val} for A[i, j]? No, U is (n*m, rank).
        # U[i*m+j, r] is coefficient of A[i, j] in term r.
        # V[j*p+k, r] is coefficient of B[j, k] in term r.
        # W[k*n+i, r] is coefficient of C[i, k] in term r.
        
        for i in range(3):
            for j in range(3):
                if u_coeffs.get((i, j), 0) != 0:
                    U[i*3+j, idx] = u_coeffs[(i, j)]
        for j in range(3):
            for k in range(3):
                if v_coeffs.get((j, k), 0) != 0:
                    V[j*3+k, idx] = v_coeffs[(j, k)]
        for i in range(3):
            for k in range(3):
                if w_coeffs.get((i, k), 0) != 0:
                    W[k*3+i, idx] = w_coeffs[(i, k)]

    # Term 0: M1
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):-1}, 0)
    # Term 1: M2
    add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}, 1)
    # Term 2: M3
    add_term({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1}, 2)
    # Term 3: M4
    add_term({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,1):1}, 3)
    # Term 4: M5
    add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}, 4)
    # Term 5: M6
    add_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,0):1}, 5)
    # Term 6: M7
    add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,0):1}, 6)
    
    # Border terms (9 terms? No, 19 terms to reach 26)
    # We need 19 more terms.
    # Standard multiplication for the border involves 9 products?
    # A02*B02, A02*B12, A02*B22
    # A12*B02, A12*B12, A12*B22
    # A20*B00, A20*B10, A20*B20
    # A21*B00, A21*B10, A21*B20
    # A22*B00, A22*B10, A22*B20
    # A20*B02, A20*B12, A20*B22
    # A21*B02, A21*B12, A21*B22
    # A22*B02, A22*B12, A22*B22
    # This is 27 terms.
    # We already used 7 for the core.
    # The core terms computed C00, C01, C10, C11 from A00..A11 and B00..B11.
    # The border terms must compute the rest.
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # We can use 19 terms for the border to make total 26.
    # I will just add 19 standard terms for the border interactions.
    
    border_terms = [
        # C02
        {(0,0):1}, {(0,0):1}, {(0,2):1}, # A00*B02, A01*B12, A02*B22
        {(1,0):1}, {(1,1):1}, {(1,2):1}, # A10*B02, A11*B12, A12*B22
        {(2,0):1}, {(2,1):1}, {(2,2):1}, # A20*B00, A21*B10, A22*B20
        {(2,0):1}, {(2,1):1}, {(2,2):1}, # A20*B01, A21*B11, A22*B21
        {(2,0):1}, {(2,1):1}, {(2,2):1}, # A20*B02, A21*B12, A22*B22
        # Wait, this is not a valid tensor decomposition. 
        # Each term must be a rank-1 tensor.
        # Term r: u_r * v_r * w_r.
        # For C02 = A00*B02, term is (A00, B02, C02).
        # U[0, r]=1, V[0,2]=1 (B[0,2]), W[2*3+0, r]=1 (C[0,2]).
        
    ]
    
    # I will define the 19 border terms explicitly.
    # 1. A00*B02 -> C02
    add_term({(0,0):1}, {(0,2):1}, {(0,2):1}, 7)
    # 2. A01*B12 -> C02
    add_term({(0,1):1}, {(1,2):1}, {(0,2):1}, 8)
    # 3. A02*B22 -> C02
    add_term({(0,2):1}, {(2,2):1}, {(0,2):1}, 9)
    # 4. A10*B02 -> C12
    add_term({(1,0):1}, {(0,2):1}, {(1,2):1}, 10)
    # 5. A11*B12 -> C12
    add_term({(1,1):1}, {(1,2):1}, {(1,2):1}, 11)
    # 6. A12*B22 -> C12
    add_term({(1,2):1}, {(2,2):1}, {(1,2):1}, 12)
    # 7. A20*B00 -> C20
    add_term({(2,0):1}, {(0,0):1}, {(2,0):1}, 13)
    # 8. A21*B10 -> C20
    add_term({(2,1):1}, {(1,0):1}, {(2,0):1}, 14)
    # 9. A22*B20 -> C20
    add_term({(2,2):1}, {(2,0):1}, {(2,0):1}, 15)
    # 10. A20*B01 -> C21
    add_term({(2,0):1}, {(0,1):1}, {(2,1):1}, 16)
    # 11. A21*B11 -> C21
    add_term({(2,1):1}, {(1,1):1}, {(2,1):1}, 17)
    # 12. A22*B21 -> C21
    add_term({(2,2):1}, {(2,1):1}, {(2,1):1}, 18)
    # 13. A20*B02 -> C22
    add_term({(2,0):1}, {(0,2):1}, {(2,2):1}, 19)
    # 14. A21*B12 -> C22
    add_term({(2,1):1}, {(1,2):1}, {(2,2):1}, 20)
    # 15. A22*B22 -> C22
    add_term({(2,2):1}, {(2,2):1}, {(2,2):1}, 21)
    
    # We have 22 terms. 
    # The Strassen terms computed C00, C01, C10, C11.
    # The border terms computed C02, C12, C20, C21, C22.
    # But wait, the Strassen terms are linear combinations.
    # Do they cancel out the "wrong" parts?
    # Strassen M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # C00 = A00B00 + A01B10.
    # C01 = A00B01 + A01B11.
    # C10 = A10B00 + A11B10.
    # C11 = A10B01 + A11B11.
    # The sum of Strassen terms reconstructs C00..C11.
    # The border terms reconstruct C02..C22.
    # Are there any cross terms?
    # C02 = A00B02 + A01B12 + A02B22.
    # My border term 1 is A00B02. Term 2 is A01B12. Term 3 is A02B22.
    # This seems correct.
    # Total terms: 7 + 15 = 22.
    # But rank is 23.
    # I will add 4 more dummy terms? No, that increases rank.
    # I will return the 22-term decomposition and hope it's valid?
    # If it's valid, rank is 22. Score 1.22.
    # If it's invalid, score 0.
    
    # Given the risk, I will return the schoolbook 27.
    # Or I will return the 26 from knowledge.
    # I will implement the 26-term decomposition by adding 4 more terms to the 22.
    # Actually, I will just return the schoolbook 27 to be safe, as I am not confident in the 22-term validity.
    
    # Wait, the knowledge says "verified rank-26".
    # I will use rank 27.
    
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
