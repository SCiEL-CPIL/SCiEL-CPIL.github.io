"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition of rank 14 for 2x2x4 multiplication.
    # We treat the 2x4 matrix B as two 2x2 matrices B1 and B2.
    # C1 = A B1, C2 = A B2.
    # We use Strassen for each, but share the A-dependent terms if possible?
    # Actually, standard Strassen for 2x2 uses 7 multiplications.
    # Doing it twice gives 14.
    # We can try to merge them.
    
    n, m, p = DIMENSIONS
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen formulas for 2x2:
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For B1 (cols 0,1): e=B00, f=B01, g=B10, h=B11
    # For B2 (cols 2,3): e=B02, f=B03, g=B12, h=B13
    
    # We define the 7 multiplications for B1 (indices 0-6) and B2 (indices 7-13).
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        U[u_idx][w_idx] += val
        V[v_idx][w_idx] += val
        # W is set separately based on output
    
    # We need to map the Strassen multiplications to U, V, W.
    # U corresponds to linear forms in A.
    # V corresponds to linear forms in B.
    # W corresponds to the output matrix coefficients.
    
    # Since we are doing two separate Strassen decompositions, 
    # the U parts are the same for both (since A is the same), 
    # but the V parts are different.
    # This means we cannot simply use 7 columns for A and 7 for B.
    # Each multiplication $m_r$ is $u_r(A) v_r(B)$.
    # For B1, $v_r(B1)$. For B2, $v_r(B2)$.
    # So we have 7 pairs $(u_r, v_{r,1})$ and 7 pairs $(u_r, v_{r,2})$.
    # But $u_r$ is the same for both.
    # So we can use the same U columns for both?
    # No, the product is $u_r(A) v_{r,1}(B1)$ for C1 and $u_r(A) v_{r,2}(B2)$ for C2.
    # These are different terms in the sum.
    # So we need 14 columns in U?
    # Yes, because $U_{i,j,r}$ is multiplied by $V_{j,k,r}$.
    # If we use the same $r$ for B1 and B2, then $V_{j,k,r}$ would have to be both $v_{r,1}$ and $v_{r,2}$, which is impossible.
    # So we need distinct columns for each multiplication.
    # Total rank 14.
    
    # Define Strassen U coefficients (for A)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # u1 = a+d -> [1, 0, 0, 1]
    # u2 = c+d -> [0, 0, 1, 1]
    # u3 = a   -> [1, 0, 0, 0]
    # u4 = d   -> [0, 0, 0, 1]
    # u5 = a+b -> [1, 1, 0, 0]
    # u6 = c-a -> [-1, 0, 1, 0]
    # u7 = b-d -> [0, 1, 0, -1]
    
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Define Strassen V coefficients for B1 (cols 0,1)
    # B1 indices: 00, 01, 10, 11 -> 0, 1, 2, 3 in V (j*p+k)
    # v1 = e+h -> [1, 0, 0, 1]
    # v2 = e   -> [1, 0, 0, 0]
    # v3 = f-h -> [0, 1, 0, -1]
    # v4 = g-e -> [0, 0, 1, -1]
    # v5 = h   -> [0, 0, 0, 1]
    # v6 = e+f -> [1, 1, 0, 0]
    # v7 = g+h -> [0, 0, 1, 1]
    
    V_strassen_B1 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Define Strassen V coefficients for B2 (cols 2,3)
    # B2 indices: 02, 03, 12, 13 -> 2, 3, 6, 7 in V (j*p+k)
    # Note: V index is j*4+k.
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 6
    # j=1, k=3 -> 7
    
    V_strassen_B2 = [
        [0, 0, 1, 0, 0, 0, 0, 0], # e+h (B02+B13) -> indices 2, 7
        [0, 0, 1, 0, 0, 0, 0, 0], # e (B02)
        [0, 0, 0, 1, 0, 0, 0, -1], # f-h (B03-B13) -> indices 3, 7
        [0, 0, 0, 0, 0, 0, 1, -1], # g-e (B12-B02) -> indices 6, 2
        [0, 0, 0, 0, 0, 0, 0, 1], # h (B13)
        [0, 0, 1, 1, 0, 0, 0, 0], # e+f (B02+B03)
        [0, 0, 0, 0, 0, 0, 1, 1]  # g+h (B12+B13)
    ]
    # Correct V_strassen_B2 indices:
    # v1: e+h -> B02, B13 -> idx 2, 7
    # v2: e -> B02 -> idx 2
    # v3: f-h -> B03, B13 -> idx 3, 7
    # v4: g-e -> B12, B02 -> idx 6, 2
    # v5: h -> B13 -> idx 7
    # v6: e+f -> B02, B03 -> idx 2, 3
    # v7: g+h -> B12, B13 -> idx 6, 7
    
    V_strassen_B2 = [
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, -1],
        [0, 0, -1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1]
    ]

    # Fill U, V for the 14 columns
    for r in range(7):
        # Column r for B1
        for i in range(2):
            for j in range(2):
                U[i*2+j][r] = U_strassen[r][i*2+j]
        for j in range(2):
            for k in range(2):
                V[j*4+k][r] = V_strassen_B1[r][j*2+k]
        
        # Column r+7 for B2
        for i in range(2):
            for j in range(2):
                U[i*2+j][r+7] = U_strassen[r][i*2+j]
        for j in range(2):
            for k in range(2, 4):
                V[j*4+k][r+7] = V_strassen_B2[r][j*2 + (k-2)]

    # Now fill W
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Output indices: k*n + i
    # C11 (i=0, k=0) -> idx 0
    # C12 (i=0, k=1) -> idx 1
    # C21 (i=1, k=0) -> idx 2
    # C22 (i=1, k=1) -> idx 3
    # C13 (i=0, k=2) -> idx 4
    # C14 (i=0, k=3) -> idx 5
    # C23 (i=1, k=2) -> idx 6
    # C24 (i=1, k=3) -> idx 7
    
    # For B1 (cols 0,1), use m1..m7 (indices 0..6)
    # C11: m1, m4, -m5, m7
    W[0][0] += 1; W[0][3] += 1; W[0][4] -= 1; W[0][6] += 1
    # C12: m3, m5
    W[1][2] += 1; W[1][4] += 1
    # C21: m2, m4
    W[2][1] += 1; W[2][3] += 1
    # C22: m1, -m2, m3, m6
    W[3][0] += 1; W[3][1] -= 1; W[3][2] += 1; W[3][5] += 1
    
    # For B2 (cols 2,3), use m1..m7 (indices 7..13)
    # C13 (i=0, k=2) -> idx 4
    W[4][7] += 1; W[4][10] += 1; W[4][11] -= 1; W[4][13] += 1
    # C14 (i=0, k=3) -> idx 5
    W[5][9] += 1; W[5][11] += 1
    # C23 (i=1, k=2) -> idx 6
    W[6][8] += 1; W[6][10] += 1
    # C24 (i=1, k=3) -> idx 7
    W[7][7] += 1; W[7][8] -= 1; W[7][9] += 1; W[7][12] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
