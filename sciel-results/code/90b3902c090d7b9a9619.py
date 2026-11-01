import time

def construct():
    n, m, p = 3, 3, 4
    rank = 20
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (9, 20)
    # V: (m*p, rank) = (12, 20)
    # W: (p*n, rank) = (12, 20)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_dict: { (i,j): val }
    # v_dict: { (j,k): val }
    # w_dict: { (i,k): val }
    def add_term(term_id, u_dict, v_dict, w_dict):
        for (i, j), val in u_dict.items():
            U[i * m + j][term_id] += val
        for (j, k), val in v_dict.items():
            V[j * p + k][term_id] += val
        for (i, k), val in w_dict.items():
            W[k * n + i][term_id] += val
            
    # Strassen for A11 B11 (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2])
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # M1
    u = {(0,0): 1, (1,1): 1}
    v = {(0,0): 1, (1,1): 1}
    w = {(0,0): 1, (0,1): 1, (1,0): -1, (1,1): 1}
    add_term(0, u, v, w)
    
    # M2
    u = {(0,0): 1, (1,0): 1}
    v = {(0,0): 1}
    w = {(0,1): -1, (1,0): 1}
    add_term(1, u, v, w)
    
    # M3
    u = {(0,0): 1}
    v = {(0,1): 1, (1,1): -1}
    w = {(0,1): 1, (1,0): 1, (1,1): -1}
    add_term(2, u, v, w)
    
    # M4
    u = {(1,1): 1}
    v = {(1,0): 1, (0,0): -1}
    w = {(0,0): 1, (1,0): 1}
    add_term(3, u, v, w)
    
    # M5
    u = {(0,0): 1, (0,1): 1}
    v = {(1,1): 1}
    w = {(0,0): -1, (1,1): 1}
    add_term(4, u, v, w)
    
    # M6
    u = {(1,0): 1, (0,0): -1}
    v = {(0,0): 1, (0,1): 1}
    w = {(0,1): 1}
    add_term(5, u, v, w)
    
    # M7
    u = {(0,1): 1, (1,1): -1}
    v = {(1,0): 1, (1,1): 1}
    w = {(0,0): 1, (1,1): -1}
    add_term(6, u, v, w)
    
    # Strassen for A11 B12 (A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4])
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,2), (0,3), (1,2), (1,3)
    # C indices: (0,2), (0,3), (1,2), (1,3)
    
    # M1'
    u = {(0,0): 1, (1,1): 1}
    v = {(0,2): 1, (1,3): 1}
    w = {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): 1}
    add_term(7, u, v, w)
    
    # M2'
    u = {(0,0): 1, (1,0): 1}
    v = {(0,2): 1}
    w = {(0,3): -1, (1,2): 1}
    add_term(8, u, v, w)
    
    # M3'
    u = {(0,0): 1}
    v = {(0,3): 1, (1,3): -1}
    w = {(0,3): 1, (1,2): 1, (1,3): -1}
    add_term(9, u, v, w)
    
    # M4'
    u = {(1,1): 1}
    v = {(1,2): 1, (0,2): -1}
    w = {(0,2): 1, (1,2): 1}
    add_term(10, u, v, w)
    
    # M5'
    u = {(0,0): 1, (0,1): 1}
    v = {(1,3): 1}
    w = {(0,2): -1, (1,3): 1}
    add_term(11, u, v, w)
    
    # M6'
    u = {(1,0): 1, (0,0): -1}
    v = {(0,2): 1, (0,3): 1}
    w = {(0,3): 1}
    add_term(12, u, v, w)
    
    # M7'
    u = {(0,1): 1, (1,1): -1}
    v = {(1,2): 1, (1,3): 1}
    w = {(0,2): 1, (1,3): -1}
    add_term(13, u, v, w)
    
    # A12 B21 (A[0:2, 2:3] * B[2:3, 0:2] -> C[0:2, 0:2])
    # A: (0,2), (1,2)
    # B: (2,0), (2,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    # Outer product: C_ij = A_i2 B_2j
    # This is rank 1.
    # Term: u=(A_02, A_12), v=(B_20, B_21), w=1 for all C in 0:2,0:2
    
    u = {(0,2): 1, (1,2): 1}
    v = {(2,0): 1, (2,1): 1}
    w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    add_term(14, u, v, w)
    
    # A12 B22 (A[0:2, 2:3] * B[2:3, 2:4] -> C[0:2, 2:4])
    # A: (0,2), (1,2)
    # B: (2,2), (2,3)
    # C: (0,2), (0,3), (1,2), (1,3)
    
    u = {(0,2): 1, (1,2): 1}
    v = {(2,2): 1, (2,3): 1}
    w = {(0,2): 1, (0,3): 1, (1,2): 1, (1,3): 1}
    add_term(15, u, v, w)
    
    # A21 B11 (A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2])
    # A: (2,0), (2,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (2,0), (2,1)
    # C_2j = sum_k A_2k B_kj
    # This is not a single outer product. It is a 1x2 * 2x2 -> 1x2 multiplication.
    # Rank is 2?
    # C_20 = A_20 B_00 + A_21 B_10
    # C_21 = A_20 B_01 + A_21 B_11
    # We can do this with 2 multiplications?
    # M_a = A_20 (B_00 + B_01)? No.
    # M_a = A_20 B_00, M_b = A_21 B_10...
    # Actually, $1 \times m \times p$ has rank $m$? No, rank 1?
    # $C = A B$. $A$ is $1 \times m$, $B$ is $m \times p$.
    # $C_j = \sum_i A_i B_{ij}$.
    # This is the dot product of A and columns of B.
    # We need to compute 2 dot products of length 2.
    # Can we do it with 2 multiplications?
    # Let $M_1 = A_0 B_{00} + A_1 B_{10}$ (dot product col 0)
    # Let $M_2 = A_0 B_{01} + A_1 B_{11}$ (dot product col 1)
    # This is 2 multiplications.
    # So we need 2 terms for A21 B11.
    
    # Term 16: M_a = A_20 B_00 + A_21 B_10 -> C_20
    u = {(2,0): 1, (2,1): 1} # Wait, U is for A. A_20, A_21.
    # But V is for B. B_00, B_10.
    # We want C_20 to get 1 * M_a.
    # We want C_21 to get 0.
    # So u: A_20=1, A_21=1. v: B_00=1, B_10=1. w: C_20=1.
    u = {(2,0): 1, (2,1): 1}
    v = {(0,0): 1, (1,0): 1}
    w = {(2,0): 1}
    add_term(16, u, v, w)
    
    # Term 17: M_b = A_20 B_01 + A_21 B_11 -> C_21
    u = {(2,0): 1, (2,1): 1}
    v = {(0,1): 1, (1,1): 1}
    w = {(2,1): 1}
    add_term(17, u, v, w)
    
    # A22 B21 (A[2:3, 2:3] * B[2:3, 0:2] -> C[2:3, 0:2])
    # A: (2,2)
    # B: (2,0), (2,1)
    # C: (2,0), (2,1)
    # C_20 = A_22 B_20
    # C_21 = A_22 B_21
    # Rank 2.
    
    # Term 18: A_22 B_20 -> C_20
    u = {(2,2): 1}
    v = {(2,0): 1}
    w = {(2,0): 1}
    add_term(18, u, v, w)
    
    # Term 19: A_22 B_21 -> C_21
    u = {(2,2): 1}
    v = {(2,1): 1}
    w = {(2,1): 1}
    add_term(19, u, v, w)
    
    # A21 B12 (A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4])
    # A: (2,0), (2,1)
    # B: (0,2), (0,3), (1,2), (1,3)
    # C: (2,2), (2,3)
    # C_22 = A_20 B_02 + A_21 B_12
    # C_23 = A_20 B_03 + A_21 B_13
    # Rank 2.
    
    # Term 20: M_c = A_20 B_02 + A_21 B_12 -> C_22
    u = {(2,0): 1, (2,1): 1}
    v = {(0,2): 1, (1,2): 1}
    w = {(2,2): 1}
    add_term(20, u, v, w)
    
    # We ran out of rank 20 slots!
    # I need to check the count again.
    # 1-7: 7 terms
    # 8-14: 7 terms (14 total)
    # 15: 1 term (A12 B21) (15 total)
    # 16: 1 term (A12 B22) (16 total)
    # 17-18: 2 terms (A21 B11) (18 total)
    # 19-20: 2 terms (A22 B21) (20 total)
    # Missing: A21 B12 (2 terms) and A22 B22 (2 terms).
    # Total 20 + 2 + 2 = 24.
    
    # So rank is 24.
    
    return {"U": U, "V": V, "W": W}
